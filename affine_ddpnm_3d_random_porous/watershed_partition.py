"""Net-clearance watershed pore partition for the random-sphere 3-D medium.

Contrast partition to the Voronoi baseline in ``random_porous.py``:

* pores are basins of the clearance (distance-to-solid) function, selected
  as local maxima with persistence above explicit thresholds — the pore count
  is never prescribed to equal the number of solid spheres;
* throats are the watershed facet sets between neighbouring basins, split
  per connected component so that one label pair may carry several distinct
  interfaces;
* ``interface_pairs`` entries are pore-basin ids, not solid-sphere ids.

The solver pipeline consumes only mesh-level data (``facet_interface_ids``,
``interface_pairs/centers/normals/areas``), so this partition plugs into
``ddpnm_core`` unchanged; the Voronoi baseline is preserved untouched in
``random_porous.py`` (imported here as ``build_partition_voronoi``).

Boundary policy (documented design decision): the clearance function counts
every sphere wall and the four lateral cube walls (``y``/``z`` faces); the
inlet (``x=0``) and outlet (``x=1``) faces are open boundaries and are *not*
treated as solid, so edge pores are truncated by the physical openings rather
than by artificial walls.  Policy ``"cube_and_spheres"`` (all six faces) is
available for comparison, and ``"spheres"`` for analytic tests.

Persistence selection rule: a component is a marker (pore seed) when

    pers >= abs_threshold   and   pers >= rel_threshold * birth_value,

with the additional rule that the component born at the global maximum of the
clearance function is always a marker (it is the unique survivor of the
superlevel filtration on the connected fluid domain).  Low-persistence
components merge into their parent basin; no labels are left orphaned.

Determinism: plateau values (exactly equal clearances) are activated batch-
wise; within a batch, cells are processed in ascending index order and merges
resolve ties by (birth value, component id), so cell or edge permutations
never change the output.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gmsh
import numpy as np
from dolfinx import mesh as dmesh
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI

from random_porous import SPHERES, build_partition as build_partition_voronoi
from random_porous import sphere_clearance

# ---------------------------------------------------------------------------
# Boundary policies for the clearance function
# ---------------------------------------------------------------------------

POLICIES = ("spheres", "walls_and_spheres", "cube_and_spheres")


def clearance_from_policy(points: np.ndarray, policy: str = "walls_and_spheres") -> np.ndarray:
    """Clearance of *points* under the chosen boundary policy.

    ``spheres``            — only the sphere walls count;
    ``walls_and_spheres``  — spheres + the four lateral cube walls (default,
                             inlet/outlet are open boundaries);
    ``cube_and_spheres``   — spheres + all six cube faces.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    result = sphere_clearance(pts)
    if policy == "spheres":
        return result
    if policy == "walls_and_spheres":
        side = np.minimum(
            np.minimum(pts[:, 1], 1.0 - pts[:, 1]),
            np.minimum(pts[:, 2], 1.0 - pts[:, 2]),
        )
        return np.minimum(result, side)
    if policy == "cube_and_spheres":
        cube = np.minimum.reduce(
            [
                pts[:, 0], 1.0 - pts[:, 0],
                pts[:, 1], 1.0 - pts[:, 1],
                pts[:, 2], 1.0 - pts[:, 2],
            ]
        )
        return np.minimum(result, cube)
    raise ValueError(f"Unknown boundary policy {policy!r}; expected one of {POLICIES}.")


# ---------------------------------------------------------------------------
# Unpartitioned fluid mesh (no internal CAD cuts)
# ---------------------------------------------------------------------------

def _distance_threshold(
    surfaces: list[int], size_min: float, size_max: float, band: float
) -> int | None:
    if not surfaces:
        return None
    distance = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance, "SurfacesList", surfaces)
    gmsh.model.mesh.field.setNumber(distance, "Sampling", 120)
    threshold = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
    gmsh.model.mesh.field.setNumber(threshold, "SizeMin", size_min)
    gmsh.model.mesh.field.setNumber(threshold, "SizeMax", size_max)
    gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(threshold, "DistMax", band)
    return threshold


def generate_unpartitioned_mesh(
    bulk_size: float = 0.10,
    sphere_size: float = 0.045,
    boundary_size: float = 0.060,
    sphere_band: float = 0.12,
    boundary_band: float = 0.10,
    mesh_file: Path | None = None,
) -> dmesh.Mesh:
    """Generate the fluid-domain tetrahedral mesh with *no* internal cuts.

    The unit cube minus the 27 frozen spheres, fragmented once by OCC to
    obtain the conforming fluid volume, with boundary physical groups for
    sphere walls / inlet / outlet / outer walls and size fields near the
    solid surfaces.  No Voronoi or any other internal interface surface is
    embedded — pore labels come from the watershed on this mesh only.
    """
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("watershed_ddpnm_random_porous")
        box = gmsh.model.occ.addBox(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        grains = [
            gmsh.model.occ.addSphere(float(x), float(y), float(z), float(r))
            for x, y, z, r in SPHERES
        ]
        fluid, _ = gmsh.model.occ.cut(
            [(3, box)],
            [(3, tag) for tag in grains],
            removeObject=True,
            removeTool=True,
        )
        gmsh.model.occ.synchronize()

        volumes = [tag for _, tag in gmsh.model.getEntities(3)]
        if not volumes:
            raise RuntimeError("The fluid volume is empty after the sphere cut.")
        physical = gmsh.model.addPhysicalGroup(3, volumes, tag=1000)
        gmsh.model.setPhysicalName(3, physical, "fluid")

        sphere_surfaces: list[int] = []
        cube_surfaces: dict[str, list[int]] = {"inlet": [], "outlet": [], "outer": []}
        for _, surface in gmsh.model.getEntities(2):
            surface_type = gmsh.model.getType(2, surface).lower()
            if "sphere" in surface_type:
                sphere_surfaces.append(surface)
                continue
            if "plane" in surface_type:
                center = np.asarray(
                    gmsh.model.occ.getCenterOfMass(2, surface), dtype=float
                )
                if abs(center[0]) < 5.0e-7:
                    cube_surfaces["inlet"].append(surface)
                elif abs(center[0] - 1.0) < 5.0e-7:
                    cube_surfaces["outlet"].append(surface)
                else:
                    cube_surfaces["outer"].append(surface)
                continue
            raise RuntimeError(f"Unexpected boundary surface type {surface_type!r}.")

        for name, tag, surfaces in [
            ("sphere_walls", 1, sphere_surfaces),
            ("inlet", 2, cube_surfaces["inlet"]),
            ("outlet", 3, cube_surfaces["outlet"]),
            ("outer_walls", 4, cube_surfaces["outer"]),
        ]:
            if surfaces:
                group = gmsh.model.addPhysicalGroup(2, surfaces, tag=tag)
                gmsh.model.setPhysicalName(2, group, name)

        fields = [
            _distance_threshold(sphere_surfaces, sphere_size, bulk_size, sphere_band),
            _distance_threshold(
                cube_surfaces["inlet"]
                + cube_surfaces["outlet"]
                + cube_surfaces["outer"],
                boundary_size,
                bulk_size,
                boundary_band,
            ),
        ]
        active_fields = [field for field in fields if field is not None]
        minimum = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", active_fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(minimum)

        gmsh.option.setNumber("Mesh.MeshSizeMin", min(sphere_size, boundary_size))
        gmsh.option.setNumber("Mesh.MeshSizeMax", bulk_size)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.model.mesh.generate(3)
        if mesh_file is not None:
            mesh_file.parent.mkdir(parents=True, exist_ok=True)
            gmsh.write(str(mesh_file.resolve()))
        data = gmshio.model_to_mesh(gmsh.model, MPI.COMM_SELF, 0, gdim=3)
        return data.mesh
    finally:
        gmsh.finalize()


# ---------------------------------------------------------------------------
# Mesh topology helpers
# ---------------------------------------------------------------------------

def cell_centers(msh: dmesh.Mesh) -> np.ndarray:
    """Barycenters of all cells."""
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim, 0)
    c2v = msh.topology.connectivity(tdim, 0)
    n_cells = msh.topology.index_map(tdim).size_local
    centers = np.empty((n_cells, 3), dtype=float)
    for cell in range(n_cells):
        centers[cell] = msh.geometry.x[c2v.links(cell), :3].mean(axis=0)
    return centers


def cell_adjacency(msh: dmesh.Mesh) -> list[list[int]]:
    """Adjacency lists of the cell graph (shared-facet edges)."""
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    f2c = msh.topology.connectivity(fdim, tdim)
    n_cells = msh.topology.index_map(tdim).size_local
    adjacency: list[list[int]] = [[] for _ in range(n_cells)]
    n_facets = msh.topology.index_map(fdim).size_local
    for facet in range(n_facets):
        cells = f2c.links(facet)
        if len(cells) != 2:
            continue
        a, b = int(cells[0]), int(cells[1])
        adjacency[a].append(b)
        adjacency[b].append(a)
    for neighbors in adjacency:
        neighbors.sort()
    return adjacency


def compute_cell_clearance(
    msh: dmesh.Mesh, policy: str = "walls_and_spheres"
) -> np.ndarray:
    """Clearance at cell barycenters under the given boundary policy."""
    return clearance_from_policy(cell_centers(msh), policy)


def tetrahedron_volumes(points: np.ndarray, tetrahedra: np.ndarray) -> np.ndarray:
    xyz = points[tetrahedra]
    matrices = np.stack(
        (xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0], xyz[:, 3] - xyz[:, 0]),
        axis=1,
    )
    return np.abs(np.linalg.det(matrices)) / 6.0


# ---------------------------------------------------------------------------
# Superlevel merge tree and persistence
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MergeTreeComponent:
    comp_id: int
    birth_value: float
    birth_cell: int
    death_level: float  # merge (saddle) level; survivors: the minimum clearance
    parent: int         # surviving component id, -1 for the final survivor(s)
    persistence: float  # birth_value - death_level


def build_superlevel_merge_tree(
    clearance: np.ndarray, adjacency: list[list[int]]
) -> list[MergeTreeComponent]:
    """Superlevel merge tree of *clearance* on the cell graph.

    Cells are activated in descending clearance order; plateau batches of
    exactly equal values are activated together before any merge, so a flat
    top is one component, never a spurious family of maxima.  A merge at
    level ``lambda`` kills the component with the smaller birth value
    (ties by smaller component id) with persistence ``birth - lambda``.

    Returns one :class:`MergeTreeComponent` per created component; the final
    survivor(s) have ``parent == -1`` and ``death_level`` equal to the
    minimum clearance (they survive the whole filtration).
    """
    values = np.asarray(clearance, dtype=float)
    n = len(values)
    order = np.argsort(-values, kind="stable")

    # Plateau batches: contiguous runs of equal values in the descending order.
    batches: list[tuple[float, np.ndarray]] = []
    index = 0
    while index < n:
        value = values[order[index]]
        end = index
        while end < n and values[order[end]] == value:
            end += 1
        batches.append((float(value), order[index:end]))
        index = end

    uf = np.arange(n, dtype=np.int64)  # union-find over cell indices

    def find(x: int) -> int:
        root = x
        while uf[root] != root:
            root = int(uf[root])
        while uf[x] != root:
            uf[x] = root
            x = int(uf[x])
        return root

    active = np.zeros(n, dtype=bool)
    comp_root = np.full(n, -1, dtype=np.int64)   # uf root cell -> component id
    birth_cell = np.full(n, -1, dtype=np.int64)  # component id -> birth cell
    birth_value = np.full(n, np.inf, dtype=float)
    parent_comp = np.full(n, -1, dtype=np.int64)
    death_level = np.full(n, np.inf, dtype=float)
    n_components = 0

    for value, cells in batches:
        # 1. Activate the whole plateau batch before any merge.
        for cell in cells:
            cell = int(cell)
            active[cell] = True
            comp_root[cell] = n_components
            birth_cell[n_components] = cell
            birth_value[n_components] = value
            n_components += 1
        # 2. Merge every edge to an active neighbour (older components and
        #    within-batch components alike); results depend only on birth
        #    values and component ids, never on edge order.
        for cell in cells:
            cell = int(cell)
            for neighbour in adjacency[cell]:
                if not active[neighbour]:
                    continue
                root_a = find(cell)
                root_b = find(neighbour)
                if root_a == root_b:
                    continue
                comp_a = int(comp_root[root_a])
                comp_b = int(comp_root[root_b])
                birth_a = birth_value[comp_a]
                birth_b = birth_value[comp_b]
                if birth_a > birth_b or (birth_a == birth_b and comp_a < comp_b):
                    survive, die = comp_a, comp_b
                    keep_root, drop_root = root_a, root_b
                else:
                    survive, die = comp_b, comp_a
                    keep_root, drop_root = root_b, root_a
                death_level[die] = value
                parent_comp[die] = survive
                uf[drop_root] = keep_root
                comp_root[keep_root] = survive

    min_clearance = float(values.min()) if n else 0.0
    components: list[MergeTreeComponent] = []
    for comp_id in range(n_components):
        if parent_comp[comp_id] != -1:
            death = float(death_level[comp_id])
        else:
            death = min_clearance  # survivor: alive through the whole filtration
        components.append(
            MergeTreeComponent(
                comp_id=comp_id,
                birth_value=float(birth_value[comp_id]),
                birth_cell=int(birth_cell[comp_id]),
                death_level=death,
                parent=int(parent_comp[comp_id]),
                persistence=float(birth_value[comp_id] - death),
            )
        )
    return components


def select_persistent_maxima(
    components: list[MergeTreeComponent],
    abs_threshold: float = 0.02,
    rel_threshold: float = 0.05,
) -> np.ndarray:
    """Marker mask over components under the double persistence threshold.

    A component is a marker when ``pers >= abs_threshold`` and
    ``pers >= rel_threshold * birth``.  The component(s) born at the global
    maximum clearance (the final survivor(s) of the filtration) are always
    markers, guaranteeing every cell resolves to a marker.
    """
    markers = np.zeros(len(components), dtype=bool)
    for index, component in enumerate(components):
        if (
            component.persistence >= abs_threshold
            and component.persistence >= rel_threshold * component.birth_value
        ):
            markers[index] = True
    survivors = [c for c in components if c.parent == -1]
    if survivors:
        max_birth = max(c.birth_value for c in survivors)
        for c in survivors:
            if c.birth_value == max_birth:
                markers[c.comp_id] = True
    return markers


def watershed_labels(
    components: list[MergeTreeComponent],
    markers: np.ndarray,
    n_cells: int,
    clearance: np.ndarray,
    adjacency: list[list[int]],
) -> np.ndarray:
    """Basin labels by descending flood-fill from the marker seeds.

    Every marked component seeds its birth cell with its component id.
    Cells are then processed in descending clearance order (plateau batches
    of exactly equal values enter together) and each unlabelled cell takes
    the smallest marked component id among its already-labelled neighbours —
    the tie-break favours the strongest (earliest-born) basin.  Because a
    cell only ever inherits a label from an already-labelled neighbour, each
    basin grows by adjacency and is connected by construction (the defect of
    the previous nearest-marked-ancestor rule, which could strand a single
    cell between basins of a different label).

    Cells whose monotone path to any seed passes through a saddle below their
    own level (their merge-tree component chain contains no marked member)
    stay unlabelled after the flood; an ascending backfill propagates the
    same min-label rule bottom-up and repeats the sweep until exhausted,
    because a low cell can only be labelled once a neighbouring dip resolves.
    The domain is connected and the global-maximum component is always
    marked, so every cell is eventually labelled; a defensive check raises
    otherwise.

    Labels are remapped to ``0..k-1`` ordered by (birth value descending,
    component id ascending), so label 0 is the strongest basin.
    """
    values = np.asarray(clearance, dtype=float)
    order = np.argsort(-values, kind="stable")

    # Plateau batches: contiguous runs of equal values in the descending order.
    batches: list[np.ndarray] = []
    index = 0
    while index < n_cells:
        end = index
        while end < n_cells and values[order[end]] == values[order[index]]:
            end += 1
        batches.append(order[index:end])
        index = end

    comp_of_birth_cell = {
        component.birth_cell: component.comp_id for component in components
    }
    raw = np.full(n_cells, -1, dtype=np.int64)
    for component in components:
        if markers[component.comp_id]:
            raw[component.birth_cell] = component.comp_id

    def propagate_batch(batch: np.ndarray) -> bool:
        """Fixed point over a plateau batch; True if any cell was labelled.

        Cells are scanned in ascending cell order within a round; a cell
        labelled mid-round counts as a labelled neighbour for the rest of the
        round.  A round that makes no progress breaks, leaving the remaining
        cells of the batch to the backfill (spec: "one round without progress
        -> break").
        """
        pending = [int(cell) for cell in batch]
        changed = False
        while pending:
            progressed = False
            still: list[int] = []
            for cell in pending:
                if raw[cell] != -1:
                    continue  # seeds and previously labelled cells are final
                best = -1
                for neighbour in adjacency[cell]:
                    label = raw[neighbour]
                    if label != -1 and (best == -1 or label < best):
                        best = label
                if best != -1:
                    raw[cell] = best
                    changed = True
                    progressed = True
                else:
                    still.append(cell)
            if not progressed:
                break
            pending = still
        return changed

    # Descending flood: seeds are pre-assigned above, then each plateau batch
    # enters whole and propagates to its fixed point.
    for batch in batches:
        propagate_batch(batch)

    # Ascending backfill: repeat the sweep until a whole sweep makes no
    # progress (a single pass can stall a low cell whose only labelled
    # neighbour resolves only in a later batch).
    if np.any(raw == -1):
        n_remaining = int(np.count_nonzero(raw == -1))
        while n_remaining:
            progressed = False
            for batch in reversed(batches):
                progressed = propagate_batch(batch) or progressed
            if not progressed:
                raise RuntimeError(
                    f"watershed_labels: {n_remaining} cells could not be "
                    "resolved (domain disconnected or global maximum not "
                    "marked)."
                )
            n_remaining = int(np.count_nonzero(raw == -1))

    unique = sorted({int(label) for label in raw})
    unique.sort(key=lambda cid: (-components[cid].birth_value, components[cid].comp_id))
    mapping = {cid: i for i, cid in enumerate(unique)}
    labels = np.asarray([mapping[int(label)] for label in raw], dtype=np.int32)
    return labels


# ---------------------------------------------------------------------------
# Floating-basin merge (solid-contact post-process)
# ---------------------------------------------------------------------------

def merge_floating_basins(
    msh: dmesh.Mesh,
    labels: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Merge basins with no solid-wall facet into their largest neighbour.

    A basin whose boundary consists only of interface facets (and possibly
    the open inlet/outlet) has *no velocity Dirichlet boundary* in its local
    Stokes solve: the plain-splu pipeline factorises a matrix with a rigid-
    mode nullspace and the response columns explode (measured: |G| up to
    1e14 vs ~1e-2 for anchored basins), which corrupts the whole Schur
    system.  The classic DDPNM formulation treats regions that touch the
    solid, so each such floating basin is merged into the neighbour sharing
    the largest interface area (ties: smaller label id).  Merging strictly
    reduces the floating count (the receiver keeps its own wall facets), so
    the loop terminates; at least one basin always touches the solid because
    the fluid domain does.

    Returns ``(relabelled contiguous labels, record)`` where the record
    contains ``n_before``, ``n_after`` and the per-floating-basin host map.
    """
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    f2c = msh.topology.connectivity(fdim, tdim)
    msh.topology.create_connectivity(fdim, 0)
    f2v = msh.topology.connectivity(fdim, 0)
    n_labels = int(labels.max()) + 1

    exterior = dmesh.exterior_facet_indices(msh.topology)
    wall_count = np.zeros(n_labels, dtype=int)
    for facet in exterior:
        cells = f2c.links(int(facet))
        if len(cells) != 1:
            continue
        pore = int(labels[cells[0]])
        verts = msh.geometry.x[f2v.links(int(facet)), :3]
        midpoint = verts.mean(axis=0)
        if midpoint[0] <= 1.0e-8 or midpoint[0] >= 1.0 - 1.0e-8:
            continue  # open inlet/outlet: not a solid wall
        wall_count[pore] += 1

    # Shared interface area per label pair (internal facets only).
    pair_area: dict[tuple[int, int], float] = {}
    n_facets = msh.topology.index_map(fdim).size_local
    for facet in range(n_facets):
        cells = f2c.links(facet)
        if len(cells) != 2:
            continue
        a, b = int(labels[cells[0]]), int(labels[cells[1]])
        if a == b:
            continue
        triangle = msh.geometry.x[f2v.links(facet), :3]
        area = 0.5 * float(
            np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]))
        )
        key = (min(a, b), max(a, b))
        pair_area[key] = pair_area.get(key, 0.0) + area

    host_of: dict[int, int] = {}
    merged = np.full(n_labels, False, dtype=bool)
    n_merged = 0
    while True:
        floating = [
            label for label in range(n_labels)
            if not merged[label] and wall_count[label] == 0
        ]
        if not floating:
            break
        for label in floating:
            best_host = -1
            best_area = -1.0
            for (a, b), area in pair_area.items():
                if a == label:
                    other = b
                elif b == label:
                    other = a
                else:
                    continue
                if area > best_area or (area == best_area and other < best_host):
                    best_area = area
                    best_host = other
            if best_host < 0:
                raise RuntimeError(
                    f"Floating basin {label} has no neighbouring basin to merge into."
                )
            host_of[int(label)] = int(best_host)
            merged[int(label)] = True
            n_merged += 1
            labels = np.where(labels == label, best_host, labels)
        # After relabelling, a host that was floating may now touch the
        # solid only through its own facets (unchanged); re-scan below.
        if n_merged >= n_labels:
            break  # defensive: every basin merged is impossible for a
                   # connected fluid domain touching the solid

    if n_merged:
        unique = np.unique(labels)
        remap = {int(old): new for new, old in enumerate(sorted(unique))}
        labels = np.asarray([remap[int(label)] for label in labels], dtype=np.int32)
    return labels, {
        "n_before": n_labels,
        "n_after": int(labels.max()) + 1,
        "n_merged": n_merged,
        "host_of": host_of,
    }


# ---------------------------------------------------------------------------
# Interface extraction (per connected component)
# ---------------------------------------------------------------------------

def _group_facets_by_pair_and_component(
    facet_ids: np.ndarray,
    facet_vertex_keys: np.ndarray,
    facet_pair_keys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split *facet_ids* into edge-connected components per label pair.

    Parameters
    ----------
    facet_ids : int array of facet indices (uniqueness enforced by caller).
    facet_vertex_keys : (n, 4) int array, each row the sorted vertex ids of a
        facet (identical rows identify the same mesh edge set).
    facet_pair_keys : (n, 2) int array, each row the canonical label pair
        ``(min, max)`` of the facet's two adjacent cells.

    Returns
    -------
    component_ids : int array, one entry per facet, the component index
        within its label pair.
    n_components_per_pair : int array over label pairs (indexed by the pair
        order of first appearance), the number of components of each pair.
    """
    n_facets = len(facet_ids)
    # component_ids is indexed by facet position (parallel to the inputs).
    component_ids = np.full(n_facets, -1, dtype=np.int64)
    pair_order: dict[tuple[int, int], int] = {}
    pair_counts: list[int] = []

    for start in range(n_facets):
        if component_ids[start] != -1:
            continue
        pair = (int(facet_pair_keys[start, 0]), int(facet_pair_keys[start, 1]))
        if pair not in pair_order:
            pair_order[pair] = len(pair_counts)
            pair_counts.append(0)
        # Collect the whole pair group (facets may be interleaved).
        group = np.flatnonzero(
            (facet_pair_keys[:, 0] == pair[0]) & (facet_pair_keys[:, 1] == pair[1])
        )
        local_of: dict[int, int] = {int(p): i for i, p in enumerate(group)}
        # Edge adjacency within the group: two facets share an edge when
        # their vertex-key rows share a canonical (sorted) edge pair.  The
        # rows are in cyclic order, so the edges are the consecutive pairs
        # (wrap-around included); for triangles any order is cyclic.
        edge_map: dict[tuple[int, int], list[int]] = {}
        for position in group:
            row = facet_vertex_keys[int(position)]
            n_vertices = len(row)
            for k in range(n_vertices):
                edge = tuple(
                    sorted((int(row[k]), int(row[(k + 1) % n_vertices])))
                )
                edge_map.setdefault(edge, []).append(int(position))
        seen = np.zeros(len(group), dtype=bool)
        for local_index, position in enumerate(group):
            if seen[local_index]:
                continue
            # BFS over facets sharing edges, ascending facet order.
            queue = [int(position)]
            seen[local_index] = True
            component = pair_counts[pair_order[pair]]
            pair_counts[pair_order[pair]] += 1
            while queue:
                current = queue.pop()
                component_ids[current] = component
                row = facet_vertex_keys[current]
                n_vertices = len(row)
                for k in range(n_vertices):
                    edge = tuple(
                        sorted((int(row[k]), int(row[(k + 1) % n_vertices])))
                    )
                    for other in edge_map.get(edge, ()):
                        other_local = local_of[other]
                        if not seen[other_local]:
                            seen[other_local] = True
                            queue.append(other)
    return component_ids, np.asarray(pair_counts, dtype=np.int64)


def split_interface_components(
    msh: dmesh.Mesh, labels: np.ndarray
) -> tuple[tuple[tuple[int, int], ...], np.ndarray]:
    """Extract per-component interfaces from the basin labels.

    Every internal facet whose adjacent cells carry different labels belongs
    to an interface; facets of the same label pair are split into
    edge-connected components, each of which receives a distinct interface
    id (the assembler keys interfaces by id, so repeated label pairs are
    legitimate).  Ids are assigned deterministically: label pairs in sorted
    order, components in ascending facet order.
    """
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    f2c = msh.topology.connectivity(fdim, tdim)
    n_facets = msh.topology.index_map(fdim).size_local

    facet_ids: list[int] = []
    vertex_keys: list[tuple[int, ...]] = []
    pair_keys: list[tuple[int, int]] = []
    msh.topology.create_connectivity(fdim, 0)
    f2v = msh.topology.connectivity(fdim, 0)
    for facet in range(n_facets):
        cells = f2c.links(facet)
        if len(cells) != 2:
            continue
        a, b = int(labels[cells[0]]), int(labels[cells[1]])
        if a == b:
            continue
        # Cyclic vertex order: the component split canonicalises edges.
        vertices = tuple(int(v) for v in f2v.links(facet))
        facet_ids.append(facet)
        vertex_keys.append(vertices)
        pair_keys.append((min(a, b), max(a, b)))

    if not facet_ids:
        return (), np.full(n_facets, -1, dtype=np.int32)

    component_ids, _ = _group_facets_by_pair_and_component(
        np.asarray(facet_ids, dtype=np.int64),
        np.asarray(vertex_keys, dtype=np.int64),
        np.asarray(pair_keys, dtype=np.int64),
    )

    # Deterministic global ordering: pair (min, max) ascending, then
    # component id (components are already numbered by facet order).
    entries: list[tuple[tuple[int, int], int]] = sorted(
        zip(pair_keys, [int(c) for c in component_ids], strict=True)
    )
    merged: dict[tuple[tuple[int, int], int], int] = {}
    pairs: list[tuple[int, int]] = []
    for pair, component in entries:
        key = (pair, component)
        if key not in merged:
            merged[key] = len(pairs)
            pairs.append(pair)

    facet_interface_ids = np.full(n_facets, -1, dtype=np.int32)
    for local, global_facet in enumerate(facet_ids):
        facet_interface_ids[global_facet] = merged[(pair_keys[local], int(component_ids[local]))]
    return tuple(pairs), facet_interface_ids


# ---------------------------------------------------------------------------
# Interface geometry
# ---------------------------------------------------------------------------

def compute_interface_geometry(
    msh: dmesh.Mesh,
    facet_interface_ids: np.ndarray,
    pairs: tuple[tuple[int, int], ...],
    labels: np.ndarray,
    facet_centroid_clearance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-interface area, centroid, normal, normal dispersion, saddle value.

    Facet normals are oriented from the low label to the high label (the
    pair convention); the interface normal is the area-weighted average, the
    dispersion is ``(1/A) * sum_f A_f (1 - n_f . nbar)^2``, and the saddle
    value is the maximum clearance over the interface facet centroids
    (an estimate of the throat clearance).
    """
    fdim = msh.topology.dim - 1
    tdim = msh.topology.dim
    msh.topology.create_connectivity(fdim, 0)
    f2v = msh.topology.connectivity(fdim, 0)
    msh.topology.create_connectivity(fdim, tdim)
    f2c = msh.topology.connectivity(fdim, tdim)
    msh.topology.create_connectivity(tdim, 0)
    c2v = msh.topology.connectivity(tdim, 0)
    n_cells = msh.topology.index_map(tdim).size_local
    cell_barycenters = np.empty((n_cells, 3), dtype=float)
    for cell in range(n_cells):
        cell_barycenters[cell] = msh.geometry.x[c2v.links(cell), :3].mean(axis=0)

    n_interfaces = len(pairs)
    centers = np.zeros((n_interfaces, 3), dtype=float)
    normals = np.zeros((n_interfaces, 3), dtype=float)
    areas = np.zeros(n_interfaces, dtype=float)
    dispersion = np.zeros(n_interfaces, dtype=float)
    saddles = np.zeros(n_interfaces, dtype=float)

    for interface_id, (label_lo, label_hi) in enumerate(pairs):
        facets = np.flatnonzero(facet_interface_ids == interface_id)
        total_area = 0.0
        centroid = np.zeros(3, dtype=float)
        normal_sum = np.zeros(3, dtype=float)
        saddle = -np.inf
        facet_units: list[np.ndarray] = []
        for facet in facets:
            triangle = msh.geometry.x[f2v.links(int(facet)), :3]
            normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            area = 0.5 * float(np.linalg.norm(normal))
            if area <= 0.0:
                continue
            unit = normal / np.linalg.norm(normal)
            # Orient low label -> high label.
            cells = f2c.links(int(facet))
            direction = (
                cell_barycenters[int(cells[1])] - cell_barycenters[int(cells[0])]
            )
            if labels[int(cells[0])] != label_lo:
                direction = -direction
            if float(np.dot(unit, direction)) < 0.0:
                unit = -unit
            total_area += area
            centroid += area * triangle.mean(axis=0)
            normal_sum += area * unit
            facet_units.append(unit)
            saddle = max(saddle, float(facet_centroid_clearance[facet]))
        if total_area <= 0.0:
            raise RuntimeError(f"Interface {interface_id} has zero total area.")
        areas[interface_id] = total_area
        centers[interface_id] = centroid / total_area
        normal_sum /= total_area
        unit_normal = normal_sum / max(float(np.linalg.norm(normal_sum)), 1.0e-30)
        normals[interface_id] = unit_normal
        dispersion[interface_id] = (
            float(np.sum(1.0 - np.dot(facet_units, unit_normal))) / total_area
        )
        saddles[interface_id] = saddle
    return centers, normals, areas, dispersion, saddles


# ---------------------------------------------------------------------------
# Partition assembly
# ---------------------------------------------------------------------------

@dataclass
class WatershedPartitionData:
    mesh: dmesh.Mesh
    cell_centers: np.ndarray
    cell_clearance: np.ndarray
    cell_labels: np.ndarray
    maximal_balls: np.ndarray  # (n_pores, 4): center + radius (maximal balls)
    pore_peak_clearance: np.ndarray
    pore_persistence: np.ndarray
    interface_pairs: tuple[tuple[int, int], ...]
    facet_interface_ids: np.ndarray
    interface_centers: np.ndarray
    interface_normals: np.ndarray
    interface_areas: np.ndarray
    interface_normal_dispersion: np.ndarray
    interface_saddle_value: np.ndarray
    mesh_parameters: dict[str, float]
    cad_counts: dict[str, int]
    components: list[MergeTreeComponent] = None  # type: ignore[assignment]

    @property
    def pore_seeds(self) -> np.ndarray:
        return self.maximal_balls


def build_partition_watershed(
    bulk_size: float = 0.10,
    sphere_size: float = 0.045,
    boundary_size: float = 0.060,
    sphere_band: float = 0.12,
    boundary_band: float = 0.10,
    mesh_file: Path | None = None,
    policy: str = "walls_and_spheres",
    abs_threshold: float = 0.02,
    rel_threshold: float = 0.05,
) -> WatershedPartitionData:
    """Build the watershed pore partition on an unpartitioned fluid mesh.

    Pipeline: unpartitioned mesh -> cell clearance (policy) -> superlevel
    merge tree -> persistent maxima (markers) -> basin labels -> solid-
    contact merge -> per-component interfaces -> geometry.  The mesh is
    meshed once without any internal cut; labels and interfaces are pure
    mesh-level data.

    The solid-contact merge folds basins that touch no solid (no velocity
    Dirichlet boundary in their local Stokes solve) into their largest-area
    neighbour, keeping every region of the classic DDPNM kind; the merge is
    recorded in ``cad_counts``.
    """
    msh = generate_unpartitioned_mesh(
        bulk_size=bulk_size,
        sphere_size=sphere_size,
        boundary_size=boundary_size,
        sphere_band=sphere_band,
        boundary_band=boundary_band,
        mesh_file=mesh_file,
    )
    centers = cell_centers(msh)
    cell_clearance = clearance_from_policy(centers, policy)
    adjacency = cell_adjacency(msh)
    components = build_superlevel_merge_tree(cell_clearance, adjacency)
    markers = select_persistent_maxima(components, abs_threshold, rel_threshold)
    labels = watershed_labels(components, markers, len(centers), cell_clearance, adjacency)

    # Basins that touch no solid have no velocity Dirichlet boundary in the
    # local Stokes solve (rigid-mode nullspace); merge them into their
    # largest-area neighbour so every region is of the classic DDPNM kind.
    labels, merge_record = merge_floating_basins(msh, labels)

    pairs, facet_interface_ids = split_interface_components(msh, labels)

    # Saddle estimates from the clearance at interface facet barycentres.
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, 0)
    f2v = msh.topology.connectivity(fdim, 0)
    n_facets = msh.topology.index_map(fdim).size_local
    facet_barycenters = np.empty((n_facets, 3), dtype=float)
    for facet in range(n_facets):
        facet_barycenters[facet] = msh.geometry.x[f2v.links(facet), :3].mean(axis=0)
    facet_clearance = clearance_from_policy(facet_barycenters, policy)

    interface_centers, interface_normals, interface_areas, dispersion, saddles = (
        compute_interface_geometry(
            msh, facet_interface_ids, pairs, labels, facet_clearance
        )
    )

    n_pores = int(np.max(labels)) + 1
    maximal_balls = np.zeros((n_pores, 4), dtype=float)
    pore_peak = np.zeros(n_pores, dtype=float)
    pore_persistence = np.zeros(n_pores, dtype=float)
    comp_of_label: dict[int, int] = {}
    for label in range(n_pores):
        cells = np.flatnonzero(labels == label)
        peak = int(cells[np.argmax(cell_clearance[cells])])
        maximal_balls[label] = (*centers[peak], cell_clearance[peak])
        pore_peak[label] = cell_clearance[peak]
        # Persistence of the marker component: birth cell of the marker.
        for component in components:
            if component.birth_cell == peak:
                comp_of_label[label] = component.comp_id
                pore_persistence[label] = component.persistence
                break

    return WatershedPartitionData(
        mesh=msh,
        cell_centers=centers,
        cell_clearance=cell_clearance,
        cell_labels=labels,
        maximal_balls=maximal_balls,
        pore_peak_clearance=pore_peak,
        pore_persistence=pore_persistence,
        interface_pairs=pairs,
        facet_interface_ids=facet_interface_ids,
        interface_centers=interface_centers,
        interface_normals=interface_normals,
        interface_areas=interface_areas,
        interface_normal_dispersion=dispersion,
        interface_saddle_value=saddles,
        mesh_parameters={
            "bulk_size": bulk_size,
            "sphere_size": sphere_size,
            "boundary_size": boundary_size,
            "sphere_band": sphere_band,
            "boundary_band": boundary_band,
            "policy": policy,
            "abs_threshold": abs_threshold,
            "rel_threshold": rel_threshold,
        },
        cad_counts={
            "interface_surface_patches": int(len(pairs)),
            "n_pore_basins": n_pores,
            "floating_basins_merged": merge_record["n_merged"],
            "basins_before_merge": merge_record["n_before"],
            "floating_basin_hosts": merge_record["host_of"],
        },
        components=components,
    )


# ---------------------------------------------------------------------------
# Topology invariants (section 12.4 of the hand-off)
# ---------------------------------------------------------------------------

def check_topology_invariants(
    partition: WatershedPartitionData,
    cell_volumes: np.ndarray,
) -> dict[str, bool]:
    """Verify the 12.4 invariants; returns a name -> ok dictionary."""
    msh = partition.mesh
    labels = partition.cell_labels
    facet_interface_ids = partition.facet_interface_ids
    tdim = msh.topology.dim
    fdim = tdim - 1
    n_cells = msh.topology.index_map(tdim).size_local
    n_facets = msh.topology.index_map(fdim).size_local

    checks: dict[str, bool] = {}

    checks["every_cell_labeled"] = bool(
        len(labels) == n_cells and np.all(labels >= 0)
    )

    # Basins connected.
    adjacency = cell_adjacency(msh)
    basin_sizes = []
    for label in sorted(np.unique(labels)):
        cells = np.flatnonzero(labels == label)
        seen = np.zeros(n_cells, dtype=bool)
        queue = [int(cells[0])]
        seen[queue[0]] = True
        count = 0
        while queue:
            current = queue.pop()
            count += 1
            for neighbour in adjacency[current]:
                if labels[neighbour] == label and not seen[neighbour]:
                    seen[neighbour] = True
                    queue.append(neighbour)
        basin_sizes.append(count == len(cells))
    checks["every_basin_connected"] = bool(np.all(basin_sizes))

    # Internal interface facets: exactly two adjacent cells, different labels.
    msh.topology.create_connectivity(fdim, tdim)
    f2c = msh.topology.connectivity(fdim, tdim)
    interface_facets = np.flatnonzero(facet_interface_ids >= 0)
    two_cells = True
    different_labels = True
    for facet in interface_facets:
        cells = f2c.links(int(facet))
        if len(cells) != 2:
            two_cells = False
            break
        a, b = int(labels[cells[0]]), int(labels[cells[1]])
        if a == b:
            different_labels = False
            break
    checks["interface_facets_have_two_cells"] = two_cells
    checks["interface_facets_labels_differ"] = different_labels

    # Each interface id is one edge-connected component (split guarantees it;
    # re-check connectivity on the final ids).
    components_connected = True
    for interface_id in range(len(partition.interface_pairs)):
        facets = list(np.flatnonzero(facet_interface_ids == interface_id))
        if not facets:
            components_connected = False
            break
        msh.topology.create_connectivity(fdim, 0)
        f2v = msh.topology.connectivity(fdim, 0)

        def _facet_edges(facet: int) -> list[tuple[int, int]]:
            row = [int(v) for v in f2v.links(facet)]
            return [
                tuple(sorted((row[k], row[(k + 1) % len(row)])))
                for k in range(len(row))
            ]

        edge_map: dict[tuple[int, int], list[int]] = {}
        for facet in facets:
            for edge in _facet_edges(facet):
                edge_map.setdefault(edge, []).append(facet)
        seen = {facets[0]}
        queue = [facets[0]]
        while queue:
            current = queue.pop()
            for edge in _facet_edges(current):
                for other in edge_map.get(edge, ()):
                    if other not in seen:
                        seen.add(other)
                        queue.append(other)
        if len(seen) != len(facets):
            components_connected = False
            break
    checks["interface_components_edge_connected"] = components_connected

    # Interface edges terminate only at the exterior or at junctions with
    # other interface facets.
    exterior = dmesh.exterior_facet_indices(msh.topology)
    msh.topology.create_connectivity(fdim, 0)
    f2v = msh.topology.connectivity(fdim, 0)

    def _facet_edges(facet: int) -> list[tuple[int, int]]:
        row = [int(v) for v in f2v.links(facet)]
        return [
            tuple(sorted((row[k], row[(k + 1) % len(row)])))
            for k in range(len(row))
        ]

    exterior_edges: set[tuple[int, int]] = set()
    for facet in exterior:
        exterior_edges.update(_facet_edges(int(facet)))
    interface_edge_count: dict[tuple[int, int], int] = {}
    for facet in interface_facets:
        for edge in _facet_edges(int(facet)):
            interface_edge_count[edge] = interface_edge_count.get(edge, 0) + 1
    terminates_ok = True
    for edge, count in interface_edge_count.items():
        if edge not in exterior_edges and count < 2:
            terminates_ok = False
            break
    checks["interfaces_terminate_at_walls_or_junctions"] = terminates_ok

    # Pore volumes sum to the fluid volume (roundoff) and are positive.
    volume_by_label = np.zeros(int(np.max(labels)) + 1, dtype=float)
    np.add.at(volume_by_label, labels, cell_volumes)
    total = float(np.sum(cell_volumes))
    checks["pore_volumes_sum_to_fluid"] = bool(
        abs(float(np.sum(volume_by_label)) - total) <= 1.0e-9 * max(total, 1.0)
    )
    checks["interface_areas_positive"] = bool(np.all(partition.interface_areas > 0.0))

    # Cell barycentres for the aggregate orientation check below.
    msh.topology.create_connectivity(tdim, 0)
    c2v = msh.topology.connectivity(tdim, 0)
    cell_barycenters = np.empty((n_cells, 3), dtype=float)
    for cell in range(n_cells):
        cell_barycenters[cell] = msh.geometry.x[c2v.links(cell), :3].mean(axis=0)

    # Normal orientation is consistent with the low->high pair convention,
    # checked in the aggregate: for each interface, the area-weighted mean of
    # the adjacent cell barycentres on the high side minus that on the low
    # side must point along the interface normal.  A per-facet comparison
    # misfires on the staircase surfaces of discrete watersheds (riser facets
    # can be anti-parallel to the mean normal); the aggregate displacement
    # still points along the mean normal for any legitimate interface.
    orientation_ok = True
    for interface_id, (lo, hi) in enumerate(partition.interface_pairs):
        facets = np.flatnonzero(facet_interface_ids == interface_id)
        hi_mean = np.zeros(3, dtype=float)
        lo_mean = np.zeros(3, dtype=float)
        total_area = 0.0
        for facet in facets:
            cells = f2c.links(int(facet))
            triangle = msh.geometry.x[f2v.links(int(facet)), :3]
            area = 0.5 * float(
                np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]))
            )
            if int(labels[cells[0]]) == lo:
                hi_mean += area * cell_barycenters[int(cells[1])]
                lo_mean += area * cell_barycenters[int(cells[0])]
            else:
                hi_mean += area * cell_barycenters[int(cells[0])]
                lo_mean += area * cell_barycenters[int(cells[1])]
            total_area += area
        if total_area <= 0.0:
            orientation_ok = False
            break
        direction = (hi_mean - lo_mean) / total_area
        if float(np.dot(direction, partition.interface_normals[interface_id])) <= 0.0:
            orientation_ok = False
            break
    checks["normals_oriented_low_to_high"] = orientation_ok

    return checks

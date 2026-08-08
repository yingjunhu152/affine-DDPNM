"""Unit tests for the watershed pore partition (hand-off sections 12.1-12.3).

The merge-tree, persistence and labelling algorithms are graph-agnostic
(they consume ``(clearance, adjacency)``), so the analytic three-sphere test
runs on a regular-grid sampling of the clearance function — the same code
path that the tetrahedral smoke exercises on the real mesh.

Run with the fenicsx environment:
    export PATH="/d/Miniconda3/envs/fenicsx/Library/bin:$PATH"
    python test_watershed_partition.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPOSITORY_DIR = Path(__file__).resolve().parent.parent
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from watershed_partition import (
    _group_facets_by_pair_and_component,
    build_superlevel_merge_tree,
    select_persistent_maxima,
    watershed_labels,
)


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------

def _grid_adjacency(shape: tuple[int, int, int]) -> list[list[int]]:
    """6-neighbour adjacency of a regular grid of ``shape`` cells."""
    size = int(np.prod(shape))
    cells = np.arange(size).reshape(shape)
    adjacency: list[set[int]] = [set() for _ in range(size)]
    for index in np.ndindex(shape):
        cell = int(cells[index])
        for axis in range(3):
            if index[axis] + 1 < shape[axis]:
                nxt = list(index)
                nxt[axis] += 1
                other = int(cells[tuple(nxt)])
                adjacency[cell].add(other)
                adjacency[other].add(cell)
    return [sorted(neighbours) for neighbours in adjacency]


def _permute_graph(clearance: np.ndarray, adjacency: list[list[int]]):
    """Return a randomly permuted copy of the graph (deterministic seed)."""
    rng = np.random.default_rng(12345)
    n = len(clearance)
    perm = rng.permutation(n)
    inverse = np.empty(n, dtype=np.int64)
    inverse[perm] = np.arange(n)
    permuted_adjacency = [
        sorted(
            int(inverse[neighbour])
            for neighbour in adjacency[int(perm[cell])]
        )
        for cell in range(n)
    ]
    return clearance[perm], permuted_adjacency, perm, inverse


# ---------------------------------------------------------------------------
# 12.1 Three equal spheres: analytic saddle recovery
# ---------------------------------------------------------------------------

def _three_sphere_clearance(points: np.ndarray, centers: np.ndarray, r: float) -> np.ndarray:
    """Clearance = min over the three spheres and the six cube faces."""
    distances = np.linalg.norm(
        points[:, None, :] - centers[None, :, :], axis=2
    )
    sphere = np.min(distances, axis=1) - r
    cube = np.minimum.reduce(
        [
            points[:, 0], 1.0 - points[:, 0],
            points[:, 1], 1.0 - points[:, 1],
            points[:, 2], 1.0 - points[:, 2],
        ]
    )
    return np.minimum(sphere, cube)


def _three_sphere_cap_setup():
    """Three equal spheres whose two cap maxima bracket the throat saddle.

    Three spheres (r = 0.09) with centres in the plane z = 0.5 forming a
    non-degenerate triangle with circumradius R_c = 0.15 and circumcentre
    o = (0.5, 0.5, 0.5), in a tight box x, y in [0.26, 0.74], z in [0, 1].
    The box hugs the spheres (c1 = (0.5, 0.65, 0.5) touches y = 0.74), so the
    corner pockets have clearance at most ~0.17 while the two caps — the
    clearance maxima on the three-sphere medial axis {o + z e_z} between the
    spheres and the top/bottom walls — reach d_cap = 0.2241 at z = 0.7759
    and z = 0.2241.  q(z) = sqrt(R_c^2 + z^2) - r has a minimum on the axis
    at o with q(o) = R_c - r = 0.06: the throat saddle.  Its stable manifold
    is the cross-section plane z = 0.5 perpendicular to the medial tangent
    e_z, which must be the watershed interface between the two cap basins.
    """
    r = 0.09
    R_c = 0.15
    half = R_c * np.sqrt(3.0) / 2.0
    offset = R_c / 2.0
    centers = np.asarray(
        [
            (0.5, 0.5 + R_c, 0.5),
            (0.5 - half, 0.5 - offset, 0.5),
            (0.5 + half, 0.5 - offset, 0.5),
        ],
        dtype=float,
    )
    box = (0.26, 0.74, 0.26, 0.74, 0.0, 1.0)
    return r, R_c, centers, box


def _box_clearance(
    points: np.ndarray, centers: np.ndarray, r: float, box: tuple[float, ...]
) -> np.ndarray:
    """Clearance = min over the three spheres and the six box faces."""
    distances = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
    sphere = np.min(distances, axis=1) - r
    lo0, hi0, lo1, hi1, lo2, hi2 = box
    walls = np.minimum.reduce(
        [
            points[:, 0] - lo0, hi0 - points[:, 0],
            points[:, 1] - lo1, hi1 - points[:, 1],
            points[:, 2] - lo2, hi2 - points[:, 2],
        ]
    )
    return np.minimum(sphere, walls)


def test_three_sphere_saddle() -> None:
    """Recover the throat saddle at the circumcentre of three equal spheres."""
    r, R_c, centers, box = _three_sphere_cap_setup()
    q_saddle = R_c - r
    d_cap = 0.2241  # balance sqrt(R_c^2 + t^2) - r = 0.5 - t at t = 0.2759
    side = np.linalg.norm(centers[1] - centers[2])
    assert abs(side - np.sqrt(3.0) * R_c) < 1.0e-12

    nx, ny, nz = 40, 40, 80
    h = 0.0125
    axis_x = np.linspace(box[0] + h / 2, box[1] - h / 2, nx)
    axis_y = np.linspace(box[2] + h / 2, box[3] - h / 2, ny)
    axis_z = np.linspace(box[4] + h / 2, box[5] - h / 2, nz)
    xx, yy, zz = np.meshgrid(axis_x, axis_y, axis_z, indexing="ij")
    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    clearance = _box_clearance(points, centers, r, box)

    # Fluid cells only (the real mesh is the fluid domain; in-sphere cells
    # carry negative clearance and are excluded exactly like the gmsh mesh).
    fluid = np.flatnonzero(clearance >= 0.0)
    fluid_pos = {int(cell): i for i, cell in enumerate(fluid)}
    full_adjacency = _grid_adjacency((nx, ny, nz))
    adjacency = [
        sorted(int(fluid_pos[n]) for n in full_adjacency[int(cell)] if n in fluid_pos)
        for cell in fluid
    ]
    fluid_clearance = clearance[fluid]

    components = build_superlevel_merge_tree(fluid_clearance, adjacency)
    markers = select_persistent_maxima(components, abs_threshold=0.005, rel_threshold=0.03)
    labels = watershed_labels(components, markers, len(fluid), fluid_clearance, adjacency)
    n_basins = int(labels.max()) + 1
    assert n_basins >= 2

    # Flood labelling keeps every basin connected (the defect it fixes:
    # chain-based labelling stranded single cells between other basins).
    for label in range(n_basins):
        cells = np.flatnonzero(labels == label)
        seen = np.zeros(len(labels), dtype=bool)
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
        assert count == len(cells), f"basin {label} is disconnected"

    def nearest(point: np.ndarray) -> int:
        return int(np.argmin(np.sum((points[fluid] - point) ** 2, axis=1)))

    # Saddle value at the circumcentre (1-Lipschitz grid error <= sqrt(3) h).
    o_cell = nearest(np.asarray([0.5, 0.5, 0.5]))
    o_clearance = fluid_clearance[o_cell]
    assert abs(o_clearance - q_saddle) <= 3.0 * h, (
        f"clearance at circumcentre {o_clearance:.4f} != {q_saddle:.4f}"
    )

    # Medial-axis profile: q(z) grows away from the saddle along e_z.
    for z_target in (0.60, 0.65, 0.70, 0.75):
        branch_cell = nearest(np.asarray([0.5, 0.5, z_target]))
        q = np.sqrt(R_c**2 + (z_target - 0.5) ** 2) - r
        assert abs(fluid_clearance[branch_cell] - q) <= 3.0 * h, (
            f"branch profile at z={z_target}: {fluid_clearance[branch_cell]:.4f} "
            f"vs analytic {q:.4f}"
        )

    # The two cap maxima bracket the saddle on the medial axis.
    top_cell = nearest(np.asarray([0.5, 0.5, 0.7759]))
    bottom_cell = nearest(np.asarray([0.5, 0.5, 0.2241]))
    d_top, d_bottom = fluid_clearance[top_cell], fluid_clearance[bottom_cell]
    assert abs(d_top - d_cap) <= 3.0 * h and abs(d_bottom - d_cap) <= 3.0 * h, (
        f"cap maxima {d_top:.4f}, {d_bottom:.4f} vs analytic {d_cap:.4f}"
    )
    top_label = int(labels[top_cell])
    bottom_label = int(labels[bottom_cell])
    assert top_label != bottom_label, (
        "the two cap maxima must be separate basins (throat saddle between them)"
    )

    # The cap-cap interface is the discrete watershed surface approximating
    # the cross-section plane z = 0.5 through o, perpendicular to the medial
    # tangent e_z.  On a coarse grid the surface is a staircase whose level
    # sits near the merge level of the two basins, so the recovered structure
    # is asserted within a few grid spacings of the saddle (it converges to
    # the saddle plane as h -> 0).
    from collections import defaultdict

    pair_facets: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray]]] = (
        defaultdict(list)
    )
    cell_shape = (nx, ny, nz)
    grid_cells = np.arange(np.prod(cell_shape)).reshape(cell_shape)
    for index in np.ndindex(cell_shape):
        cell = int(grid_cells[index])
        if cell not in fluid_pos:
            continue
        local = fluid_pos[cell]
        for axis in range(3):
            if index[axis] + 1 < cell_shape[axis]:
                nxt = list(index)
                nxt[axis] += 1
                other = int(grid_cells[tuple(nxt)])
                if other not in fluid_pos:
                    continue
                la, lb = int(labels[local]), int(labels[fluid_pos[other]])
                if la == lb:
                    continue
                pair = (min(la, lb), max(la, lb))
                unit = np.zeros(3)
                unit[axis] = 1.0
                facet_point = 0.5 * (points[cell] + points[other])
                pair_facets[pair].append((facet_point, unit))

    pair = (min(top_label, bottom_label), max(top_label, bottom_label))
    assert pair in pair_facets, "no interface between the two cap basins"
    facets = pair_facets[pair]
    midpoints = np.asarray([point for point, _ in facets])
    saddle_estimate = float(np.max(_box_clearance(midpoints, centers, r, box)))
    assert abs(saddle_estimate - q_saddle) <= 4.0 * h, (
        f"interface saddle value {saddle_estimate:.4f} != {q_saddle:.4f}"
    )

    # The interface crosses the medial axis near the saddle: it has a facet
    # within a few spacings of the branch line x = 0.5, y = 0.5, at z within
    # ten spacings of the cross-section plane z = 0.5, and that crossing is
    # transverse to the axis (its normal is close to e_z).
    branch_line = np.asarray([0.5, 0.5])
    distances = np.linalg.norm(midpoints[:, :2] - branch_line, axis=1)
    index_nearest = int(np.argmin(distances))
    assert distances[index_nearest] <= 3.0 * h, (
        f"cap-cap interface misses the medial axis: {distances[index_nearest]:.4f}"
    )
    crossing = facets[index_nearest][1]
    assert abs(midpoints[index_nearest, 2] - 0.5) <= 10.0 * h, (
        f"interface crossing at z = {midpoints[index_nearest, 2]:.4f}, "
        f"far from the cross-section plane z = 0.5"
    )
    assert abs(crossing[2]) >= np.cos(np.deg2rad(45.0)), (
        f"interface not transverse to the medial axis at the crossing: {crossing}"
    )

    print(f"   three-sphere: {n_basins} basins, saddle {o_clearance:.4f} "
          f"(analytic {q_saddle:.4f}), caps {d_top:.4f}/{d_bottom:.4f}, "
          f"interface crosses the axis at z = {midpoints[index_nearest, 2]:.4f}, "
          f"saddle estimate {saddle_estimate:.4f}")


# ---------------------------------------------------------------------------
# 12.2 Plateau batch processing and permutation invariance
# ---------------------------------------------------------------------------

def test_plateau_batch_processing() -> None:
    """Adjacent equal maxima form one basin; permutations change nothing."""
    # Two adjacent maxima at 2.0 on a 2-D grid (rendered as a flat 3-D slab).
    shape = (6, 6, 1)
    n = int(np.prod(shape))
    cells = np.arange(n).reshape(shape)
    clearance = np.full(n, 1.0)
    clearance[cells[1:5, 1:5, 0]] = 1.5
    clearance[cells[2, 2, 0]] = 2.0
    clearance[cells[3, 2, 0]] = 2.0  # adjacent plateau partner
    adjacency = _grid_adjacency(shape)

    def run(values, adj):
        components = build_superlevel_merge_tree(values, adj)
        markers = select_persistent_maxima(components, abs_threshold=0.1, rel_threshold=0.05)
        return watershed_labels(components, markers, len(values), values, adj), components

    labels, components = run(clearance, adjacency)
    # One basin from the plateau pair: the partners merge at their own level
    # (flat top = one component), so the dying partner has persistence 0.0
    # and the survivor persists to the minimum clearance (2.0 - 1.0 = 1.0).
    assert int(labels.max()) + 1 == 1
    plateau_cells = {int(cells[2, 2, 0]), int(cells[3, 2, 0])}
    assert set(int(c) for c in np.flatnonzero(labels == 0)) >= plateau_cells
    persistent = [c for c in components if c.birth_value == 2.0]
    assert len(persistent) == 2
    dying = [c for c in persistent if c.parent != -1]
    survivors = [c for c in persistent if c.parent == -1]
    assert len(survivors) == 1 and len(dying) == 1
    assert abs(dying[0].persistence) < 1.0e-12  # merged at its own plateau level
    assert abs(survivors[0].persistence - 1.0) < 1.0e-12

    # Permutation invariance: shuffle cells and rebuild; the basin count,
    # persistence values and basin cell sets (in original ids) must match.
    permuted, permuted_adj, perm, inverse = _permute_graph(clearance, adjacency)
    labels_p, components_p = run(permuted, permuted_adj)
    assert int(labels_p.max()) + 1 == 1
    original_of = {int(perm[cell]): cell for cell in range(n)}
    plateau_original = {original_of[c] for c in plateau_cells}
    basin_p = set(int(original_of[c]) for c in np.flatnonzero(labels_p == 0))
    assert plateau_original <= basin_p
    pers_p = sorted(c.persistence for c in components_p)
    pers_orig = sorted(c.persistence for c in components)
    assert np.allclose(pers_p, pers_orig, atol=1.0e-12)

    # Two separate maxima: two basins at low threshold, one at high threshold.
    shape2 = (6, 6, 1)
    cells2 = np.arange(int(np.prod(shape2))).reshape(shape2)
    clearance2 = np.full(int(np.prod(shape2)), 0.5)
    clearance2[cells2[0:3, 0:3, 0]] = 1.0
    clearance2[cells2[3:6, 3:6, 0]] = 1.0
    clearance2[cells2[1, 1, 0]] = 2.0
    clearance2[cells2[4, 4, 0]] = 2.0
    adjacency2 = _grid_adjacency(shape2)
    components2 = build_superlevel_merge_tree(clearance2, adjacency2)
    markers_low = select_persistent_maxima(components2, abs_threshold=0.2, rel_threshold=0.05)
    labels_low = watershed_labels(components2, markers_low, len(clearance2), clearance2, adjacency2)
    assert int(labels_low.max()) + 1 == 2
    # Persistence of each maximum is 2.0 - 0.5 = 1.5: the two 3x3 blocks
    # touch only diagonally at level 1.0, so the merge happens at the outer
    # ring level 0.5.  A threshold above 1.5 rejects both; the global-max
    # rule then keeps exactly one basin.
    markers_high = select_persistent_maxima(components2, abs_threshold=2.0, rel_threshold=0.05)
    labels_high = watershed_labels(components2, markers_high, len(clearance2), clearance2, adjacency2)
    assert int(labels_high.max()) + 1 == 1  # global-max rule keeps one basin
    assert len(markers_high) == len(components2)
    assert int(markers_high.sum()) == 1

    print("   plateau: 1 basin from adjacent maxima, permutations invariant, "
          "2 -> 1 basins under high threshold")


# ---------------------------------------------------------------------------
# 12.3 Interface components: one label pair, two disconnected interfaces
# ---------------------------------------------------------------------------

def _facet_rows(start: int, count: int) -> np.ndarray:
    return np.arange(start, start + count, dtype=np.int64)


def test_interface_components() -> None:
    """Two disconnected facet patches of the same label pair split into two
    interface components."""
    # Lattice of corner ids for a 4 x 4 x 1 slab of unit cubes:
    # corner (i, j, k) -> id i + 4 j + 16 k.
    def corner(i: int, j: int, k: int) -> int:
        return i + 4 * j + 16 * k

    # Block A: cubes (0,0), (1,0) carry label 1; block B: cubes (2,3), (3,3)
    # carry label 1; everything else label 0.  Facets between labels (0,1):
    #  - block A faces at y=0 (cubes (0,0),(1,0)): two facets sharing an edge;
    #  - block A faces at x=2 (cube (1,0)) and block B faces at x=2 (cube (2,3))...
    # To keep the two patches clearly disconnected we use only the y=0 faces
    # of block A (patch A) and the y=4 faces of block B (patch B).
    def face_y0(i: int) -> tuple[int, int, int, int]:
        # Cyclic order of the y = 0 face of cube (i, 0): (i,0,0) -> (i+1,0,0)
        # -> (i+1,0,1) -> (i,0,1).
        return (corner(i, 0, 0), corner(i + 1, 0, 0),
                corner(i + 1, 0, 1), corner(i, 0, 1))

    def face_y4(i: int) -> tuple[int, int, int, int]:
        return (corner(i, 4, 0), corner(i + 1, 4, 0),
                corner(i + 1, 4, 1), corner(i, 4, 1))

    patch_a = [face_y0(0), face_y0(1)]  # two adjacent y=0 faces
    patch_b = [face_y4(2), face_y4(3)]  # two adjacent y=4 faces
    facet_ids = _facet_rows(100, len(patch_a) + len(patch_b))
    # Cyclic vertex order (edges are consecutive pairs, wrap-around included).
    vertex_keys = np.asarray(patch_a + patch_b, dtype=np.int64)
    pair_keys = np.asarray([(0, 1)] * len(facet_ids), dtype=np.int64)

    component_ids, counts = _group_facets_by_pair_and_component(
        facet_ids, vertex_keys, pair_keys
    )
    assert counts.tolist() == [2]
    assert sorted(int(c) for c in component_ids) == [0, 0, 1, 1]
    # Patch A (first two facets) and patch B (last two) in separate components.
    assert component_ids[0] == component_ids[1]
    assert component_ids[2] == component_ids[3]
    assert component_ids[0] != component_ids[2]

    # Determinism under facet order permutations.
    order = np.asarray([2, 0, 3, 1], dtype=np.int64)
    component_ids_p, counts_p = _group_facets_by_pair_and_component(
        facet_ids[order], vertex_keys[order], pair_keys[order]
    )
    assert counts_p.tolist() == [2]
    # Component ids are assigned in facet order; the multiset is invariant.
    assert sorted(int(c) for c in component_ids_p) == [0, 0, 1, 1]

    # Interleaved pairs: (0,1) facets (0, 2) and (1,2) facets (1, 3) mixed
    # in the input; each pair contributes two disconnected components.
    pair_keys_mixed = np.asarray(
        [(0, 1), (1, 2), (0, 1), (1, 2)], dtype=np.int64
    )
    component_ids_m, counts_m = _group_facets_by_pair_and_component(
        facet_ids, vertex_keys, pair_keys_mixed
    )
    assert counts_m.tolist() == [2, 2]
    assert component_ids_m[0] != component_ids_m[2]  # (0,1): two components
    assert component_ids_m[1] != component_ids_m[3]  # (1,2): two components

    print("   interface components: two patches of pair (0,1) split into two "
          "components, permutation and interleaving invariant")


def main() -> None:
    print("test_watershed_partition:")
    test_three_sphere_saddle()
    test_plateau_batch_processing()
    test_interface_components()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Floating-basin merge tests (mesh-based, no gmsh)
# ---------------------------------------------------------------------------

def test_merge_floating_basins():
    """A basin with no solid-wall facet is merged into its neighbour."""
    from dolfinx import mesh as dmesh
    from mpi4py import MPI
    from watershed_partition import merge_floating_basins

    msh = dmesh.create_unit_cube(
        MPI.COMM_SELF, 4, 4, 4, dmesh.CellType.tetrahedron
    )
    n_cells = msh.topology.index_map(msh.topology.dim).size_local

    # Interior cell: all its facets are internal (not on the cube boundary).
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(tdim, 0)
    msh.topology.create_connectivity(fdim, tdim)
    c2v = msh.topology.connectivity(tdim, 0)
    f2c = msh.topology.connectivity(fdim, tdim)
    exterior = dmesh.exterior_facet_indices(msh.topology)
    exterior_cells = {int(cells[0]) for facet in exterior
                      for cells in (f2c.links(int(facet)),) if len(cells) == 1}
    interior = [cell for cell in range(n_cells) if cell not in exterior_cells]
    assert interior, "4x4x4 cube should contain interior tets"

    labels = np.full(n_cells, 1, dtype=np.int32)
    labels[interior[0]] = 0  # floating basin: single interior cell
    merged, record = merge_floating_basins(msh, labels)
    assert record["n_merged"] == 1, record
    assert record["n_before"] == 2
    assert record["n_after"] == 1
    assert int(record["host_of"][0]) == 1
    assert np.all(merged == 1), "floating basin must be relabelled to its host"


def test_merge_keeps_wall_contacting_basins():
    """A basin touching the solid is never merged."""
    from dolfinx import mesh as dmesh
    from mpi4py import MPI
    from watershed_partition import merge_floating_basins

    msh = dmesh.create_unit_cube(
        MPI.COMM_SELF, 4, 4, 4, dmesh.CellType.tetrahedron
    )
    n_cells = msh.topology.index_map(msh.topology.dim).size_local
    labels = np.full(n_cells, 1, dtype=np.int32)
    labels[0] = 0  # corner cell: touches three cube walls
    merged, record = merge_floating_basins(msh, labels)
    assert record["n_merged"] == 0, record
    assert record["n_before"] == record["n_after"] == 2
    assert int(np.unique(merged).size) == 2

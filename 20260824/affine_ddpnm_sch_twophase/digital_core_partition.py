"""Partition adapter for watershed-labelled digital-core tetrahedral meshes.

Unlike ``random_porous``, this module derives every internal interface from
two watershed labels sharing an actual tetrahedral facet.  No sphere model,
analytic throat, or Voronoi construction is involved.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dolfinx import mesh as dmesh
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI


@dataclass
class DigitalCorePartition:
    mesh: dmesh.Mesh
    cell_centers: np.ndarray
    cell_clearance: np.ndarray
    cell_labels: np.ndarray
    maximal_balls: tuple = ()
    throats: tuple = ()
    interface_pairs: tuple[tuple[int, int], ...] = ()
    facet_interface_ids: np.ndarray | None = None
    interface_centers: np.ndarray | None = None
    interface_normals: np.ndarray | None = None
    interface_areas: np.ndarray | None = None
    mesh_parameters: dict | None = None
    cad_counts: dict | None = None

    @property
    def pore_seeds(self) -> np.ndarray:
        labels = sorted(int(value) for value in np.unique(self.cell_labels))
        return np.asarray(
            [(*self.cell_centers[self.cell_labels == label].mean(axis=0), 0.0) for label in labels],
            dtype=float,
        )


def _cell_centers(msh: dmesh.Mesh) -> np.ndarray:
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim, 0)
    c2v = msh.topology.connectivity(tdim, 0)
    count = msh.topology.index_map(tdim).size_local
    return np.asarray(
        [msh.geometry.x[c2v.links(cell), :3].mean(axis=0) for cell in range(count)],
        dtype=float,
    )


def build_digital_core_partition(
    mesh_path: Path, *, split_disconnected_interfaces: bool = True,
) -> DigitalCorePartition:
    """Read a labelled pore mesh and derive its real interfaces.

    ``split_disconnected_interfaces=False`` is intended for Cartesian domain
    decompositions.  In that case, solid obstacles may perforate one planar
    interface into several disconnected facet patches, but those patches are
    still one physical interface between the same two subdomains.
    """
    data = gmshio.read_from_msh(str(mesh_path), MPI.COMM_SELF, gdim=3)
    if data.cell_tags is None:
        raise RuntimeError("digital-core mesh has no tetrahedral watershed labels")
    msh = data.mesh
    tdim, fdim = msh.topology.dim, msh.topology.dim - 1
    n_cells = msh.topology.index_map(tdim).size_local
    labels = np.zeros(n_cells, dtype=np.int32)
    labels[np.asarray(data.cell_tags.indices, dtype=np.int32)] = np.asarray(data.cell_tags.values, dtype=np.int32)
    if np.any(labels <= 0):
        raise RuntimeError("every tetrahedron must have a positive watershed pore label")
    centers = _cell_centers(msh)
    msh.topology.create_connectivity(fdim, tdim)
    msh.topology.create_connectivity(fdim, 0)
    f2c = msh.topology.connectivity(fdim, tdim)
    f2v = msh.topology.connectivity(fdim, 0)
    n_facets = msh.topology.index_map(fdim).size_local
    pair_for_facet: list[tuple[int, int] | None] = [None] * n_facets
    for facet in range(n_facets):
        cells = f2c.links(facet)
        if len(cells) != 2:
            continue
        a, b = int(labels[cells[0]]), int(labels[cells[1]])
        if a != b:
            pair = (min(a, b), max(a, b))
            pair_for_facet[facet] = pair
    # A pore pair can meet through multiple disconnected patches.  They must
    # remain different DDPNM interfaces: one affine coordinate frame spanning
    # disconnected patches would introduce artificial (and often null) modes.
    # Facets are connected only when they share a full edge, never merely a
    # voxel corner.  This is the same conservative face-connectivity rule as
    # the watershed segmentation itself.
    facets_by_pair: dict[tuple[int, int], list[int]] = {}
    for facet, pair in enumerate(pair_for_facet):
        if pair is not None:
            facets_by_pair.setdefault(pair, []).append(facet)
    components: list[tuple[tuple[int, int], tuple[int, ...]]] = []
    if not split_disconnected_interfaces:
        components = [
            (pair, tuple(sorted(facets)))
            for pair, facets in sorted(facets_by_pair.items())
        ]
    else:
        for pair, facets in sorted(facets_by_pair.items()):
            edge_to_facets: dict[tuple[int, int], list[int]] = {}
            for facet in facets:
                vertices = [int(v) for v in f2v.links(facet)]
                for a, b in ((vertices[0], vertices[1]), (vertices[1], vertices[2]), (vertices[0], vertices[2])):
                    edge_to_facets.setdefault(tuple(sorted((a, b))), []).append(facet)
            unseen = set(facets)
            while unseen:
                seed = unseen.pop()
                component = {seed}
                stack = [seed]
                while stack:
                    current = stack.pop()
                    vertices = [int(v) for v in f2v.links(current)]
                    neighbours: set[int] = set()
                    for a, b in ((vertices[0], vertices[1]), (vertices[1], vertices[2]), (vertices[0], vertices[2])):
                        neighbours.update(edge_to_facets[tuple(sorted((a, b)))])
                    for neighbour in neighbours & unseen:
                        unseen.remove(neighbour)
                        component.add(neighbour)
                        stack.append(neighbour)
                components.append((pair, tuple(sorted(component))))
    interface_pairs = tuple(pair for pair, _ in components)
    facet_ids = np.full(n_facets, -1, dtype=np.int32)
    areas = np.zeros(len(interface_pairs), dtype=float)
    weighted_centers = np.zeros((len(interface_pairs), 3), dtype=float)
    weighted_normals = np.zeros((len(interface_pairs), 3), dtype=float)
    for interface_id, (pair, facets) in enumerate(components):
        for facet in facets:
            facet_ids[facet] = interface_id
            vertices = msh.geometry.x[f2v.links(facet), :3]
            normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
            area = 0.5 * float(np.linalg.norm(normal))
            if area <= 0.0:
                raise RuntimeError(f"degenerate interface facet {facet}")
            normal /= 2.0 * area
            cells = f2c.links(facet)
            low_cell = int(cells[0]) if labels[cells[0]] == pair[0] else int(cells[1])
            high_cell = int(cells[1]) if low_cell == int(cells[0]) else int(cells[0])
            if float(normal @ (centers[high_cell] - centers[low_cell])) < 0.0:
                normal *= -1.0
            areas[interface_id] += area
            weighted_centers[interface_id] += area * vertices.mean(axis=0)
            weighted_normals[interface_id] += area * normal
    interface_centers = weighted_centers / areas[:, None]
    magnitudes = np.linalg.norm(weighted_normals, axis=1)
    if np.any(magnitudes <= 1.0e-14):
        bad = np.flatnonzero(magnitudes <= 1.0e-14).tolist()
        raise RuntimeError(f"interface normal cancels on interfaces {bad}; refine watershed grouping")
    interface_normals = weighted_normals / magnitudes[:, None]
    return DigitalCorePartition(
        mesh=msh, cell_centers=centers, cell_clearance=np.zeros(n_cells), cell_labels=labels,
        interface_pairs=interface_pairs, facet_interface_ids=facet_ids,
        interface_centers=interface_centers, interface_normals=interface_normals,
        interface_areas=areas,
        mesh_parameters={
            "source": "Bentheimer labelled voxel mesh",
            "interface_grouping": (
                "connected facet patches" if split_disconnected_interfaces
                else "all perforated patches for each subdomain pair"
            ),
        },
        cad_counts={"pore_regions": len(np.unique(labels)), "interfaces": len(interface_pairs)},
    )

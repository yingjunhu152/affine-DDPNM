"""Geometry loading and mesh-to-pore projection utilities."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
LEGACY_FLOW = PROJECT.parent / "affine_ddpnm_twophase"
COUPLED_FLOW = PROJECT.parent / "affine_ddpnm_twophase_coupled"


def enable_flow_dependencies() -> None:
    for dependency in (LEGACY_FLOW, COUPLED_FLOW):
        path = str(dependency)
        if path not in sys.path:
            sys.path.insert(0, path)


def build_random27(mesh_file: Path | None = None):
    enable_flow_dependencies()
    from random_porous import build_partition

    return build_partition(mesh_file=mesh_file)


def build_bentheimer(mesh_file: Path, *, split_disconnected_interfaces: bool = True):
    """Load the labelled inverted-Bentheimer mesh without modifying it."""
    enable_flow_dependencies()
    from digital_core_partition import build_digital_core_partition

    partition = build_digital_core_partition(
        mesh_file, split_disconnected_interfaces=split_disconnected_interfaces
    )
    labels = np.asarray(partition.cell_labels, dtype=np.int32)
    unique = np.unique(labels)
    mapping = {int(label): index for index, label in enumerate(unique)}
    partition.cell_labels = np.asarray([mapping[int(label)] for label in labels], dtype=np.int32)
    partition.interface_pairs = tuple(
        (mapping[int(a)], mapping[int(b)]) for a, b in partition.interface_pairs
    )
    return partition


def tetrahedron_volumes(msh) -> np.ndarray:
    dofmap = np.asarray(msh.geometry.dofmap, dtype=np.int32)
    vertices = np.asarray(msh.geometry.x, dtype=float)[dofmap[:, :4], :3]
    jacobian = vertices[:, 1:, :] - vertices[:, :1, :]
    volumes = np.abs(np.linalg.det(jacobian)) / 6.0
    if np.any(volumes <= 0.0) or not np.all(np.isfinite(volumes)):
        raise RuntimeError("Mesh contains an invalid tetrahedron")
    return volumes


@dataclass(frozen=True)
class PoreProjector:
    """Volume-weighted P1 field averages on pore labels."""

    labels: np.ndarray
    cell_vertices: np.ndarray
    cell_volumes: np.ndarray
    pore_volumes: np.ndarray

    @classmethod
    def from_partition(cls, partition) -> "PoreProjector":
        msh = partition.mesh
        msh.topology.create_connectivity(msh.topology.dim, 0)
        c2v = msh.topology.connectivity(msh.topology.dim, 0)
        ncells = msh.topology.index_map(msh.topology.dim).size_local
        cell_vertices = np.vstack([c2v.links(cell) for cell in range(ncells)]).astype(np.int32)
        labels = np.asarray(partition.cell_labels, dtype=np.int32)
        cell_volumes = tetrahedron_volumes(msh)
        npores = int(labels.max()) + 1
        pore_volumes = np.bincount(labels, weights=cell_volumes, minlength=npores)
        if np.any(pore_volumes <= 0.0):
            raise RuntimeError("Every pore label must own positive volume")
        return cls(labels, cell_vertices, cell_volumes, pore_volumes)

    def average_vertices(self, vertex_values: np.ndarray) -> np.ndarray:
        values = np.asarray(vertex_values, dtype=float)
        cell_mean = np.mean(values[self.cell_vertices], axis=1)
        weighted = np.bincount(
            self.labels, weights=self.cell_volumes * cell_mean,
            minlength=len(self.pore_volumes),
        )
        return weighted / self.pore_volumes

    def viscosity(self, vertex_phi: np.ndarray, mu_water: float, mu_oil: float) -> np.ndarray:
        phi_bar = np.clip(self.average_vertices(vertex_phi), -1.0, 1.0)
        water = 0.5 * (1.0 + phi_bar)
        return water * mu_water + (1.0 - water) * mu_oil

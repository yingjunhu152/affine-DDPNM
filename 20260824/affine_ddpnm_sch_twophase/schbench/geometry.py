"""Geometry loading and mesh-to-pore projection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_BENTHEIMER_MESH = PROJECT / "data" / "bentheimer_voxel_pore_mesh.msh"


def enable_flow_dependencies() -> None:
    """Compatibility no-op: all flow dependencies now live in this project."""


def build_random27(mesh_file: Path | None = None):
    from random_porous import build_partition

    return build_partition(mesh_file=mesh_file)


def build_bentheimer(
    mesh_file: Path = DEFAULT_BENTHEIMER_MESH,
    *,
    split_disconnected_interfaces: bool = True,
):
    """Load the labelled inverted-Bentheimer mesh without modifying it."""
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
    vertex_coordinates: np.ndarray
    vertex_lumped_volumes: np.ndarray

    @classmethod
    def from_partition(cls, partition) -> "PoreProjector":
        from ddpnm_core.io import topology_vertex_coordinates

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
        coordinates = topology_vertex_coordinates(msh)
        vertex_weights = np.bincount(
            cell_vertices.reshape(-1),
            weights=np.repeat(cell_volumes / cell_vertices.shape[1], cell_vertices.shape[1]),
            minlength=len(coordinates),
        )
        return cls(
            labels, cell_vertices, cell_volumes, pore_volumes,
            coordinates, vertex_weights,
        )

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

    def phase_diagnostics(
        self,
        vertex_phi: np.ndarray,
        *,
        inlet_bandwidth: float,
    ) -> dict[str, float | int]:
        """Locate bound violations without modifying the P1 phase field.

        Volume fractions use the standard tetrahedral P1 lumped vertex weights.
        The near-inlet share is the fraction of total under-range lumped volume
        contained within one supplied inlet bandwidth of the minimum-x boundary.
        """
        phi = np.asarray(vertex_phi, dtype=float)
        if phi.shape != (len(self.vertex_coordinates),):
            raise ValueError("Phase vertex field has the wrong shape")
        total_volume = float(np.sum(self.vertex_lumped_volumes))
        below = phi < -1.0
        above = phi > 1.0
        significant_below = phi < -1.001
        below_volume = float(np.sum(self.vertex_lumped_volumes[below]))
        above_volume = float(np.sum(self.vertex_lumped_volumes[above]))
        significant_below_volume = float(
            np.sum(self.vertex_lumped_volumes[significant_below])
        )
        xmin = float(np.min(self.vertex_coordinates[:, 0]))
        near_inlet = self.vertex_coordinates[:, 0] <= xmin + inlet_bandwidth
        min_index = int(np.argmin(phi))
        incident = np.flatnonzero(np.any(self.cell_vertices == min_index, axis=1))
        if len(incident):
            incident_pore_weights = np.bincount(
                self.labels[incident],
                weights=self.cell_volumes[incident],
                minlength=len(self.pore_volumes),
            )
            min_pore = int(np.argmax(incident_pore_weights))
        else:
            min_pore = -1
        pore_averages = self.average_vertices(phi)
        cell_jump = np.ptp(phi[self.cell_vertices], axis=1)
        point = self.vertex_coordinates[min_index]
        return {
            "phi_below_minus_one_vertex_fraction": float(np.mean(below)),
            "phi_above_plus_one_vertex_fraction": float(np.mean(above)),
            "phi_below_minus_one_lumped_volume_fraction": below_volume / total_volume,
            "phi_above_plus_one_lumped_volume_fraction": above_volume / total_volume,
            "phi_below_minus_one_minus_1e3_lumped_volume_fraction": (
                significant_below_volume / total_volume
            ),
            "phi_bound_violation_l1_volume_mean": float(np.sum(
                self.vertex_lumped_volumes
                * (np.maximum(-1.0 - phi, 0.0) + np.maximum(phi - 1.0, 0.0))
            )) / total_volume,
            "phi_below_minus_one_near_inlet_share": (
                float(np.sum(self.vertex_lumped_volumes[below & near_inlet])) / below_volume
                if below_volume > 0.0 else 0.0
            ),
            "phi_significant_undershoot_near_inlet_share": (
                float(np.sum(
                    self.vertex_lumped_volumes[significant_below & near_inlet]
                )) / significant_below_volume
                if significant_below_volume > 0.0 else 0.0
            ),
            "phi_undershoot_amplitude": max(0.0, -1.0 - float(phi[min_index])),
            "phi_overshoot_amplitude": max(0.0, float(np.max(phi)) - 1.0),
            "phi_min_x": float(point[0]),
            "phi_min_y": float(point[1]),
            "phi_min_z": float(point[2]),
            "phi_min_distance_from_inlet": float(point[0] - xmin),
            "phi_min_pore": min_pore,
            "phi_max_cell_vertex_jump": float(np.max(cell_jump)),
            "pore_average_phi_min": float(np.min(pore_averages)),
            "pore_average_phi_max": float(np.max(pore_averages)),
            "pore_average_clipped_count": int(np.count_nonzero(
                (pore_averages < -1.0) | (pore_averages > 1.0)
            )),
        }

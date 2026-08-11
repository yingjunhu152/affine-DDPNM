from __future__ import annotations

import numpy as np

from .geometry import PARTITION_PLANES, SPHERE_CENTERS, SPHERE_RADIUS, PartitionData


def tetra_diameters_and_quality(
    points: np.ndarray, tetrahedra: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertex_pairs = np.asarray(
        [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=np.int32
    )
    coords = points[tetrahedra]
    edges = coords[:, vertex_pairs[:, 1], :] - coords[:, vertex_pairs[:, 0], :]
    lengths_squared = np.sum(edges * edges, axis=2)
    diameters = np.sqrt(np.max(lengths_squared, axis=1))
    matrix = np.stack(
        [coords[:, 1] - coords[:, 0], coords[:, 2] - coords[:, 0], coords[:, 3] - coords[:, 0]],
        axis=1,
    )
    volumes = np.abs(np.linalg.det(matrix)) / 6.0
    quality = 12.0 * np.power(3.0 * volumes, 2.0 / 3.0) / np.sum(lengths_squared, axis=1)
    return diameters, quality, volumes


def boundary_surface_audit(
    points: np.ndarray, boundary_triangles: np.ndarray
) -> dict:
    triangles = points[boundary_triangles]
    outer = np.zeros(len(boundary_triangles), dtype=bool)
    for axis in range(3):
        outer |= np.all(np.abs(triangles[:, :, axis]) <= 5.0e-8, axis=1)
        outer |= np.all(np.abs(triangles[:, :, axis] - 1.0) <= 5.0e-8, axis=1)
    sphere_triangles = triangles[~outer]
    if not len(sphere_triangles):
        raise RuntimeError("No spherical boundary triangles were identified.")
    cross = np.cross(
        sphere_triangles[:, 1] - sphere_triangles[:, 0],
        sphere_triangles[:, 2] - sphere_triangles[:, 0],
    )
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    edge_lengths = np.stack(
        [
            np.linalg.norm(sphere_triangles[:, 1] - sphere_triangles[:, 0], axis=1),
            np.linalg.norm(sphere_triangles[:, 2] - sphere_triangles[:, 1], axis=1),
            np.linalg.norm(sphere_triangles[:, 0] - sphere_triangles[:, 2], axis=1),
        ],
        axis=1,
    )
    centroids = sphere_triangles.mean(axis=1)
    sphere_ids = np.argmin(
        np.linalg.norm(centroids[:, None, :] - SPHERE_CENTERS[None, :, :], axis=2),
        axis=1,
    )
    counts = np.asarray(
        [np.count_nonzero(sphere_ids == index) for index in range(len(SPHERE_CENTERS))]
    )
    discrete_areas = np.asarray(
        [np.sum(areas[sphere_ids == index]) for index in range(len(SPHERE_CENTERS))]
    )
    analytic_area = 4.0 * np.pi * SPHERE_RADIUS**2
    area_ratios = discrete_areas / analytic_area
    maximum_chord = float(np.max(edge_lengths))
    half_chord = min(0.5 * maximum_chord, SPHERE_RADIUS)
    sagitta = SPHERE_RADIUS - np.sqrt(
        max(SPHERE_RADIUS**2 - half_chord**2, 0.0)
    )
    return {
        "outer_cube_triangles": int(np.count_nonzero(outer)),
        "sphere_triangles": int(len(sphere_triangles)),
        "triangles_per_sphere": {
            "minimum": int(np.min(counts)),
            "median": float(np.median(counts)),
            "maximum": int(np.max(counts)),
        },
        "sphere_discrete_to_analytic_area_ratio": {
            "minimum": float(np.min(area_ratios)),
            "mean": float(np.mean(area_ratios)),
            "maximum": float(np.max(area_ratios)),
        },
        "sphere_surface_edges": {
            "median_length": float(np.median(edge_lengths)),
            "maximum_length": maximum_chord,
            "maximum_sagitta_over_radius": float(sagitta / SPHERE_RADIUS),
        },
    }


def mesh_audit(
    partition: PartitionData,
    points: np.ndarray,
    tetrahedra: np.ndarray,
    boundary_triangles: np.ndarray,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    diameters, quality, volumes = tetra_diameters_and_quality(points, tetrahedra)
    centers = partition.cell_centers
    sphere_distance = np.min(
        np.linalg.norm(
            centers[:, None, :] - SPHERE_CENTERS[None, :, :], axis=2
        )
        - SPHERE_RADIUS,
        axis=1,
    )
    outer_distance = np.minimum(np.min(centers, axis=1), np.min(1.0 - centers, axis=1))
    interface_distance = np.min(
        np.abs(centers[:, :, None] - PARTITION_PLANES[None, None, :]),
        axis=(1, 2),
    )
    parameters = partition.mesh_parameters
    masks = {
        "near_sphere": sphere_distance <= 0.5 * parameters["sphere_band"],
        "near_outer_boundary": outer_distance <= 0.5 * parameters["boundary_band"],
        "near_interface": interface_distance <= 0.5 * parameters["interface_band"],
        "bulk": (
            (sphere_distance >= parameters["sphere_band"])
            & (outer_distance >= parameters["boundary_band"])
            & (interface_distance >= parameters["interface_band"])
        ),
    }
    zones: dict[str, dict[str, float | int]] = {}
    for name, mask in masks.items():
        values = diameters[mask]
        zones[name] = {
            "tetrahedra": int(np.count_nonzero(mask)),
            "median_diameter": float(np.median(values)) if len(values) else float("nan"),
            "mean_diameter": float(np.mean(values)) if len(values) else float("nan"),
            "maximum_diameter": float(np.max(values)) if len(values) else float("nan"),
        }
    interface_counts = [
        int(np.count_nonzero(partition.facet_interface_ids == interface_id))
        for interface_id in range(len(partition.interface_pairs))
    ]
    audit = {
        "cad_entities": partition.cad_counts,
        "boundary_surfaces": boundary_surface_audit(points, boundary_triangles),
        "fluid_volume": {
            "discrete": float(np.sum(volumes)),
            "analytic": float(
                1.0 - len(SPHERE_CENTERS) * (4.0 / 3.0) * np.pi * SPHERE_RADIUS**3
            ),
        },
        "diameter": {
            "minimum": float(np.min(diameters)),
            "median": float(np.median(diameters)),
            "maximum": float(np.max(diameters)),
        },
        "mean_ratio_quality": {
            "minimum": float(np.min(quality)),
            "median": float(np.median(quality)),
            "mean": float(np.mean(quality)),
        },
        "zones": zones,
        "interfaces": {
            "minimum_triangles": int(min(interface_counts)),
            "maximum_triangles": int(max(interface_counts)),
            "triangles_per_interface": interface_counts,
        },
    }
    audit["fluid_volume"]["relative_error"] = float(
        abs(audit["fluid_volume"]["discrete"] - audit["fluid_volume"]["analytic"])
        / audit["fluid_volume"]["analytic"]
    )
    return audit, diameters, quality, volumes

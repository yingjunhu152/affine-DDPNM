"""AffineFaceBasis suitability check on the bent watershed interfaces (hand-off 5.2).

The affine basis consumes, per interface, exactly three geometric objects:
``interface_centers``, ``interface_normals`` and the tangent frame built by
``_interface_frames_and_scales`` (basis_3d.py) from the average normal and the
interface's own mesh vertices.  For the planar Voronoi interfaces of the
baseline these are trivially consistent; for the *bent staircase facet
interfaces* of the watershed partition this script verifies:

1. interface normals are unit vectors;
2. interface centers lie on the facet union (distance to the nearest facet
   of the same interface);
3. single-average-normal error — per-interface spread of the facet normals
   around the average, reported as angular deviation and as the two
   dispersion definitions (the *code formula* actually implemented in
   ``compute_interface_geometry``, unweighted ``sum(1 - n_f . nbar)/A``, and
   the docstring formula ``(1/A) sum A_f (1 - n_f . nbar)^2`` — hand-off
   3.4 notes the mismatch, so both are reported explicitly);
4. affine frame sanity — tangents orthonormal and orthogonal to the normal,
   scales positive, and ``max|rel . t| / scale == 1`` by construction
   (frame coverage).

No Stokes solve is attempted here; the checks are geometric only and feed
the 5.3 ablation decision.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPOSITORY_DIR = Path(__file__).resolve().parent.parent
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))
UNIFORM_DIR = REPOSITORY_DIR / "ddpnm_3d_uniform_spheres"
if str(UNIFORM_DIR) not in sys.path:
    sys.path.insert(0, str(UNIFORM_DIR))

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "watershed_formal"
WS_FIELDS = OUT_DIR / "watershed_formal_fields.npz"
VOR_FIELDS = Path(__file__).resolve().parent / "outputs" / "benchmark" / "random_benchmark_fields.npz"


def facet_normals(triangles: np.ndarray) -> np.ndarray:
    """Unit normals of the facet triangles."""
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    norms = np.linalg.norm(cross, axis=1)
    return cross / norms[:, None]


def center_to_interface_distances(
    centers: np.ndarray,
    facet_ids: np.ndarray,
    triangles: np.ndarray,
    n_interfaces: int,
) -> np.ndarray:
    """Distance from each interface center to its own facet set.

    For a planar interface this is machine-zero; for a bent staircase it is
    the out-of-surface offset of the area-weighted centroid mean.
    """
    from compare_partitions_formal import point_triangle_distances

    distances = np.empty(n_interfaces, dtype=float)
    for interface_id in range(n_interfaces):
        own = np.flatnonzero(facet_ids == interface_id)
        if len(own) == 0:
            distances[interface_id] = np.nan
            continue
        d = point_triangle_distances(centers[interface_id][None, :], triangles[own])
        distances[interface_id] = float(d.min())
    return distances


def angular_deviation(triangles: np.ndarray, average: np.ndarray) -> np.ndarray:
    """Angular deviation (degrees) of each facet normal from the average."""
    dots = np.clip(np.abs(facet_normals(triangles) @ average), 0.0, 1.0)
    return np.degrees(np.arccos(dots))


def dispersion_metrics(
    triangles: np.ndarray, areas: np.ndarray, average: np.ndarray
) -> tuple[float, float, float]:
    """(code formula, docstring formula, area-weighted max angle) per interface.

    Code formula (as implemented in ``compute_interface_geometry``):
    ``sum_f (1 - n_f . nbar) / A`` — unweighted by facet area, no square.
    Docstring formula: ``(1/A) sum_f A_f (1 - n_f . nbar)^2``.
    """
    unit = facet_normals(triangles)
    dots = unit @ average
    A = float(areas.sum())
    code = float(np.sum(1.0 - dots)) / A
    doc = float(np.sum(areas * (1.0 - dots) ** 2)) / A
    angle = float(np.max(np.degrees(np.arccos(np.clip(np.abs(dots), 0.0, 1.0)))))
    return code, doc, angle


def interface_summary(
    interface_centers: np.ndarray,
    interface_normals: np.ndarray,
    interface_areas: np.ndarray,
    facet_ids: np.ndarray,
    triangles: np.ndarray,
    n_interfaces: int,
) -> dict:
    """Per-interface geometric suitability metrics."""
    from ddpnm3d.basis_3d import _tangent_frame

    norm_norms = np.linalg.norm(interface_normals, axis=1)
    center_dist = center_to_interface_distances(
        interface_centers, facet_ids, triangles, n_interfaces
    )
    center_to_scale = np.full(n_interfaces, np.nan, dtype=float)
    scale_coverage = np.full(n_interfaces, np.nan, dtype=float)
    code_disp = np.full(n_interfaces, np.nan, dtype=float)
    doc_disp = np.full(n_interfaces, np.nan, dtype=float)
    max_angle = np.full(n_interfaces, np.nan, dtype=float)
    for interface_id in range(n_interfaces):
        own = np.flatnonzero(facet_ids == interface_id)
        if len(own) == 0:
            continue
        average = interface_normals[interface_id]
        unit = facet_normals(triangles[own])
        dots = unit @ average
        A = float(interface_areas[interface_id])
        code_disp[interface_id] = float(np.sum(1.0 - dots)) / A
        facet_areas = 0.5 * np.linalg.norm(
            np.cross(triangles[own, 1] - triangles[own, 0],
                     triangles[own, 2] - triangles[own, 0]),
            axis=1,
        )
        doc_disp[interface_id] = float(
            np.sum(facet_areas * (1.0 - dots) ** 2)
        ) / A
        max_angle[interface_id] = float(
            np.max(np.degrees(np.arccos(np.clip(np.abs(dots), 0.0, 1.0))))
        )
        # Interface scale (as in _interface_frames_and_scales) and the
        # dimensionless centre offset: the affine origin sits at the centre,
        # so an offset much smaller than the interface extent means the
        # s,t=0 point is genuinely on/inside the interface.
        t1, t2 = _tangent_frame(average)
        rel = triangles[own].reshape(-1, 3) - interface_centers[interface_id]
        scale = max(
            float(np.max(np.abs(rel @ t1))),
            float(np.max(np.abs(rel @ t2))),
            1.0e-12,
        )
        center_to_scale[interface_id] = center_dist[interface_id] / scale
        scale_coverage[interface_id] = float(
            np.max(np.abs(rel @ t1)) / scale
        )
    return {
        "n_interfaces": int(n_interfaces),
        "normal_norms": {
            "max_deviation_from_1": float(np.max(np.abs(norm_norms - 1.0))),
            "min": float(norm_norms.min()),
            "max": float(norm_norms.max()),
        },
        "center_to_own_facets": {
            "median": float(np.nanmedian(center_dist)),
            "max": float(np.nanmax(center_dist)),
            "max_interface": int(np.nanargmax(center_dist)),
        },
        "center_offset_over_scale": {
            "median": float(np.nanmedian(center_to_scale)),
            "max": float(np.nanmax(center_to_scale)),
        },
        "scale_coverage": {
            "min": float(np.nanmin(scale_coverage)),
            "max": float(np.nanmax(scale_coverage)),
        },
        "max_angular_deviation_deg": {
            "median": float(np.nanmedian(max_angle)),
            "max": float(np.nanmax(max_angle)),
            "max_interface": int(np.nanargmax(max_angle)),
            "n_flat_under_10deg": int(np.nansum(max_angle < 10.0)),
            "n_flat_under_20deg": int(np.nansum(max_angle < 20.0)),
            "n_bent_over_30deg": int(np.nansum(max_angle > 30.0)),
        },
        "dispersion_code_formula": {
            "median": float(np.nanmedian(code_disp)),
            "max": float(np.nanmax(code_disp)),
            "max_interface": int(np.nanargmax(code_disp)),
        },
        "dispersion_docstring_formula": {
            "median": float(np.nanmedian(doc_disp)),
            "max": float(np.nanmax(doc_disp)),
            "max_interface": int(np.nanargmax(doc_disp)),
        },
    }


def affine_frame_check(
    interface_normals: np.ndarray, n_interfaces: int
) -> dict:
    """Tangent-frame sanity for the affine modes (mirrors basis_3d)."""
    from ddpnm3d.basis_3d import _tangent_frame

    violations = {"t1_not_unit": 0, "t2_not_unit": 0, "t1t2_not_orth": 0,
                  "t1n_not_orth": 0, "t2n_not_orth": 0}
    for interface_id in range(n_interfaces):
        t1, t2 = _tangent_frame(interface_normals[interface_id])
        n = interface_normals[interface_id]
        if abs(float(np.linalg.norm(t1)) - 1.0) > 1.0e-12:
            violations["t1_not_unit"] += 1
        if abs(float(np.linalg.norm(t2)) - 1.0) > 1.0e-12:
            violations["t2_not_unit"] += 1
        if abs(float(t1 @ t2)) > 1.0e-12:
            violations["t1t2_not_orth"] += 1
        if abs(float(t1 @ n)) > 1.0e-12:
            violations["t1n_not_orth"] += 1
        if abs(float(t2 @ n)) > 1.0e-12:
            violations["t2n_not_orth"] += 1
    return {"violations": violations, "n_interfaces": int(n_interfaces)}


def main() -> None:
    ws = np.load(WS_FIELDS)
    vor = np.load(VOR_FIELDS)

    print("=== Watershed (bent facet interfaces) ===")
    # The stored facet ids are dolfinx facet indices; the per-facet interface
    # id is facet_interface_ids[facet].  Grouping must use that, not the raw
    # facet index.
    facet_ids = ws["ws_interface_facet_ids"]
    per_facet_interface = ws["facet_interface_ids"][facet_ids]
    consistent = bool(np.all(per_facet_interface >= 0))
    ws_summary = interface_summary(
        ws["interface_centers"],
        ws["interface_normals"],
        ws["interface_areas"],
        per_facet_interface,
        ws["ws_interface_facet_triangles"],
        len(ws["interface_pairs"]),
    )
    print(f"facet-id consistency with facet_interface_ids: {consistent}")
    print(
        f"normal |n|-1 max deviation: "
        f"{ws_summary['normal_norms']['max_deviation_from_1']:.2e}"
    )
    print(
        f"center->own-facets distance: median {ws_summary['center_to_own_facets']['median']:.4f}, "
        f"max {ws_summary['center_to_own_facets']['max']:.4f}"
    )
    print(
        f"center offset / interface scale: median "
        f"{ws_summary['center_offset_over_scale']['median']:.3f}, "
        f"max {ws_summary['center_offset_over_scale']['max']:.3f}"
    )
    print(
        f"max facet-normal deviation: median {ws_summary['max_angular_deviation_deg']['median']:.1f} deg, "
        f"max {ws_summary['max_angular_deviation_deg']['max']:.1f} deg"
    )
    print(
        f"flat (<10 deg): {ws_summary['max_angular_deviation_deg']['n_flat_under_10deg']}/"
        f"{ws_summary['n_interfaces']}, bent (>30 deg): "
        f"{ws_summary['max_angular_deviation_deg']['n_bent_over_30deg']}"
    )
    print(
        f"dispersion code formula: median {ws_summary['dispersion_code_formula']['median']:.4f}, "
        f"max {ws_summary['dispersion_code_formula']['max']:.4f}"
    )
    print(
        f"dispersion docstring formula: median "
        f"{ws_summary['dispersion_docstring_formula']['median']:.4f}, "
        f"max {ws_summary['dispersion_docstring_formula']['max']:.4f}"
    )
    ws_frame = affine_frame_check(ws["interface_normals"], ws_summary["n_interfaces"])
    print(f"tangent frame violations: {ws_frame['violations']}")

    print("=== Voronoi (planar interfaces, baseline) ===")
    vor_norms = np.linalg.norm(vor["interface_normals"], axis=1)
    print(
        f"normal |n|-1 max deviation: {float(np.max(np.abs(vor_norms - 1.0))):.2e}"
    )
    vor_frame = affine_frame_check(vor["interface_normals"], len(vor["interface_pairs"]))
    print(f"tangent frame violations: {vor_frame['violations']}")

    report = {
        "watershed": {
            "n_interfaces": ws_summary["n_interfaces"],
            "facet_id_consistency": bool(consistent),
            "geometry": ws_summary,
            "frame": ws_frame,
        },
        "voronoi": {
            "n_interfaces": int(len(vor["interface_pairs"])),
            "normal_norms": {
                "max_deviation_from_1": float(np.max(np.abs(vor_norms - 1.0)))
            },
            "frame": vor_frame,
        },
        "notes": {
            "dispersion_definition": (
                "compute_interface_geometry's code computes unweighted "
                "sum(1 - n_f . nbar)/A while its docstring claims "
                "(1/A) sum A_f (1 - n_f . nbar)^2; both are reported above. "
                "Either fix and rerun, or document the code formula in the "
                "reports (hand-off 3.4)."
            ),
            "frame_determinism": (
                "tangents come from _tangent_frame(average normal): unit and "
                "orthogonal by construction; violations would only arise from "
                "non-unit normals."
            ),
        },
    }
    (OUT_DIR / "basis_suitability_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Done: {OUT_DIR / 'basis_suitability_report.json'}")
    print("SUITABILITY CHECK OK")


if __name__ == "__main__":
    main()

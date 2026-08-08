"""Four-combination ablation: partition drawing x reduced basis (hand-off 5.3).

Combines the frozen Voronoi benchmark (outputs/benchmark/, the baseline that
must not be touched) with the watershed benchmark run at the same mesh
parameters (outputs/ablation_4way/watershed/) into one comparison table:

    Voronoi x Classic, Voronoi x Affine, Watershed x Classic, Watershed x
    Affine — velocity L2, broken-H1, pressure, outlet flux error, interface
    unknowns, first-solve time, speedup vs FEM, Schur symmetry, mass
    residual, min Schur eigenvalue.

The Voronoi report predates the per-method algebra diagnostics, so the Schur
symmetry error is recomputed there from the stored Schur matrices.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
ABLATION_DIR = PROJECT_DIR / "outputs" / "ablation_4way"


def load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def schur_symmetry_from_npz(npz_path: Path) -> dict[str, float]:
    data = np.load(npz_path)
    result = {}
    for name in ("classic", "affine"):
        schur = data[f"schur_matrix_{name}"]
        scale = max(float(np.linalg.norm(schur)), 1.0e-30)
        result[name] = float(np.linalg.norm(schur - schur.T) / scale)
    return result


def combo_rows(report: dict, symmetry: dict[str, float]) -> dict:
    rows: dict[str, dict] = {}
    for method, short in (("Classic-DDPNM-1", "classic"), ("Affine-DDPNM-9", "affine")):
        data = report["methods"][method]
        rows[short] = {
            "global_unknowns": data["global_unknowns"],
            "first_solve_seconds": report["timings"][method]["first_solve_seconds"],
            "speedup_vs_fem_first_solve": data.get(
                "speedup_vs_fem_first_solve", 0.0
            ),
            "velocity_relative_l2": data["velocity_relative_l2"],
            "velocity_relative_broken_h1": data["velocity_relative_broken_h1"],
            "pressure_relative_l2": data["pressure_relative_l2"],
            "pressure_aligned_relative_l2": data["pressure_aligned_relative_l2"],
            "outlet_flux_relative_error": data["outlet_flux_relative_error"],
            "schur_symmetry_error": data.get(
                "schur_symmetry_error", symmetry.get(short, np.nan)
            ),
            "max_mass_residual": data.get("max_mass_residual", np.nan),
            "min_schur_eigenvalue": data.get("min_schur_eigenvalue", np.nan),
            "boundary_fluxes": data.get("boundary_fluxes", {}),
        }
    return rows


def main() -> None:
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    vor_report = load_report(PROJECT_DIR / "outputs" / "benchmark" / "random_affine_report.json")
    ws_report = load_report(ABLATION_DIR / "watershed" / "random_affine_report.json")

    vor_symmetry = schur_symmetry_from_npz(
        PROJECT_DIR / "outputs" / "benchmark" / "random_benchmark_fields.npz"
    )
    ws_symmetry = schur_symmetry_from_npz(
        ABLATION_DIR / "watershed" / "random_benchmark_fields.npz"
    )

    combos = {
        "voronoi": combo_rows(vor_report, vor_symmetry),
        "watershed": combo_rows(ws_report, ws_symmetry),
    }

    # Mass conservation: inlet/outlet flux balance of the reduced methods.
    for partition in combos:
        for short in ("classic", "affine"):
            fluxes = combos[partition][short]["boundary_fluxes"]
            if not fluxes:
                continue
            inlet = float(fluxes.get("inlet", 0.0) or fluxes.get("Inlet", 0.0))
            outlet = float(fluxes.get("outlet", 0.0) or fluxes.get("Outlet", 0.0))
            combos[partition][short]["inlet_outlet_flux_imbalance"] = (
                abs(inlet + outlet) / max(abs(inlet), 1.0e-30)
            )

    table = {
        "metric": [
            "global_interface_unknowns",
            "velocity_relative_l2",
            "velocity_relative_broken_h1",
            "pressure_relative_l2",
            "outlet_flux_relative_error",
            "first_solve_seconds",
            "speedup_vs_fem_first_solve",
            "schur_symmetry_error",
            "max_mass_residual",
            "min_schur_eigenvalue",
            "inlet_outlet_flux_imbalance",
        ]
    }
    for partition in ("voronoi", "watershed"):
        for short in ("classic", "affine"):
            row = combos[partition][short]
            table[f"{partition}_{short}"] = [
                row["global_unknowns"],
                row["velocity_relative_l2"],
                row["velocity_relative_broken_h1"],
                row["pressure_relative_l2"],
                row["outlet_flux_relative_error"],
                row["first_solve_seconds"],
                row["speedup_vs_fem_first_solve"],
                row["schur_symmetry_error"],
                row["max_mass_residual"],
                row["min_schur_eigenvalue"],
                row.get("inlet_outlet_flux_imbalance", np.nan),
            ]

    report = {
        "table": table,
        "context": {
            "voronoi": {
                "mesh_cells": vor_report["counts"]["global_tetrahedra"],
                "pore_subdomains": vor_report["counts"]["pore_subdomains"],
                "interfaces": vor_report["counts"]["interfaces"],
                "mesh_parameters": {
                    k: vor_report["parameters"][k]
                    for k in (
                        "mesh_size", "sphere_size", "boundary_size",
                        "interface_size", "sphere_band", "boundary_band",
                        "interface_band",
                    )
                },
            },
            "watershed": {
                "mesh_cells": ws_report["counts"]["global_tetrahedra"],
                "pore_subdomains": ws_report["counts"]["pore_subdomains"],
                "interfaces": ws_report["counts"]["interfaces"],
                "mesh_parameters": {
                    k: ws_report["parameters"][k]
                    for k in (
                        "mesh_size", "sphere_size", "boundary_size",
                        "sphere_band", "boundary_band", "policy",
                        "abs_threshold", "rel_threshold",
                    )
                },
                "note": (
                    "watershed mesh has no interface size field (no internal "
                    "cuts); the unpartitioned mesh coarsens the saddle "
                    "regions relative to the Voronoi mesh."
                ),
            },
            "monolithic_fem": {
                "voronoi_mesh_dofs": vor_report["reference"]["mixed_dofs"],
                "watershed_mesh_dofs": ws_report["reference"]["mixed_dofs"],
                "voronoi_solve_seconds": vor_report["timings"]["Monolithic-FEM"]["first_solve_seconds"],
                "watershed_solve_seconds": ws_report["timings"]["Monolithic-FEM"]["first_solve_seconds"],
            },
        },
        "caveats": {
            "meshes_differ": (
                "The two partitions are meshed differently by construction: "
                "the Voronoi mesh embeds the saddle planes as internal cuts "
                "(with an interface size field), the watershed mesh has no "
                "cuts and no interface refinement.  Same geometry, same "
                "sphere/boundary fields."
            ),
            "watershed_exact_schur_skipped": (
                "The exact dense FE-trace Schur was not rerun for the "
                "watershed partition; correctness was established on the "
                "Voronoi partition (monolithic difference ~1e-12)."
            ),
        },
    }
    (ABLATION_DIR / "ablation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (ABLATION_DIR / "ablation_report.md").write_text(
        _markdown(report), encoding="utf-8"
    )

    header = ["metric"] + [
        f"{p}_{s}" for p in ("voronoi", "watershed") for s in ("classic", "affine")
    ]

    def fmt(value) -> str:
        if isinstance(value, str):
            return value
        return f"{value:.4g}"

    widths = [
        max(len(fmt(x)) for x in [h] + [table[h][i] for h in table]) + 2
        for i, h in enumerate(header)
    ]
    print("".join(str(h).ljust(widths[i]) for i, h in enumerate(header)))
    for metric in table["metric"]:
        print(
            "".join(
                fmt(table[h][table["metric"].index(metric)]).ljust(widths[i])
                for i, h in enumerate(header)
            )
        )
    print(f"Done: {ABLATION_DIR.resolve()}")


def _markdown(report: dict) -> str:
    lines = [
        "# Four-combination ablation: partition x basis (hand-off 5.3)",
        "",
        f"- Voronoi: {report['context']['voronoi']['mesh_cells']} cells, "
        f"{report['context']['voronoi']['pore_subdomains']} regions, "
        f"{report['context']['voronoi']['interfaces']} interfaces",
        f"- Watershed: {report['context']['watershed']['mesh_cells']} cells, "
        f"{report['context']['watershed']['pore_subdomains']} basins, "
        f"{report['context']['watershed']['interfaces']} interfaces",
        f"- Monolithic FEM: {report['context']['monolithic_fem']['voronoi_mesh_dofs']} "
        f"dofs ({report['context']['monolithic_fem']['voronoi_solve_seconds']:.1f} s) / "
        f"{report['context']['monolithic_fem']['watershed_mesh_dofs']} dofs "
        f"({report['context']['monolithic_fem']['watershed_solve_seconds']:.1f} s)",
        "",
        "| metric | VxClassic | VxAffine | WxClassic | WxAffine |",
        "|---|---|---|---|---|",
    ]
    for metric in report["table"]["metric"]:
        cells = " | ".join(
            f"{report['table'][f'{p}_{s}'][report['table']['metric'].index(metric)]:.4g}"
            for p in ("voronoi", "watershed")
            for s in ("classic", "affine")
        )
        lines.append(f"| {metric} | {cells} |")
    lines += [
        "",
        "## Caveats",
        "",
    ]
    for key, value in report["caveats"].items():
        lines.append(f"- **{key}**: {value}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()

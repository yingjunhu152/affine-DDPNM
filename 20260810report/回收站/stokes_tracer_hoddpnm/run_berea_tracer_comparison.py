from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import scipy.sparse as sp
import ufl
from basix.ufl import element
from dolfinx import fem
from dolfinx import mesh as dmesh
from mpi4py import MPI
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import LinearOperator, gmres, splu, spsolve
from scipy.spatial import cKDTree


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
DEFAULT_SOURCE_DIR = THIS_DIR


def preparse_source_dir(argv: list[str]) -> Path:
    for index, item in enumerate(argv):
        if item == "--source-dir" and index + 1 < len(argv):
            return Path(argv[index + 1])
        if item.startswith("--source-dir="):
            return Path(item.split("=", 1)[1])
    return DEFAULT_SOURCE_DIR


SOURCE_DIR = preparse_source_dir(sys.argv)
sys.path.insert(0, str(SOURCE_DIR))

import real_porous_hoddpnm_validation as base  # noqa: E402


METHODS = ("FEM", "HODDPNM")
COLORS = {
    "FEM": "#2b2d42",
    "HODDPNM": "#d1495b",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--volume-npy", type=Path, default=SOURCE_DIR / "data" / "berea_100_to_300.npz")
    parser.add_argument("--pore-value", type=float, default=1.0)
    parser.add_argument("--crop", default="20:36,150:166,30:46")
    parser.add_argument("--regions", type=int, default=16)
    parser.add_argument("--voxel-size", type=float, default=1.0)
    parser.add_argument("--interface-thickness", type=float, default=1.05)
    parser.add_argument("--pressure-interface-thickness", type=float, default=2.0)
    parser.add_argument("--pressure-stabilization", type=float, default=1.0e-6)
    parser.add_argument("--diffusivity", type=float, default=0.05)
    parser.add_argument("--porosity", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.20)
    parser.add_argument("--t-final", type=float, default=60.0)
    parser.add_argument("--supg", action="store_true")
    parser.add_argument("--supg-factor", type=float, default=0.50)
    parser.add_argument("--methods", default="FEM,HODDPNM")
    parser.add_argument("--schur-solver", choices=("gmres", "exact"), default="gmres")
    parser.add_argument("--schur-preconditioner", choices=("ilu", "none"), default="ilu")
    parser.add_argument("--schur-rtol", type=float, default=1.0e-10)
    parser.add_argument("--schur-atol", type=float, default=1.0e-12)
    parser.add_argument("--schur-restart", type=int, default=120)
    parser.add_argument("--schur-maxiter", type=int, default=400)
    parser.add_argument("--ilu-drop-tol", type=float, default=1.0e-4)
    parser.add_argument("--ilu-fill-factor", type=float, default=12.0)
    parser.add_argument("--max-active-ratio", type=float, default=0.50)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/singlephase_tracer_hoddpnm_strict"))
    args = parser.parse_args()

    methods = parse_methods(args.methods)
    validate_args(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    t_all = time.perf_counter()
    memory_trace = {"start": base.process_memory_mib()}

    pore = load_pore(args.volume_npy, args.pore_value, parse_crop(args.crop))
    mesh = base.build_real_porous_mesh(pore, args.voxel_size, args.regions)
    assembled = base.assemble_taylor_hood_system(
        mesh,
        args.pressure_stabilization,
        args.interface_thickness,
        args.pressure_interface_thickness,
    )
    memory_trace["after_stokes_assembly"] = base.process_memory_mib()

    stokes_solutions, stokes_rows = compute_stokes_solutions(mesh, assembled, methods, args)
    enforce_hoddpnm_compression(stokes_rows, args.max_active_ratio)
    memory_trace["after_stokes_solutions"] = base.process_memory_mib()

    tracer_rows: list[dict[str, object]] = []
    histories: dict[str, dict[str, np.ndarray]] = {}
    final_vertex_fields: dict[str, np.ndarray] = {}
    final_dof_fields: dict[str, np.ndarray] = {}
    tracer_mass_matrix: sp.csr_matrix | None = None

    for method in methods:
        velocity_p1 = velocity_on_vertices(mesh, assembled, stokes_solutions[method])
        t0 = time.perf_counter()
        tracer = solve_tracer(
            mesh,
            velocity_p1,
            diffusivity=args.diffusivity,
            porosity=args.porosity,
            dt=args.dt,
            t_final=args.t_final,
            supg=args.supg,
            supg_factor=args.supg_factor,
        )
        tracer_time = time.perf_counter() - t0
        if tracer_mass_matrix is None:
            tracer_mass_matrix = tracer["mass_matrix"]

        history = tracer["history"]
        history["method"] = method
        histories[method] = history
        final_vertex_fields[method] = tracer["final_concentration_vertices"]
        final_dof_fields[method] = tracer["final_concentration"]

        tracer_rows.append(
            {
                "method": method,
                "tracer_time_seconds": float(tracer_time),
                "mean_outlet_concentration": float(np.mean(history["cout"])),
                "final_outlet_concentration": float(history["cout"][-1]),
                "final_concentration_min": float(tracer["final_min"]),
                "final_concentration_max": float(tracer["final_max"]),
                "final_concentration_below_zero": int(tracer["final_below_zero"]),
                "final_concentration_above_one": int(tracer["final_above_one"]),
                "raw_final_concentration_min_before_limiter": float(tracer["raw_final_min_before_limiter"]),
                "raw_final_concentration_max_before_limiter": float(tracer["raw_final_max_before_limiter"]),
                "raw_final_concentration_below_zero_before_limiter": int(tracer["raw_final_below_zero_before_limiter"]),
                "raw_final_concentration_above_one_before_limiter": int(tracer["raw_final_above_one_before_limiter"]),
                "limiter_final_mass_residual": float(tracer["limiter_final_mass_residual"]),
                "max_mass_balance_residual_rate": float(np.max(np.abs(history["mass_balance_residual_rate"]))),
                "max_mass_balance_relative_residual": float(np.max(np.abs(history["mass_balance_relative_residual"]))),
                "max_limiter_mass_residual": float(np.max(np.abs(history["limiter_mass_residual"]))),
                "final_tracer_mass": float(tracer["final_mass"]),
                "min_tracer_mass": float(tracer["min_mass"]),
                "max_tracer_mass": float(tracer["max_mass"]),
                "t10": crossing_time(history["time"], history["cout"], 0.10),
                "t50": crossing_time(history["time"], history["cout"], 0.50),
                "t90": crossing_time(history["time"], history["cout"], 0.90),
                "outlet_flux_weight": float(tracer["outlet_flux_weight"]),
                "velocity_dof_mapping_max_distance": float(tracer["velocity_dof_mapping_max_distance"]),
                "concentration_vertex_mapping_max_distance": float(tracer["concentration_vertex_mapping_max_distance"]),
            }
        )
        save_concentration_vtu(
            mesh,
            velocity_p1,
            tracer["final_concentration_vertices"],
            args.out_dir / f"{method.lower()}_tracer_final.vtu",
        )

    if tracer_mass_matrix is None:
        raise RuntimeError("no tracer mass matrix was produced")

    add_reference_errors(tracer_rows, histories, final_dof_fields, tracer_mass_matrix, reference="FEM")
    attach_stokes_metrics(tracer_rows, stokes_rows)
    memory_trace["end"] = base.process_memory_mib()

    write_csv(args.out_dir / "stokes_velocity_cases.csv", stokes_rows)
    write_csv(args.out_dir / "tracer_metrics.csv", tracer_rows)
    write_history_csv(args.out_dir / "mass_balance_history.csv", histories)
    plot_breakthrough(args.out_dir / "breakthrough_curves.png", histories)
    plot_mass_balance(args.out_dir / "mass_balance_validation.png", histories)
    plot_error_summary(args.out_dir / "tracer_error_summary.png", tracer_rows)
    plot_final_concentration(
        args.out_dir / "final_concentration_and_error.png",
        mesh,
        final_vertex_fields,
        methods,
        reference="FEM",
    )

    summary = {
        "description": "Single-phase tracer validation driven by FEniCSx Taylor-Hood P2-P1 Stokes velocity fields.",
        "important_note": "No adaptive method levels are used. HODDPNM is evaluated as a fixed Schur/static-condensation solve of the same FEniCSx Stokes matrix.",
        "validation_note": "Breakthrough, final concentration field errors, and algebraic mass-balance residuals are all reported against the FEM-Stokes tracer reference.",
        "source_dir": str(args.source_dir),
        "stokes_model": "FEniCSx Taylor-Hood P2-P1 Stokes on real Berea pore geometry.",
        "hoddpnm_model": "Known Dirichlet dofs are removed first; the Schur active unknowns are free interface velocity dofs and free interface P1 pressure dofs.",
        "tracer_model": "Transient single-phase scalar advection-diffusion on the same pore mesh with inlet step concentration and natural outlet/wall flux.",
        "mass_conservation_model": "Algebraic finite-element balance: mass rate plus transport rate equals the Dirichlet source rate up to the free-row residual; the bounded limiter preserves the raw step mass when feasible.",
        "crop": args.crop,
        "regions": args.regions,
        "velocity_interface_thickness": args.interface_thickness,
        "pressure_interface_thickness": args.pressure_interface_thickness,
        "max_active_ratio": args.max_active_ratio,
        "mixed_dofs": int(len(assembled["b"])),
        "scalar_tracer_dofs": int(len(mesh.coords)),
        "diffusivity": args.diffusivity,
        "porosity": args.porosity,
        "dt": args.dt,
        "t_final": args.t_final,
        "supg": bool(args.supg),
        "timings_seconds": {
            "total_wall_time": float(time.perf_counter() - t_all),
        },
        "memory_trace_mib": memory_trace,
        "stokes_velocity_cases": stokes_rows,
        "tracer_metrics": tracer_rows,
        "outputs": {
            "breakthrough_curves": str(args.out_dir / "breakthrough_curves.png"),
            "mass_balance_validation": str(args.out_dir / "mass_balance_validation.png"),
            "tracer_error_summary": str(args.out_dir / "tracer_error_summary.png"),
            "final_concentration_and_error": str(args.out_dir / "final_concentration_and_error.png"),
            "mass_balance_history_csv": str(args.out_dir / "mass_balance_history.csv"),
            "tracer_metrics_csv": str(args.out_dir / "tracer_metrics.csv"),
            "stokes_velocity_cases_csv": str(args.out_dir / "stokes_velocity_cases.csv"),
        },
    }
    (args.out_dir / "tracer_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(args.out_dir / "TRACER_VALIDATION_REPORT.md", summary)

    print(f"wrote {args.out_dir / 'tracer_summary.json'}")
    for row in tracer_rows:
        print(
            f"{row['method']}: tracer={row['tracer_time_seconds']:.3f}s, "
            f"curve_error={row.get('breakthrough_rel_l2_error', float('nan')):.3e}, "
            f"field_error={row.get('concentration_rel_l2_fem_integral_vs_fem', float('nan')):.3e}, "
            f"mass_balance={row['max_mass_balance_relative_residual']:.3e}, "
            f"t90={row['t90']:.3f}"
        )


def parse_methods(text: str) -> list[str]:
    methods = [item.strip().upper() for item in text.split(",") if item.strip()]
    bad = [name for name in methods if name not in METHODS]
    if bad:
        raise SystemExit(f"unknown methods: {', '.join(bad)}; allowed methods are {', '.join(METHODS)}")
    if "FEM" not in methods:
        raise SystemExit("--methods must include FEM because FEM-Stokes tracer is the validation reference")
    return methods


def validate_args(args) -> None:
    if args.dt <= 0.0:
        raise SystemExit("--dt must be positive")
    if args.t_final < 0.0:
        raise SystemExit("--t-final must be nonnegative")
    if args.interface_thickness <= 0.0 or args.pressure_interface_thickness <= 0.0:
        raise SystemExit("interface thickness values must be positive")
    if not (0.0 < args.max_active_ratio <= 1.0):
        raise SystemExit("--max-active-ratio must be in (0, 1]")
    if not args.volume_npy.exists():
        raise SystemExit(f"volume file does not exist: {args.volume_npy}")


def parse_crop(text: str) -> tuple[slice, slice, slice]:
    spans = []
    for part in text.split(","):
        start, stop = part.split(":")
        spans.append(slice(int(start), int(stop)))
    if len(spans) != 3:
        raise SystemExit("--crop must be x0:x1,y0:y1,z0:z1")
    return tuple(spans)


def load_pore(path: Path, pore_value: float, crop: tuple[slice, slice, slice]) -> np.ndarray:
    arr = np.load(path)
    volume = arr[sorted(arr.files)[0]] if isinstance(arr, np.lib.npyio.NpzFile) else arr
    pore = np.asarray(volume[crop] == pore_value, dtype=bool)
    return base.keep_largest_pore_component(pore)


def compute_stokes_solutions(mesh, assembled, methods: list[str], args) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    A = assembled["A"].tocsr()
    b = np.asarray(assembled["b"], dtype=float)
    rows: list[dict[str, object]] = []
    solutions: dict[str, np.ndarray] = {}
    validation_mats = build_stokes_validation_mass_matrices(mesh)

    t0 = time.perf_counter()
    fem_solution = spsolve(A, b)
    fem_time = time.perf_counter() - t0
    solutions["FEM"] = fem_solution
    rows.append(
        case_row(
            "FEM",
            len(b),
            1.0,
            fem_time,
            None,
            None,
            fem_solution,
            fem_solution,
            assembled,
            validation_mats,
        )
    )

    if "HODDPNM" in methods:
        t0 = time.perf_counter()
        hodd = solve_hoddpnm_known_fixed_schur(
            A,
            b,
            np.asarray(assembled["interface_mask"], dtype=bool),
            np.asarray(assembled["fixed_dofs"], dtype=np.int64),
            solver=args.schur_solver,
            rtol=args.schur_rtol,
            atol=args.schur_atol,
            restart=args.schur_restart,
            maxiter=args.schur_maxiter,
            preconditioner=args.schur_preconditioner,
            ilu_drop_tol=args.ilu_drop_tol,
            ilu_fill_factor=args.ilu_fill_factor,
        )
        solve_time = time.perf_counter() - t0
        hodd_solution = np.asarray(hodd["solution"], dtype=float)
        solutions["HODDPNM"] = hodd_solution
        active_dofs = int(hodd["n_boundary"])
        row = case_row(
            "HODDPNM",
            active_dofs,
            float(active_dofs / len(b)),
            solve_time,
            "known-fixed-schur",
            hodd["solver_info"],
            hodd_solution,
            fem_solution,
            assembled,
            validation_mats,
        )
        row["hoddpnm_velocity_interface_dofs"] = int(assembled["hoddpnm_velocity_interface_dofs"])
        row["hoddpnm_pressure_interface_dofs"] = int(assembled["hoddpnm_pressure_interface_dofs"])
        row["hoddpnm_pressure_interior_dofs"] = int(assembled["hoddpnm_pressure_interior_dofs"])
        row["hoddpnm_interface_dofs"] = int(np.count_nonzero(assembled["interface_mask"]))
        row["hoddpnm_known_fixed_dofs"] = int(hodd["n_fixed_known"])
        row["hoddpnm_free_interior_dofs_eliminated"] = int(hodd["n_interior"])
        row["hoddpnm_interior_dofs_eliminated"] = int(hodd["n_interior"])
        row["hoddpnm_boundary_dofs_total"] = active_dofs
        rows.append(row)

    return solutions, rows


def enforce_hoddpnm_compression(stokes_rows: list[dict[str, object]], max_active_ratio: float) -> None:
    hodd_row = next((row for row in stokes_rows if row["method"] == "HODDPNM"), None)
    if hodd_row is None:
        return
    active_ratio = float(hodd_row["active_dof_ratio"])
    if active_ratio > max_active_ratio:
        raise RuntimeError(
            "HODDPNM active Schur space is too large: "
            f"{int(hodd_row['active_dofs'])} active dofs, ratio={active_ratio:.1%}, "
            f"limit={max_active_ratio:.1%}. Check known-fixed elimination and interface-pressure thickness."
        )


def solve_hoddpnm_known_fixed_schur(
    A: sp.csr_matrix,
    b: np.ndarray,
    interface_mask: np.ndarray,
    fixed_dofs: np.ndarray,
    solver: str,
    rtol: float,
    atol: float,
    restart: int,
    maxiter: int,
    preconditioner: str,
    ilu_drop_tol: float,
    ilu_fill_factor: float,
) -> dict[str, object]:
    n = A.shape[0]
    interface_mask = np.asarray(interface_mask, dtype=bool).copy()
    fixed = np.unique(np.asarray(fixed_dofs, dtype=np.int64))
    fixed = fixed[(fixed >= 0) & (fixed < n)]

    fixed_mask = np.zeros(n, dtype=bool)
    fixed_mask[fixed] = True
    boundary_mask = interface_mask & ~fixed_mask
    interior_mask = ~(boundary_mask | fixed_mask)

    boundary = np.flatnonzero(boundary_mask)
    interior = np.flatnonzero(interior_mask)
    known = np.flatnonzero(fixed_mask)
    x_known = np.asarray(b[known], dtype=float)

    rhs_all = np.asarray(b, dtype=float).copy()
    if len(known):
        rhs_all -= A[:, known] @ x_known

    Aii = A[interior][:, interior].tocsc()
    Aib = A[interior][:, boundary].tocsc()
    Abi = A[boundary][:, interior].tocsc()
    Abb = A[boundary][:, boundary].tocsc()
    bi = rhs_all[interior]
    bb = rhs_all[boundary]

    lu = splu(Aii)
    yi = lu.solve(bi)
    rhs = bb - Abi @ yi

    if solver == "exact":
        Xi = lu.solve(Aib.toarray())
        S = Abb.toarray() - Abi @ Xi
        ub = np.linalg.solve(S, rhs)
        solver_info = {
            "type": "dense_exact_schur_known_fixed",
            "converged": True,
            "iterations": 1,
            "gmres_info": 0,
            "relative_residual": 0.0,
            "preconditioner": "none",
        }
        ui = yi - Xi @ ub
    else:
        scale = base.schur_diagonal_scale(Aii, Aib, Abi, Abb)
        inv_scale = 1.0 / scale

        matvec_count = 0

        def schur_matvec(x: np.ndarray) -> np.ndarray:
            nonlocal matvec_count
            matvec_count += 1
            return Abb @ x - Abi @ lu.solve(Aib @ x)

        schur_operator = LinearOperator(
            Abb.shape,
            matvec=lambda x: scale * schur_matvec(scale * x),
            dtype=float,
        )
        preconditioner_operator, preconditioner_info = base.build_schur_preconditioner(
            Aii,
            Aib,
            Abi,
            Abb,
            preconditioner,
            ilu_drop_tol,
            ilu_fill_factor,
        )
        if preconditioner_operator is not None:
            inner_preconditioner = preconditioner_operator
            preconditioner_operator = LinearOperator(
                Abb.shape,
                matvec=lambda x: inv_scale * (inner_preconditioner @ (inv_scale * x)),
                dtype=float,
        )
        residuals = []
        gmres_rtol = min(float(rtol) * 1.0e-4, 1.0e-14)
        gmres_atol = min(float(atol) * 1.0e-4, 1.0e-16)
        scaled_rhs = scale * rhs
        yb, info = gmres(
            schur_operator,
            scaled_rhs,
            rtol=gmres_rtol,
            atol=gmres_atol,
            restart=restart,
            maxiter=maxiter,
            M=preconditioner_operator,
            callback=residuals.append,
            callback_type="pr_norm",
        )
        ub = scale * yb
        ui = yi - lu.solve(Aib @ ub)
        final_residual = np.linalg.norm(schur_matvec(ub) - rhs) / max(np.linalg.norm(rhs), 1.0e-30)
        solver_info = {
            "type": f"matrix_free_{solver}_schur_known_fixed",
            "converged": bool(info == 0),
            "iterations": int(len(residuals)),
            "schur_matvecs": int(matvec_count),
            "gmres_info": int(info),
            "relative_residual": float(final_residual),
            "rtol": float(rtol),
            "atol": float(atol),
            "restart": int(restart),
            "maxiter": int(maxiter),
            "schur_diagonal_scaling": True,
            "scaled_gmres_rtol": float(gmres_rtol),
            "scaled_gmres_atol": float(gmres_atol),
            "schur_scale_min": float(np.min(scale)),
            "schur_scale_max": float(np.max(scale)),
            **preconditioner_info,
        }
        if info != 0 or not np.isfinite(final_residual) or final_residual > max(10.0 * rtol, 1.0e-9):
            raise RuntimeError(
                "HODDPNM Schur GMRES did not converge: "
                f"info={info}, iterations={len(residuals)}, relative_residual={final_residual:.3e}"
            )
        S = None

    sol = np.zeros(n, dtype=float)
    sol[known] = x_known
    sol[boundary] = ub
    sol[interior] = ui
    return {
        "solution": sol,
        "schur_matrix": S,
        "n_boundary": int(len(boundary)),
        "n_interior": int(len(interior)),
        "n_fixed_known": int(len(known)),
        "solver": solver,
        "solver_info": solver_info,
    }


def build_hoddpnm_interface_pressure_mask(assembled: dict[str, object], mesh, pressure_interface_thickness: float) -> np.ndarray:
    mask = np.zeros(len(assembled["b"]), dtype=bool)
    map_v = np.asarray(assembled["mapV"], dtype=np.int64)
    map_q = np.asarray(assembled["mapQ"], dtype=np.int64)
    original_mask = np.asarray(assembled["interface_mask"], dtype=bool)

    velocity_interface = original_mask[map_v]
    mask[map_v] = velocity_interface

    q_coords = np.asarray(assembled["Q_coords"], dtype=float)
    if len(mesh.interface_centers) == 0:
        pressure_interface = np.zeros(len(q_coords), dtype=bool)
    else:
        tree = cKDTree(mesh.interface_centers)
        dist, _ = tree.query(q_coords, k=1, workers=-1)
        pressure_interface = dist <= pressure_interface_thickness * mesh.voxel_size
    mask[map_q] = pressure_interface

    assembled["hoddpnm_velocity_interface_dofs"] = int(np.count_nonzero(velocity_interface))
    assembled["hoddpnm_pressure_interface_dofs"] = int(np.count_nonzero(pressure_interface))
    assembled["hoddpnm_pressure_interior_dofs"] = int(len(map_q) - np.count_nonzero(pressure_interface))
    return mask


def case_row(method, active_dofs, active_ratio, solve_time, level, solver_info, solution, fem_solution, assembled, validation_mats: dict[str, sp.csr_matrix]):
    velocity_error = solution[assembled["mapV"]].reshape((-1, 3)) - fem_solution[assembled["mapV"]].reshape((-1, 3))
    pressure_error = solution[assembled["mapQ"]] - fem_solution[assembled["mapQ"]]
    velocity_ref = fem_solution[assembled["mapV"]].reshape((-1, 3))
    pressure_ref = fem_solution[assembled["mapQ"]]
    velocity_error_flat = velocity_error.reshape(-1)
    velocity_ref_flat = velocity_ref.reshape(-1)
    row = {
        "method": method,
        "solve_scope": "" if level is None else str(level),
        "active_dofs": int(active_dofs),
        "active_dof_ratio": float(active_ratio),
        "stokes_solve_time_seconds": float(solve_time),
        "velocity_rel_l2_error_vs_fem": float(np.linalg.norm(velocity_error_flat) / max(np.linalg.norm(velocity_ref_flat), 1.0e-300)),
        "pressure_rel_l2_error_vs_fem": float(np.linalg.norm(pressure_error) / max(np.linalg.norm(pressure_ref), 1.0e-300)),
        "velocity_rel_l2_fem_integral_vs_fem": mass_l2_relative(velocity_error_flat, velocity_ref_flat, validation_mats["velocity"]),
        "pressure_rel_l2_fem_integral_vs_fem": mass_l2_relative(pressure_error, pressure_ref, validation_mats["pressure"]),
    }
    if solver_info is not None:
        row["solver"] = str(solver_info.get("type", ""))
        row["schur_iterations"] = int(solver_info.get("iterations", -1))
        row["schur_matvecs"] = int(solver_info.get("schur_matvecs", -1))
        row["schur_relative_residual"] = float(solver_info.get("relative_residual", np.nan))
        row["schur_preconditioner"] = str(solver_info.get("preconditioner", ""))
    return row


def velocity_on_vertices(mesh, assembled, solution: np.ndarray) -> np.ndarray:
    tree = cKDTree(np.asarray(assembled["V_coords"], dtype=float))
    _, nearest = tree.query(mesh.coords, k=1)
    velocity_p2 = solution[assembled["mapV"]].reshape((-1, 3))
    return np.asarray(velocity_p2[nearest], dtype=float)


def solve_tracer(
    mesh,
    velocity_p1: np.ndarray,
    diffusivity: float,
    porosity: float,
    dt: float,
    t_final: float,
    supg: bool,
    supg_factor: float,
) -> dict[str, object]:
    domain_ufl = ufl.Mesh(element("Lagrange", "tetrahedron", 1, shape=(3,)))
    msh = dmesh.create_mesh(MPI.COMM_SELF, mesh.cells, domain_ufl, mesh.coords)
    cell = msh.basix_cell()
    C = fem.functionspace(msh, element("Lagrange", cell, 1))
    U = fem.functionspace(msh, element("Lagrange", cell, 1, shape=(3,)))
    c = ufl.TrialFunction(C)
    w = ufl.TestFunction(C)
    u = fem.Function(U)
    velocity_node_map, velocity_map_distance = nearest_indices(
        U.tabulate_dof_coordinates(),
        mesh.coords,
        "velocity P1 dof coordinates",
    )
    u_values = np.asarray(velocity_p1, dtype=float)[velocity_node_map]
    u.x.array[:] = u_values.reshape(-1)
    u.x.scatter_forward()

    h = ufl.CellDiameter(msh)
    speed = ufl.sqrt(ufl.inner(u, u) + fem.Constant(msh, 1.0e-14))
    mass_form = fem.form(c * w * ufl.dx)
    mass_matrix = base.to_scipy_matrix(fem.assemble_matrix(mass_form)).tocsr()
    mass_step_matrix = (float(porosity) / float(dt)) * mass_matrix
    rhs_operator = fem.Constant(msh, float(porosity / dt)) * c * w * ufl.dx
    operator = (
        fem.Constant(msh, float(porosity / dt)) * c * w * ufl.dx
        + fem.Constant(msh, float(diffusivity)) * ufl.inner(ufl.grad(c), ufl.grad(w)) * ufl.dx
        + ufl.dot(u, ufl.grad(c)) * w * ufl.dx
    )
    if supg:
        tau = fem.Constant(msh, float(supg_factor)) * h / (2.0 * speed)
        streamline_test = ufl.dot(u, ufl.grad(w))
        operator += tau * streamline_test * (
            fem.Constant(msh, float(porosity / dt)) * c + ufl.dot(u, ufl.grad(c))
        ) * ufl.dx
        rhs_operator += tau * streamline_test * fem.Constant(msh, float(porosity / dt)) * c * ufl.dx

    A = base.to_scipy_matrix(fem.assemble_matrix(fem.form(operator))).tocsr()
    transport_matrix = (A - mass_step_matrix).tocsr()
    rhs_matrix = base.to_scipy_matrix(fem.assemble_matrix(fem.form(rhs_operator))).tocsr()

    coords = C.tabulate_dof_coordinates()
    vertex_to_c_dof, concentration_map_distance = nearest_indices(
        mesh.coords,
        coords,
        "scalar concentration vertex output coordinates",
    )
    lx = mesh.domain_size[0]
    inlet_dofs = np.asarray(fem.locate_dofs_geometrical(C, lambda x: np.isclose(x[0], 0.0)), dtype=np.int64)
    outlet_dofs = np.flatnonzero(np.isclose(coords[:, 0], lx))
    if len(outlet_dofs) == 0:
        xmax = float(np.max(coords[:, 0]))
        outlet_dofs = np.flatnonzero(np.isclose(coords[:, 0], xmax))

    A_bc = impose_dirichlet_rows(A, inlet_dofs)
    lu = splu(A_bc.tocsc())
    mass_weights = np.asarray(mass_matrix @ np.ones(mass_matrix.shape[1]), dtype=float)
    cold_vec = np.zeros(C.dofmap.index_map.size_local * C.dofmap.index_map_bs, dtype=float)
    cold_vec[inlet_dofs] = 1.0

    tree = cKDTree(mesh.coords)
    _, nearest_velocity_nodes = tree.query(coords[outlet_dofs], k=1)
    outlet_weights = np.maximum(velocity_p1[nearest_velocity_nodes, 0], 0.0)
    if float(np.sum(outlet_weights)) <= 1.0e-14:
        outlet_weights = np.ones(len(outlet_dofs), dtype=float)
    outlet_flux_weight = float(np.sum(outlet_weights))

    times = [0.0]
    cout = [weighted_average(cold_vec[outlet_dofs], outlet_weights)]
    masses = [scalar_mass(cold_vec, mass_matrix)]
    budget_masses = [masses[0]]
    raw_masses = [masses[0]]
    mins = [float(np.min(cold_vec))]
    maxs = [float(np.max(cold_vec))]
    raw_mins = [float(np.min(cold_vec))]
    raw_maxs = [float(np.max(cold_vec))]
    raw_below_counts = [int(np.count_nonzero(cold_vec < -1.0e-12))]
    raw_above_counts = [int(np.count_nonzero(cold_vec > 1.0 + 1.0e-12))]
    mass_rate = [0.0]
    transport_rate = [0.0]
    source_rate = [0.0]
    balance_residual_rate = [0.0]
    relative_balance_residual = [0.0]
    limiter_mass_change_abs = [0.0]
    limiter_mass_residual = [0.0]

    n_steps = int(np.ceil(t_final / dt))
    free_mask = np.ones(len(cold_vec), dtype=bool)
    free_mask[inlet_dofs] = False
    for step in range(1, n_steps + 1):
        rhs = rhs_matrix @ cold_vec
        rhs_bc = rhs.copy()
        rhs_bc[inlet_dofs] = 1.0
        craw = lu.solve(rhs_bc)
        craw[inlet_dofs] = 1.0

        raw_mass = scalar_mass(craw, mass_matrix)
        raw_masses.append(raw_mass)
        raw_mins.append(float(np.min(craw)))
        raw_maxs.append(float(np.max(craw)))
        raw_below_counts.append(int(np.count_nonzero(craw < -1.0e-12)))
        raw_above_counts.append(int(np.count_nonzero(craw > 1.0 + 1.0e-12)))

        limited, limiter_info = conservative_bounded_limiter(craw, mass_weights, inlet_dofs, raw_mass)
        limited[inlet_dofs] = 1.0
        residual = A @ craw - rhs
        step_mass_rate = (raw_mass - masses[-1]) / dt
        step_transport_rate = float(np.sum(transport_matrix @ craw))
        step_source_rate = float(np.sum(residual[inlet_dofs]))
        step_balance = float(np.sum(residual[free_mask]))
        scale = max(abs(step_mass_rate) + abs(step_transport_rate) + abs(step_source_rate), 1.0e-300)

        cold_vec = limited
        times.append(float(min(step * dt, t_final)))
        cout.append(weighted_average(cold_vec[outlet_dofs], outlet_weights))
        masses.append(scalar_mass(cold_vec, mass_matrix))
        budget_masses.append(budget_masses[-1] + dt * (step_source_rate - step_transport_rate))
        mins.append(float(np.min(cold_vec)))
        maxs.append(float(np.max(cold_vec)))
        mass_rate.append(step_mass_rate)
        transport_rate.append(step_transport_rate)
        source_rate.append(step_source_rate)
        balance_residual_rate.append(step_balance)
        relative_balance_residual.append(abs(step_balance) / scale)
        limiter_mass_change_abs.append(float(limiter_info["mass_change_abs"]))
        limiter_mass_residual.append(float(limiter_info["mass_residual"]))

    return {
        "history": {
            "time": np.asarray(times),
            "cout": np.asarray(cout),
            "mass": np.asarray(masses),
            "budget_mass": np.asarray(budget_masses),
            "raw_mass": np.asarray(raw_masses),
            "mass_rate": np.asarray(mass_rate),
            "transport_rate": np.asarray(transport_rate),
            "dirichlet_source_rate": np.asarray(source_rate),
            "mass_balance_residual_rate": np.asarray(balance_residual_rate),
            "mass_balance_relative_residual": np.asarray(relative_balance_residual),
            "limiter_mass_change_abs": np.asarray(limiter_mass_change_abs),
            "limiter_mass_residual": np.asarray(limiter_mass_residual),
            "min_c": np.asarray(mins),
            "max_c": np.asarray(maxs),
            "raw_min_c_before_limiter": np.asarray(raw_mins),
            "raw_max_c_before_limiter": np.asarray(raw_maxs),
            "raw_below_zero_before_limiter": np.asarray(raw_below_counts),
            "raw_above_one_before_limiter": np.asarray(raw_above_counts),
        },
        "final_concentration": cold_vec,
        "final_concentration_vertices": np.asarray(cold_vec[vertex_to_c_dof], dtype=float),
        "mass_matrix": mass_matrix,
        "outlet_flux_weight": outlet_flux_weight,
        "final_mass": masses[-1],
        "min_mass": min(masses),
        "max_mass": max(masses),
        "final_min": mins[-1],
        "final_max": maxs[-1],
        "final_below_zero": int(np.count_nonzero(cold_vec < -1.0e-12)),
        "final_above_one": int(np.count_nonzero(cold_vec > 1.0 + 1.0e-12)),
        "raw_final_min_before_limiter": raw_mins[-1],
        "raw_final_max_before_limiter": raw_maxs[-1],
        "raw_final_below_zero_before_limiter": raw_below_counts[-1],
        "raw_final_above_one_before_limiter": raw_above_counts[-1],
        "limiter_final_mass_residual": limiter_mass_residual[-1],
        "velocity_dof_mapping_max_distance": velocity_map_distance,
        "concentration_vertex_mapping_max_distance": concentration_map_distance,
    }


def conservative_bounded_limiter(values: np.ndarray, weights: np.ndarray, fixed_dofs: np.ndarray, target_mass: float) -> tuple[np.ndarray, dict[str, float]]:
    out = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    fixed = np.zeros(len(out), dtype=bool)
    fixed[np.asarray(fixed_dofs, dtype=np.int64)] = True
    out[fixed] = np.clip(values[fixed], 0.0, 1.0)
    adjustable = ~fixed
    initial_limited = out.copy()

    for _ in range(32):
        residual = float(target_mass - np.dot(weights, out))
        if abs(residual) <= 1.0e-12 * max(abs(target_mass), 1.0):
            break
        if residual > 0.0:
            idx = adjustable & (out < 1.0 - 1.0e-14)
            capacity = weights[idx] * (1.0 - out[idx])
            total_capacity = float(np.sum(capacity))
            if total_capacity <= 1.0e-300:
                break
            theta = min(1.0, residual / total_capacity)
            out[idx] += theta * (1.0 - out[idx])
        else:
            idx = adjustable & (out > 1.0e-14)
            capacity = weights[idx] * out[idx]
            total_capacity = float(np.sum(capacity))
            if total_capacity <= 1.0e-300:
                break
            theta = min(1.0, -residual / total_capacity)
            out[idx] -= theta * out[idx]

    final_residual = float(target_mass - np.dot(weights, out))
    return out, {
        "mass_residual": final_residual,
        "mass_change_abs": float(np.dot(weights, np.abs(out - initial_limited))),
    }


def nearest_indices(query_coords: np.ndarray, source_coords: np.ndarray, label: str, tolerance: float = 1.0e-8) -> tuple[np.ndarray, float]:
    tree = cKDTree(np.asarray(source_coords, dtype=float))
    distances, indices = tree.query(np.asarray(query_coords, dtype=float), k=1)
    max_distance = float(np.max(distances)) if len(np.atleast_1d(distances)) else 0.0
    if max_distance > tolerance:
        raise RuntimeError(f"{label} do not match mesh coordinates within {tolerance:g}; max distance is {max_distance:.3e}")
    return np.asarray(indices, dtype=np.int64), max_distance


def impose_dirichlet_rows(A: sp.csr_matrix, dofs: np.ndarray) -> sp.csr_matrix:
    A = A.tolil(copy=True)
    for dof in np.asarray(dofs, dtype=np.int64):
        A.rows[int(dof)] = [int(dof)]
        A.data[int(dof)] = [1.0]
    return A.tocsr()


def weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    denom = max(float(np.sum(weights)), 1.0e-300)
    return float(np.dot(values, weights) / denom)


def build_stokes_validation_mass_matrices(mesh) -> dict[str, sp.csr_matrix]:
    domain_ufl = ufl.Mesh(element("Lagrange", "tetrahedron", 1, shape=(3,)))
    msh = dmesh.create_mesh(MPI.COMM_SELF, mesh.cells, domain_ufl, mesh.coords)
    cell = msh.basix_cell()
    V = fem.functionspace(msh, element("Lagrange", cell, 2, shape=(3,)))
    Q = fem.functionspace(msh, element("Lagrange", cell, 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    p = ufl.TrialFunction(Q)
    q = ufl.TestFunction(Q)
    return {
        "velocity": base.to_scipy_matrix(fem.assemble_matrix(fem.form(ufl.inner(u, v) * ufl.dx))).tocsr(),
        "pressure": base.to_scipy_matrix(fem.assemble_matrix(fem.form(p * q * ufl.dx))).tocsr(),
    }


def mass_l2_relative(diff: np.ndarray, ref: np.ndarray, mass: sp.csr_matrix) -> float:
    diff = np.asarray(diff, dtype=float)
    ref = np.asarray(ref, dtype=float)
    abs_sq = float(diff @ (mass @ diff))
    ref_sq = float(ref @ (mass @ ref))
    return float(np.sqrt(max(abs_sq, 0.0)) / max(np.sqrt(max(ref_sq, 0.0)), 1.0e-300))


def mass_l2_absolute(diff: np.ndarray, mass: sp.csr_matrix) -> float:
    diff = np.asarray(diff, dtype=float)
    abs_sq = float(diff @ (mass @ diff))
    return float(np.sqrt(max(abs_sq, 0.0)))


def scalar_mass(values: np.ndarray, mass: sp.csr_matrix) -> float:
    ones = np.ones(len(values), dtype=float)
    return float(ones @ (mass @ np.asarray(values, dtype=float)))


def crossing_time(time_values: np.ndarray, signal: np.ndarray, level: float) -> float:
    above = np.flatnonzero(signal >= level)
    if len(above) == 0:
        return float("nan")
    idx = int(above[0])
    if idx == 0:
        return float(time_values[0])
    t0, t1 = float(time_values[idx - 1]), float(time_values[idx])
    y0, y1 = float(signal[idx - 1]), float(signal[idx])
    if abs(y1 - y0) <= 1.0e-14:
        return t1
    return float(t0 + (level - y0) * (t1 - t0) / (y1 - y0))


def add_reference_errors(
    rows: list[dict[str, object]],
    histories: dict[str, dict[str, np.ndarray]],
    final_dof_fields: dict[str, np.ndarray],
    mass_matrix: sp.csr_matrix,
    reference: str,
) -> None:
    ref_curve = histories[reference]["cout"]
    ref_time = histories[reference]["time"]
    ref_row = next(row for row in rows if row["method"] == reference)
    ref_field = final_dof_fields[reference]
    ref_mass = next(row for row in rows if row["method"] == reference)["final_tracer_mass"]
    for row in rows:
        method = str(row["method"])
        signal = histories[method]["cout"]
        curve_diff = signal - ref_curve
        field_diff = final_dof_fields[method] - ref_field
        row["breakthrough_rel_l2_error"] = float(np.linalg.norm(curve_diff) / max(np.linalg.norm(ref_curve), 1.0e-300))
        row["breakthrough_abs_l1_error"] = float(np.trapezoid(np.abs(curve_diff), ref_time))
        row["concentration_l2_abs_fem_integral_vs_fem"] = mass_l2_absolute(field_diff, mass_matrix)
        row["concentration_rel_l2_fem_integral_vs_fem"] = mass_l2_relative(field_diff, ref_field, mass_matrix)
        row["concentration_linf_abs_vs_fem"] = float(np.linalg.norm(field_diff, ord=np.inf))
        row["final_mass_error_vs_fem"] = float(row["final_tracer_mass"] - ref_mass)
        row["final_mass_rel_error_vs_fem"] = float((row["final_tracer_mass"] - ref_mass) / max(abs(float(ref_mass)), 1.0e-300))
        row["t10_error"] = signed_time_error(row["t10"], ref_row["t10"])
        row["t50_error"] = signed_time_error(row["t50"], ref_row["t50"])
        row["t90_error"] = signed_time_error(row["t90"], ref_row["t90"])


def attach_stokes_metrics(tracer_rows: list[dict[str, object]], stokes_rows: list[dict[str, object]]) -> None:
    by_method = {str(row["method"]): row for row in stokes_rows}
    for row in tracer_rows:
        stokes = by_method.get(str(row["method"]), {})
        row["stokes_solve_time_seconds"] = stokes.get("stokes_solve_time_seconds", "")
        row["active_dofs"] = stokes.get("active_dofs", "")
        row["active_dof_ratio"] = stokes.get("active_dof_ratio", "")
        row["velocity_rel_l2_error_vs_fem"] = stokes.get("velocity_rel_l2_error_vs_fem", "")
        row["pressure_rel_l2_error_vs_fem"] = stokes.get("pressure_rel_l2_error_vs_fem", "")
        row["velocity_rel_l2_fem_integral_vs_fem"] = stokes.get("velocity_rel_l2_fem_integral_vs_fem", "")
        row["pressure_rel_l2_fem_integral_vs_fem"] = stokes.get("pressure_rel_l2_fem_integral_vs_fem", "")


def signed_time_error(value, ref_value) -> float:
    if value != value or ref_value != ref_value:
        return float("nan")
    return float(value - ref_value)


def save_concentration_vtu(mesh, velocity: np.ndarray, concentration: np.ndarray, out: Path) -> None:
    cells = np.hstack([np.full((len(mesh.cells), 1), 4, dtype=np.int64), mesh.cells]).ravel()
    celltypes = np.full(len(mesh.cells), pv.CellType.TETRA, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, celltypes, mesh.coords)
    grid.point_data["velocity"] = velocity
    grid.point_data["concentration"] = concentration
    grid.save(out)


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#2f3640",
            "axes.labelcolor": "#1f2933",
            "xtick.color": "#1f2933",
            "ytick.color": "#1f2933",
            "grid.color": "#d8dee9",
            "grid.linewidth": 0.8,
            "font.size": 10,
            "legend.frameon": False,
        }
    )


def plot_breakthrough(out: Path, histories: dict[str, dict[str, np.ndarray]]) -> None:
    set_plot_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=220)
    for method, history in histories.items():
        ax.plot(
            history["time"],
            history["cout"],
            color=COLORS.get(method, "#555555"),
            linestyle="-" if method == "FEM" else "--",
            linewidth=2.0,
            label=method,
        )
    ax.set_xlabel("time")
    ax.set_ylabel("flux-weighted outlet concentration")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(True, alpha=0.7)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_mass_balance(out: Path, histories: dict[str, dict[str, np.ndarray]]) -> None:
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=220)
    for method, history in histories.items():
        color = COLORS.get(method, "#555555")
        axes[0].plot(history["time"], history["mass"], color=color, linewidth=2.0, label=f"{method} mass")
        axes[0].plot(history["time"], history["budget_mass"], color=color, linewidth=1.4, linestyle=":", label=f"{method} budget")
        axes[1].semilogy(
            history["time"][1:],
            np.maximum(np.abs(history["mass_balance_relative_residual"][1:]), 1.0e-18),
            color=color,
            linewidth=2.0,
            label=method,
        )
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("total tracer mass")
    axes[0].grid(True, alpha=0.7)
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("relative mass-balance residual")
    axes[1].grid(True, which="both", alpha=0.7)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_error_summary(out: Path, rows: list[dict[str, object]]) -> None:
    set_plot_style()
    methods = [str(row["method"]) for row in rows if row["method"] != "FEM"]
    if not methods:
        return
    concentration = [float(row["concentration_rel_l2_fem_integral_vs_fem"]) for row in rows if row["method"] != "FEM"]
    breakthrough = [float(row["breakthrough_rel_l2_error"]) for row in rows if row["method"] != "FEM"]
    mass = [abs(float(row["final_mass_rel_error_vs_fem"])) for row in rows if row["method"] != "FEM"]
    balance = [float(row["max_mass_balance_relative_residual"]) for row in rows if row["method"] != "FEM"]
    x = np.arange(len(methods))
    width = 0.18
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=220)
    ax.bar(x - 1.5 * width, concentration, width, label="field L2", color="#457b9d")
    ax.bar(x - 0.5 * width, breakthrough, width, label="breakthrough L2", color="#e76f51")
    ax.bar(x + 0.5 * width, mass, width, label="final mass", color="#2a9d8f")
    ax.bar(x + 1.5 * width, balance, width, label="mass balance", color="#6d597a")
    ax.set_yscale("log")
    ax.set_xticks(x, methods)
    ax.set_ylabel("relative error")
    ax.grid(True, axis="y", which="both", alpha=0.7)
    ax.legend(ncols=2)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_final_concentration(out: Path, mesh, final_fields: dict[str, np.ndarray], methods: list[str], reference: str) -> None:
    panels = list(methods)
    if "HODDPNM" in final_fields and reference in final_fields:
        panels.append("log10 error")

    use_smooth_grid = hasattr(mesh, "pore_voxels") and hasattr(mesh, "domain_size") and hasattr(mesh, "voxel_size")
    if use_smooth_grid:
        plot_fields = {
            method: build_smoothed_tracer_image(
                np.asarray(mesh.coords, dtype=float),
                np.asarray(final_fields[method], dtype=float),
                mesh,
                f"concentration_{method}",
                log_scale=False,
                sigma=1.85,
            )
            for method in methods
        }
        if "HODDPNM" in final_fields and reference in final_fields:
            error = np.abs(np.asarray(final_fields["HODDPNM"], dtype=float) - np.asarray(final_fields[reference], dtype=float))
            plot_fields["log10 error"] = build_smoothed_tracer_image(
                np.asarray(mesh.coords, dtype=float),
                error,
                mesh,
                "log10_concentration_error",
                log_scale=True,
                sigma=2.10,
            )
        shell = build_tracer_solid_surface(mesh)
    else:
        grid = build_tracer_plot_grid(mesh)
        for method in methods:
            grid.point_data[f"concentration_{method}"] = np.asarray(final_fields[method], dtype=float)
        if "HODDPNM" in final_fields and reference in final_fields:
            error = np.abs(np.asarray(final_fields["HODDPNM"], dtype=float) - np.asarray(final_fields[reference], dtype=float))
            grid.point_data["log10_concentration_error"] = np.log10(np.maximum(error, 1.0e-16))
        plot_fields = {method: grid for method in methods}
        if "HODDPNM" in final_fields and reference in final_fields:
            plot_fields["log10 error"] = grid
        shell = plot_extract_surface(grid).smooth(n_iter=50, relaxation_factor=0.06, feature_smoothing=False, boundary_smoothing=True)

    pv.global_theme.font.family = "arial"
    plotter = pv.Plotter(off_screen=True, window_size=(640 * len(panels), 820), shape=(1, len(panels)), border=False)
    plotter.set_background("white")
    for index, panel in enumerate(panels):
        plotter.subplot(0, index)
        if panel == "log10 error":
            field = plot_fields[panel]
            render_tracer_isosurface_panel(
                plotter,
                field,
                shell,
                "log10_concentration_error",
                "HODDPNM error",
                "log10 error",
                "turbo",
                tracer_error_range(field.point_data["log10_concentration_error"]),
            )
        else:
            field = plot_fields[panel]
            render_tracer_isosurface_panel(
                plotter,
                field,
                shell,
                f"concentration_{panel}",
                panel,
                "c",
                "viridis",
                (0.0, 1.0),
            )
    plotter.screenshot(str(out), transparent_background=False)
    plotter.close()


def build_tracer_plot_grid(mesh) -> pv.UnstructuredGrid:
    cells = np.hstack([np.full((len(mesh.cells), 1), 4, dtype=np.int64), mesh.cells]).ravel()
    celltypes = np.full(len(mesh.cells), pv.CellType.TETRA, dtype=np.uint8)
    return pv.UnstructuredGrid(cells, celltypes, mesh.coords)


def build_smoothed_tracer_image(
    sample_points: np.ndarray,
    sample_values: np.ndarray,
    mesh,
    scalar_name: str,
    log_scale: bool,
    sigma: float,
    n: int = 144,
) -> pv.ImageData:
    domain_size = np.asarray(mesh.domain_size, dtype=float)
    spacing = tuple((domain_size / (n - 1)).tolist())
    image = pv.ImageData(dimensions=(n, n, n), spacing=spacing, origin=(0.0, 0.0, 0.0))
    points = image.points
    values = idw_tracer(points, sample_points, sample_values, k=12)
    volume = values.reshape((n, n, n), order="F")
    fluid = tracer_fluid_mask(points, mesh).reshape((n, n, n), order="F")
    fluid_weight = gaussian_filter(fluid.astype(float), sigma=1.65)

    smooth = gaussian_filter(volume, sigma=sigma)

    if log_scale:
        finite = smooth[np.isfinite(smooth) & (fluid_weight > 0.16)]
        positive = finite[finite > 0.0]
        floor = max(float(np.percentile(positive, 2.0)) * 0.1 if len(positive) else 1.0e-16, 1.0e-16)
        smooth = np.log10(np.maximum(smooth, floor))
    else:
        smooth = np.clip(smooth, 0.0, 1.0)
    image.point_data[scalar_name] = smooth.reshape(-1, order="F")
    image.point_data["fluid_weight"] = fluid_weight.reshape(-1, order="F")
    return image


def build_tracer_solid_surface(mesh, n: int = 144) -> pv.PolyData:
    domain_size = np.asarray(mesh.domain_size, dtype=float)
    spacing = tuple((domain_size / (n - 1)).tolist())
    image = pv.ImageData(dimensions=(n, n, n), spacing=spacing, origin=(0.0, 0.0, 0.0))
    solid = (~tracer_fluid_mask(image.points, mesh)).astype(float).reshape((n, n, n), order="F")
    solid = gaussian_filter(solid, sigma=1.85)
    image.point_data["solid"] = solid.reshape(-1, order="F")
    surface = image.contour(isosurfaces=[0.5], scalars="solid")
    return surface.smooth(n_iter=150, relaxation_factor=0.08, feature_smoothing=False, boundary_smoothing=True)


def render_tracer_isosurface_panel(
    plotter: pv.Plotter,
    grid,
    shell: pv.PolyData,
    scalars: str,
    title: str,
    scalar_bar_title: str,
    cmap: str,
    clim: tuple[float, float],
) -> None:
    add_tracer_domain_box(plotter, grid.bounds)
    shell_opacity = 0.12 if scalar_bar_title == "c" else 0.30
    plotter.add_mesh(shell, color="#787878", opacity=shell_opacity, smooth_shading=True, show_edges=False)
    scalar_bar_args = {
        "title": scalar_bar_title,
        "vertical": True,
        "position_x": 0.87,
        "position_y": 0.18,
        "width": 0.045,
        "height": 0.60,
        "title_font_size": 15,
        "label_font_size": 12,
        "fmt": "%.1f",
        "color": "black",
    }
    contour = grid.contour(isosurfaces=tracer_iso_levels(clim, scalar_bar_title), scalars=scalars)
    if contour.n_points and "fluid_weight" in contour.point_data:
        contour = contour.threshold(value=0.18, scalars="fluid_weight")
    if contour.n_points:
        contour_surface = plot_extract_surface(contour)
        render_mesh = contour_surface.smooth(n_iter=90, relaxation_factor=0.06, feature_smoothing=False, boundary_smoothing=True)
        opacity = 0.74 if scalar_bar_title == "c" else 0.60
    else:
        render_mesh = grid
        opacity = 0.55
    plotter.add_mesh(
        render_mesh,
        scalars=scalars,
        cmap=cmap,
        clim=clim,
        opacity=opacity,
        smooth_shading=True,
        show_edges=False,
        scalar_bar_args=scalar_bar_args,
    )
    plotter.add_text(title, position=(0.035, 0.925), font_size=14, color="black", viewport=True)
    set_tracer_paper_camera(plotter, grid.bounds)


def tracer_iso_levels(clim: tuple[float, float], scalar_bar_title: str = "") -> np.ndarray:
    lo, hi = clim
    if hi <= lo:
        hi = lo + 1.0
    if scalar_bar_title == "c":
        return np.linspace(0.25, 0.95, 6)
    return np.linspace(lo + 0.24 * (hi - lo), hi - 0.10 * (hi - lo), 5)


def tracer_error_range(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values[np.isfinite(values)], dtype=float)
    if len(finite) == 0:
        return -16.0, -8.0
    lo = float(np.percentile(finite, 5.0))
    hi = float(np.percentile(finite, 99.2))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def idw_tracer(query_points: np.ndarray, sample_points: np.ndarray, sample_values: np.ndarray, k: int) -> np.ndarray:
    tree = cKDTree(sample_points)
    distances, indices = tree.query(query_points, k=min(k, len(sample_points)), workers=-1)
    if np.ndim(indices) == 1:
        indices = indices[:, None]
        distances = distances[:, None]
    weights = 1.0 / np.maximum(distances, 1.0e-8) ** 2
    return np.sum(weights * sample_values[indices], axis=1) / np.sum(weights, axis=1)


def tracer_fluid_mask(points: np.ndarray, mesh) -> np.ndarray:
    pore = np.zeros(mesh.domain_shape, dtype=bool)
    pore[tuple(mesh.pore_voxels.T)] = True
    ijk = np.floor(np.asarray(points, dtype=float) / mesh.voxel_size).astype(int)
    ijk = np.clip(ijk, 0, np.asarray(mesh.domain_shape) - 1)
    return pore[ijk[:, 0], ijk[:, 1], ijk[:, 2]]


def plot_extract_surface(dataset):
    try:
        return dataset.extract_surface(algorithm="dataset_surface")
    except TypeError:
        return dataset.extract_surface()


def add_tracer_domain_box(plotter: pv.Plotter, bounds: tuple[float, float, float, float, float, float]) -> None:
    cube = plot_extract_surface(pv.Cube(bounds=bounds))
    plotter.add_mesh(cube, color="#eeeeee", opacity=0.08, show_edges=False)


def set_tracer_paper_camera(plotter: pv.Plotter, bounds: tuple[float, float, float, float, float, float]) -> None:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    center = (0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax))
    scale = max(xmax - xmin, ymax - ymin, zmax - zmin)
    plotter.camera_position = [
        (center[0] + 4.10 * scale, center[1] - 4.00 * scale, center[2] + 3.10 * scale),
        center,
        (0.0, 0.0, 1.0),
    ]
    plotter.enable_parallel_projection()
    plotter.camera.zoom(1.45)
    plotter.add_light(pv.Light(position=(center[0], center[1] - 2.0 * scale, center[2] + 4.0 * scale), intensity=0.55))
    plotter.add_light(pv.Light(position=(center[0] - 2.0 * scale, center[1] + 1.5 * scale, center[2] + 2.5 * scale), intensity=0.35))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_history_csv(path: Path, histories: dict[str, dict[str, np.ndarray]]) -> None:
    rows: list[dict[str, object]] = []
    for method, history in histories.items():
        for i, t in enumerate(history["time"]):
            rows.append(
                {
                    "method": method,
                    "time": float(t),
                    "outlet_concentration": float(history["cout"][i]),
                    "mass": float(history["mass"][i]),
                    "budget_mass": float(history["budget_mass"][i]),
                    "raw_mass": float(history["raw_mass"][i]),
                    "mass_rate": float(history["mass_rate"][i]),
                    "transport_rate": float(history["transport_rate"][i]),
                    "dirichlet_source_rate": float(history["dirichlet_source_rate"][i]),
                    "mass_balance_residual_rate": float(history["mass_balance_residual_rate"][i]),
                    "mass_balance_relative_residual": float(history["mass_balance_relative_residual"][i]),
                    "limiter_mass_change_abs": float(history["limiter_mass_change_abs"][i]),
                    "limiter_mass_residual": float(history["limiter_mass_residual"][i]),
                    "min_c": float(history["min_c"][i]),
                    "max_c": float(history["max_c"][i]),
                    "raw_min_c_before_limiter": float(history["raw_min_c_before_limiter"][i]),
                    "raw_max_c_before_limiter": float(history["raw_max_c_before_limiter"][i]),
                    "raw_below_zero_before_limiter": int(history["raw_below_zero_before_limiter"][i]),
                    "raw_above_one_before_limiter": int(history["raw_above_one_before_limiter"][i]),
                }
            )
    write_csv(path, rows)


def write_report(path: Path, summary: dict[str, object]) -> None:
    rows = summary["tracer_metrics"]
    lines = [
        "# Single-Phase Stokes-Tracer HODDPNM Validation",
        "",
        "This run removes adaptive method levels and compares FEM-Stokes tracer against a fixed HODDPNM Schur/static-condensation solve of the same FEniCSx Taylor-Hood matrix.",
        "",
        "HODDPNM is the Schur-complement/static-condensation path: known Dirichlet dofs are removed first, interior velocity/interior pressure unknowns are eliminated, the free interface velocity/interface-pressure Schur problem is solved, and the full field is reconstructed.",
        "",
        "The current implementation demonstrates clear dof compression and acceleration using Schur diagonal scaling plus modest pressure stabilization. The Schur iteration count is still O(10^2), so a scalable Schur preconditioner remains the next optimization target.",
        "",
        "## Stokes Compression",
        "",
        "| method | Stokes time (s) | active dofs | active ratio | Krylov iterations | Schur matvecs | Schur residual | known fixed dofs | eliminated free interior dofs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["stokes_velocity_cases"]:
        known = row.get("hoddpnm_known_fixed_dofs", "")
        eliminated = row.get("hoddpnm_free_interior_dofs_eliminated", row.get("hoddpnm_interior_dofs_eliminated", ""))
        iterations = row.get("schur_iterations", "")
        matvecs = row.get("schur_matvecs", "")
        residual = row.get("schur_relative_residual", "")
        residual_text = "" if residual == "" else f"{float(residual):.3e}"
        lines.append(
            f"| {row['method']} | {float(row['stokes_solve_time_seconds']):.3f} | "
            f"{row['active_dofs']} | "
            f"{float(row['active_dof_ratio']):.1%} | "
            f"{iterations} | {matvecs} | {residual_text} | {known} | {eliminated} |"
        )
    lines.extend(
        [
            "",
        "## Main Metrics",
        "",
        "| method | tracer time (s) | breakthrough rel L2 | concentration rel L2 | final mass rel err | max balance residual | final range | raw limiter hits | t90 |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|",
        ]
    )
    for row in rows:
        raw_hits = int(row["raw_final_concentration_below_zero_before_limiter"]) + int(row["raw_final_concentration_above_one_before_limiter"])
        lines.append(
            f"| {row['method']} | {row['tracer_time_seconds']:.3f} | "
            f"{row['breakthrough_rel_l2_error']:.3e} | "
            f"{row['concentration_rel_l2_fem_integral_vs_fem']:.3e} | "
            f"{row['final_mass_rel_error_vs_fem']:.3e} | "
            f"{row['max_mass_balance_relative_residual']:.3e} | "
            f"[{row['final_concentration_min']:.3f}, {row['final_concentration_max']:.3f}] | "
            f"{raw_hits} | "
            f"{row['t90']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Breakthrough curves: `{summary['outputs']['breakthrough_curves']}`",
            f"- Mass balance validation: `{summary['outputs']['mass_balance_validation']}`",
            f"- Error summary: `{summary['outputs']['tracer_error_summary']}`",
            f"- Final concentration and error: `{summary['outputs']['final_concentration_and_error']}`",
            f"- CSV metrics: `{summary['outputs']['tracer_metrics_csv']}`",
            f"- Mass history CSV: `{summary['outputs']['mass_balance_history_csv']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()


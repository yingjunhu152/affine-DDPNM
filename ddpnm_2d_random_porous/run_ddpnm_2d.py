from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from ddpnm2d.geometry import build_partition
from ddpnm2d.report import write_outputs
from ddpnm2d.solver import solve_ddpnm, solve_reference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the 2D DD-PNM algorithm on the fixed random porous medium."
    )
    parser.add_argument("--mesh-size", type=float, default=0.04)
    parser.add_argument("--wall-size", type=float, default=0.018)
    parser.add_argument("--throat-size", type=float, default=0.010)
    parser.add_argument("--wall-band", type=float, default=0.065)
    parser.add_argument("--throat-band", type=float, default=0.045)
    parser.add_argument("--interface-order", type=int, choices=(0, 1), default=1)
    parser.add_argument("--viscosity", type=float, default=1.0)
    parser.add_argument("--inlet-pressure", type=float, default=1.0)
    parser.add_argument("--outlet-pressure", type=float, default=0.0)
    parser.add_argument("--pressure-stabilization", type=float, default=1.0e-10)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/enriched"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    print("[1/4] Imposing analytic saddle cuts and generating the locally refined mesh...")
    partition = build_partition(
        mesh_size=args.mesh_size,
        wall_size=args.wall_size,
        throat_size=args.throat_size,
        wall_band=args.wall_band,
        throat_band=args.throat_band,
    )
    print(
        f"      {len(set(partition.cell_labels.tolist()))} subdomains, "
        f"{len(partition.interface_pairs)} internal interfaces"
    )
    print("[2/4] Computing local P2-P1 Stokes unit-traction responses and DtN maps...")
    ddpnm = solve_ddpnm(
        partition,
        viscosity=args.viscosity,
        inlet_pressure=args.inlet_pressure,
        outlet_pressure=args.outlet_pressure,
        pressure_stabilization=args.pressure_stabilization,
        interface_order=args.interface_order,
    )
    print(
        f"      Schur size {len(ddpnm.interface_pressures)}, "
        f"max interface flux residual {ddpnm.max_mass_residual:.3e}"
    )
    print("[3/4] Solving monolithic Taylor-Hood reference problem...")
    reference = solve_reference(
        partition.mesh,
        viscosity=args.viscosity,
        inlet_pressure=args.inlet_pressure,
        outlet_pressure=args.outlet_pressure,
        pressure_stabilization=args.pressure_stabilization,
    )
    print("[4/4] Reconstructing fields and writing validation outputs...")
    parameters = {
        "mesh_size": args.mesh_size,
        "wall_size": args.wall_size,
        "throat_size": args.throat_size,
        "wall_band": args.wall_band,
        "throat_band": args.throat_band,
        "interface_order": args.interface_order,
        "viscosity": args.viscosity,
        "inlet_pressure": args.inlet_pressure,
        "outlet_pressure": args.outlet_pressure,
        "pressure_stabilization": args.pressure_stabilization,
    }
    report = write_outputs(partition, ddpnm, reference, args.out_dir, parameters)
    elapsed = time.perf_counter() - started
    report["wall_time_seconds"] = elapsed
    with (args.out_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    validation = report["validation"]
    print(f"Done in {elapsed:.2f} s")
    print(
        "Vertex-sampled relative errors: "
        f"velocity={validation['vertex_velocity_relative_l2']:.3e}, "
        f"pressure={validation['vertex_pressure_mean_aligned_relative_l2']:.3e}"
    )
    print(f"Outputs: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()

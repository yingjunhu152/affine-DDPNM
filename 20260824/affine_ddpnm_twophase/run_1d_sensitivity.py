#!/usr/bin/env python3
"""Velocity-perturbation sensitivity of the 1-D Buckley--Leverett recovery,
compared against the 3-D empirical error transfer (0.6--14.9 pp recovery bias).

Theory anchors
--------------
1. **Mass-balance law (exact)**: pre-breakthrough, R(t) = (fw(S_inj)-fw(Swi))/
   (1-Swi) * PVI(t) = 1.25 * PVI(t), where PVI = (1/V) int u_inlet dt.  The
   recovery is set *only* by the inlet (throughput) flux -- a zero-mean
   interior velocity perturbation that leaves u(0) unchanged cannot move R.
2. **Throughput channel**: a perturbation with inlet value delta(0) moves the
   recovery by dR = 1.25 * PVI_ref * delta(0) (first order, exact in 1-D).
3. **Leak channel (3-D only)**: multi-path flow lets water leave the outlet
   early (fast streamlines), so the measured recovery falls short of the
   1.25*PVI line: R = 1.25*PVI - leak.  The 3-D deviation decomposes exactly
   as  dR = 1.25*dPVI - d_leak.

3-D empirical decomposition (from outputs/benchmark_twophase/):
   method   eps(vel rel L2)  PVI/PVI_FEM-1 (=eta)   dR measured
   Classic        0.6532          0.732                +0.1488
   W1n-3          0.3075          0.147                +0.0280
   Affine-9       0.0651          0.028                +0.0060
   theta = eta/eps: 1.12 (Classic), 0.48 (W1n), 0.43 (Affine).

This script verifies laws 1-2 numerically with the P1-CG mirror solver
(variable total velocity) and decomposes the 3-D data exactly.

Run: conda run -n fenicsx --no-capture-output python run_1d_sensitivity.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from validate_1d_buckley_leverett import COREY, S_INJ, SWI, solve_1d_blev, welge_shock

L, NX, DT = 10.0, 400, 0.0025  # CFL ~0.28 at eps=0.65 (pure-shock P1-CG limit ~0.5)
T_FINAL = 2.3            # PVI_ref = u*T/L = 0.23 (matches the 3-D benchmark)
N_PORES = 27             # same pore count as the random-27 medium
PVI_REF = 1.0 * T_FINAL / L
R_COEF = (1.0 / (1.0 - SWI)) * 1.0    # (fw(S_inj)-fw(Swi))/(1-Swi) = 1.25
DRAWS_STRUCTURE = 12
DRAWS_MIXED = 8
SEEDS_3D = {
    "Classic-DDPNM-1": {"eps": 0.653203, "pvi": 0.3998, "r": 0.4279, "flux_err": 0.7272},
    "NormalLinear-DDPNM-3": {"eps": 0.307488, "pvi": 0.2646, "r": 0.3071, "flux_err": 0.1465},
    "Affine-DDPNM-9": {"eps": 0.065058, "pvi": 0.2372, "r": 0.2851, "flux_err": 0.0271},
}
PVI_FEM_3D = 0.2308
R_FEM_3D = 0.2791


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir", type=Path,
        default=PROJECT_DIR / "outputs" / "sensitivity_1d",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_DIR / "outputs" / "benchmark_twophase" / "affine_ddpnm_twophase_report.json",
    )
    return parser.parse_args()


def pore_indices(x: np.ndarray) -> np.ndarray:
    return np.minimum(np.floor(x / (L / N_PORES)).astype(int), N_PORES - 1)


def unit_norm(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values ** 2)))


def structure_delta(eps: float, rng: np.random.Generator) -> np.ndarray:
    """Zero-mean per-pore +-1 noise on pores 1..N_PORES-1, delta(0)=0,
    normalized to relative L2 norm eps over the whole domain."""
    x = np.linspace(0.0, L, NX + 1)  # P1 dof coords of the interval mesh
    pore = pore_indices(x)
    noise = rng.choice(np.asarray([-1.0, 1.0]), size=N_PORES).astype(float)
    noise[0] = 0.0
    noise -= noise[1:].mean()       # exactly zero mean over pores 1..
    delta = noise[pore]
    delta = delta / unit_norm(delta) * eps
    delta[x < 1.0e-12] = 0.0        # exact zero at the inlet dof
    return delta


def mixed_delta(eps: float, theta: float, rng: np.random.Generator) -> np.ndarray:
    """delta = theta*eps (coherent, incl. inlet) + sqrt(1-theta^2)*eps*noise
    (zero-mean, zero at inlet): relative L2 norm eps, inlet value theta*eps."""
    delta = np.full(NX + 1, theta * eps)
    noise = structure_delta(1.0, rng)
    delta = delta + float(np.sqrt(1.0 - theta ** 2)) * eps * noise
    return delta


def front_position(s: np.ndarray, x: np.ndarray, s_star: float) -> float:
    below = np.flatnonzero(s < s_star)
    if len(below) == 0:
        return x[-1]
    i0 = int(below[0])
    if i0 == 0:
        return x[0]
    s0, s1 = float(s[i0 - 1]), float(s[i0])
    if abs(s1 - s0) <= 1.0e-14:
        return x[i0]
    return x[i0 - 1] + (s_star - s0) * (x[i0] - x[i0 - 1]) / (s1 - s0)


def field_rel_l2(s: np.ndarray, s_ref: np.ndarray, x: np.ndarray) -> float:
    h = x[1] - x[0]
    num = float(np.sqrt(np.sum((s - s_ref) ** 2) * h))
    denom = float(np.sqrt(np.sum(s_ref ** 2) * h))
    return num / denom


def run_case(u_values: np.ndarray | None, s_ref: np.ndarray | None,
             x: np.ndarray, s_star: float, rng=None) -> dict:
    result = solve_1d_blev(L, NX, 1.0, DT, T_FINAL, COREY, swi=SWI, s_inj=S_INJ,
                           u_values=u_values)
    s = np.asarray(result["final_saturation"], dtype=float)
    masses = np.asarray(result["masses"], dtype=float)
    recovery = float((masses[-1] - masses[0]) / (L * (1.0 - SWI)))
    u0 = 1.0 if u_values is None else float(np.asarray(u_values)[0])
    pvi = u0 * T_FINAL / L
    return {
        "recovery": recovery,
        "pvi": pvi,
        "r_over_125pvi": recovery / (1.25 * pvi),
        "front": front_position(s, x, s_star),
        "field_rel_l2": field_rel_l2(s, s_ref, x) if s_ref is not None else 0.0,
        "s_min": float(np.min(s)),
        "s_max": float(np.max(s)),
        "max_balance_resid": float(np.max(np.abs(result["relative_conservation_residual"]))),
        "saturation": s,
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0_all = time.perf_counter()
    v_s, s_star = welge_shock(COREY, SWI, S_INJ)
    x = np.linspace(0.0, L, NX + 1)
    rng = np.random.default_rng(20260812)
    rows: list[dict[str, object]] = []

    # ---- reference run -----------------------------------------------------
    ref = run_case(None, None, x, s_star)
    s_ref = ref["saturation"]
    rows.append({"family": "reference", "eps": 0.0, "theta": 0.0, "draw": 0,
                 "dR": 0.0, "dPVI": 0.0, "r_over_125pvi": ref["r_over_125pvi"],
                 "field_rel_l2": 0.0, "front_dev": 0.0})
    print(f"reference: R={ref['recovery']:.5f}, PVI={ref['pvi']:.5f}, "
          f"R/(1.25 PVI)={ref['r_over_125pvi']:.10f}, front={ref['front']:.4f} "
          f"(Welge {v_s * T_FINAL:.4f}), balance={ref['max_balance_resid']:.1e}")

    # ---- flux / coherent family (theta = 1) -------------------------------
    for eps in (0.03, 0.065, 0.15, 0.3, 0.65):
        case = run_case(np.full(NX + 1, 1.0 + eps), s_ref, x, s_star)
        rows.append({"family": "coherent", "eps": eps, "theta": 1.0, "draw": 0,
                     "dR": case["recovery"] - ref["recovery"],
                     "dPVI": case["pvi"] - ref["pvi"],
                     "r_over_125pvi": case["r_over_125pvi"],
                     "field_rel_l2": case["field_rel_l2"],
                     "front_dev": case["front"] - ref["front"]})
        print(f"coherent eps={eps}: dR={case['recovery'] - ref['recovery']:+.6f} "
              f"vs 1.25*dPVI={1.25 * (case['pvi'] - ref['pvi']):+.6f}")

    # ---- pure structure family (theta = 0): dR must vanish ----------------
    for eps in (0.065, 0.6532):
        for draw in range(DRAWS_STRUCTURE):
            delta = structure_delta(eps, rng)
            case = run_case(1.0 + delta, s_ref, x, s_star)
            rows.append({"family": "structure", "eps": eps, "theta": 0.0, "draw": draw,
                         "dR": case["recovery"] - ref["recovery"],
                         "dPVI": case["pvi"] - ref["pvi"],
                         "r_over_125pvi": case["r_over_125pvi"],
                         "field_rel_l2": case["field_rel_l2"],
                         "front_dev": case["front"] - ref["front"]})
    # ---- mixed family (theta = 0.45, the W1n/Affine coherent fraction) ----
    for eps in (0.0651, 0.3075, 0.6532):
        for draw in range(DRAWS_MIXED):
            delta = mixed_delta(eps, 0.45, rng)
            case = run_case(1.0 + delta, s_ref, x, s_star)
            rows.append({"family": "mixed", "eps": eps, "theta": 0.45, "draw": draw,
                         "dR": case["recovery"] - ref["recovery"],
                         "dPVI": case["pvi"] - ref["pvi"],
                         "r_over_125pvi": case["r_over_125pvi"],
                         "field_rel_l2": case["field_rel_l2"],
                         "front_dev": case["front"] - ref["front"]})

    # ---- aggregates --------------------------------------------------------
    def agg(family: str, eps: float) -> dict[str, float]:
        group = [r for r in rows if r["family"] == family and abs(r["eps"] - eps) < 1e-12]
        return {
            "dR_mean": float(np.mean([r["dR"] for r in group])),
            "dR_std": float(np.std([r["dR"] for r in group])),
            "field_l2_mean": float(np.mean([r["field_rel_l2"] for r in group])),
            "front_dev_mean": float(np.mean([r["front_dev"] for r in group])),
            "n": len(group),
        }

    # ---- 3-D empirical decomposition --------------------------------------
    report = json.loads(args.report.read_text(encoding="utf-8"))
    empirical: dict[str, object] = {}
    for method, seed in SEEDS_3D.items():
        eta = seed["pvi"] / PVI_FEM_3D - 1.0
        dR_meas = seed["r"] - R_FEM_3D
        dR_throughput = 1.25 * (seed["pvi"] - PVI_FEM_3D)
        leak = dR_throughput - dR_meas
        empirical[method] = {
            "eps": seed["eps"],
            "eta": eta,
            "theta": eta / seed["eps"],
            "dR_measured": dR_meas,
            "dR_throughput_125dPVI": dR_throughput,
            "leak": leak,
            "leak_over_eps": leak / seed["eps"],
        }
        print(
            f"3D {method}: eps={seed['eps']:.4f} eta={eta:.4f} theta={eta / seed['eps']:.2f} "
            f"dR={dR_meas:+.4f} = 1.25*dPVI({dR_throughput:+.4f}) - leak({leak:+.4f})"
        )

    summary = {
        "description": "1-D velocity-perturbation sensitivity of BL recovery vs 3-D empirical transfer",
        "law": "R = (fw(S_inj)-fw(Swi))/(1-Swi) * PVI = 1.25*PVI pre-breakthrough; dR = 1.25*dPVI",
        "decomposition": "dR = 1.25*dPVI - leak",
        "setup": {
            "L": L, "nx": NX, "dt": DT, "t_final": T_FINAL,
            "pvi_ref": PVI_REF, "corey": COREY, "swi": SWI, "s_inj": S_INJ,
            "welge_v_s": v_s, "welge_s_star": s_star,
        },
        "reference": {
            "recovery": ref["recovery"], "pvi": ref["pvi"],
            "r_over_125pvi": ref["r_over_125pvi"],
            "max_balance_resid": ref["max_balance_resid"],
        },
        "aggregates": {
            f"{family}_eps{eps}": agg(family, eps)
            for family, eps_list in (("structure", (0.065, 0.6532)), ("mixed", (0.0651, 0.3075, 0.6532)))
            for eps in eps_list
        },
        "empirical_3d": empirical,
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - t0_all,
    }
    (args.out_dir / "sensitivity_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )

    # ---- CSV ---------------------------------------------------------------
    csv_lines = [
        "family,eps,theta,draw,dR,dPVI,r_over_125pvi,field_rel_l2,front_dev"
    ]
    for r in rows:
        csv_lines.append(
            f"{r['family']},{r['eps']},{r['theta']},{r['draw']},"
            f"{r['dR']:.8e},{r['dPVI']:.8e},{r['r_over_125pvi']:.10f},"
            f"{r['field_rel_l2']:.6e},{r['front_dev']:.6e}"
        )
    (args.out_dir / "sensitivity_metrics.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    # ---- plots -------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.3))
    fig.patch.set_facecolor("#fcfcfb")
    ink, secondary = "#0b0b0b", "#52514e"
    method_colors = {
        "Classic-DDPNM-1": "#2a78d6", "NormalLinear-DDPNM-3": "#eb6834",
        "Affine-DDPNM-9": "#1baf7a",
    }

    ax = axes[0]
    ax.set_facecolor("#fcfcfb")
    for eps in (0.065, 0.6532):
        g = agg("structure", eps)
        ax.errorbar(eps, g["dR_mean"], yerr=2 * g["dR_std"], fmt="o", color="#898781",
                    markersize=6, capsize=3, label="structure (θ=0)" if eps == 0.065 else None)
    ax.axhline(0.0, color="#c3c2b7", linewidth=1.0)
    for eps in (0.0651, 0.3075, 0.6532):
        g = agg("mixed", eps)
        ax.plot(eps, g["dR_mean"], "s", color="#52514e", markersize=7,
                label="mixed (θ=0.45)" if eps == 0.0651 else None)
    for eps in (0.03, 0.065, 0.15, 0.3, 0.65):
        g = [r for r in rows if r["family"] == "coherent" and abs(r["eps"] - eps) < 1e-12][0]
        ax.plot(eps, g["dR"], "^", color="#0b0b0b", markersize=7,
                label="coherent (θ=1)" if eps == 0.03 else None)
    eps_line = np.linspace(0.0, 0.7, 50)
    for theta, style in ((0.45, "--"), (1.0, ":")):
        ax.plot(eps_line, 1.25 * PVI_REF * theta * eps_line, style, color="#898781",
                linewidth=1.4)
    ax.text(0.55, 1.25 * PVI_REF * 0.45 * 0.55 + 0.004, "1.25·PVI·θ·ε, θ=0.45",
            color="#52514e", fontsize=8)
    ax.text(0.55, 1.25 * PVI_REF * 1.0 * 0.55 - 0.016, "θ=1", color="#52514e", fontsize=8)
    for method, seed in empirical.items():
        dR_tp = seed["dR_throughput_125dPVI"]
        dR_meas = seed["dR_measured"]
        ax.plot([seed["eps"]], [dR_tp], "D", mfc="none", mec=method_colors[method],
                markersize=8, zorder=5)
        ax.plot([seed["eps"]], [dR_meas], "D", color=method_colors[method], markersize=8, zorder=5)
        ax.plot([seed["eps"], seed["eps"]], [dR_meas, dR_tp], color=method_colors[method],
                linewidth=1.2, zorder=4)
    ax.set_xlabel("velocity rel L2 error ε", color=secondary)
    ax.set_ylabel("recovery deviation ΔR", color=secondary)
    ax.set_title("1-D sensitivity: ΔR vs ε (theory lines, 3-D points)", color=ink, fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left", frameon=False)
    ax.grid(True, color="#e1e0d9", linewidth=0.6)

    ax = axes[1]
    methods = list(empirical)
    for i, method in enumerate(methods):
        seed = empirical[method]
        tp, leak, net = seed["dR_throughput_125dPVI"], seed["leak"], seed["dR_measured"]
        ax.bar(i, tp, width=0.55, color="#2a78d6", label="1.25·ΔPVI (throughput)" if i == 0 else None)
        ax.bar(i, -leak, width=0.55, bottom=tp, color="#e34948",
               label="−leak (early water out)" if i == 0 else None)
        ax.plot(i, net, "o", color="#0b0b0b", markersize=7,
                label="ΔR measured" if i == 0 else None)
    ax.axhline(0.0, color="#c3c2b7", linewidth=1.0)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.replace("-DDPNM", "").replace("NormalLinear", "W1n") for m in methods],
                       rotation=15, fontsize=8)
    ax.set_ylabel("recovery deviation", color=secondary)
    ax.set_title("3-D decomposition: ΔR = 1.25·ΔPVI − leak", color=ink, fontsize=10)
    ax.legend(fontsize=7.5, loc="lower right", frameon=False)
    ax.grid(True, axis="y", color="#e1e0d9", linewidth=0.6)

    ax = axes[2]
    for eps in (0.065, 0.6532):
        g = agg("structure", eps)
        ax.plot(eps, g["field_l2_mean"], "o", color="#eb6834", markersize=7)
    ax.plot([0.065, 0.6532], [0.065, 0.6532], ":", color="#898781", linewidth=1.2)
    ax.text(0.35, 0.30, "field L2 ≈ ε (structure)", color="#52514e", fontsize=8)
    for eps in (0.065, 0.6532):
        g = agg("structure", eps)
        ax.plot(eps, g["front_dev_mean"] / 10.0, "s", color="#1baf7a", markersize=7)
    ax.text(0.35, 0.10, "front dev / 10", color="#52514e", fontsize=8)
    ax.set_xlabel("ε (structure family)", color=secondary)
    ax.set_ylabel("field L2 / front deviation", color=secondary)
    ax.set_title("Structure perturbs fields, not recovery (ΔR≈0)", color=ink, fontsize=10)
    ax.grid(True, color="#e1e0d9", linewidth=0.6)

    fig.tight_layout()
    out_png = args.out_dir / "sensitivity_plot.png"
    fig.savefig(out_png, dpi=200, facecolor="#fcfcfb", bbox_inches="tight")
    print(f"Done: {args.out_dir.resolve()} ({time.perf_counter() - t0_all:.1f} s)")


if __name__ == "__main__":
    main()

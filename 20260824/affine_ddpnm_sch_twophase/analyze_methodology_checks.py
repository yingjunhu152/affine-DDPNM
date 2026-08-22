"""Summarize the viscous-form and initial-profile methodology checks."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "reports"

OPERATOR_CASES = {
    "random27": {
        "gradient": ROOT / "outputs/baseline_campaign_20260822/production/random27_dt_1",
        "symmetric": ROOT / "outputs/dev_smoke_random27_symmetric_20260822",
        "report": "random27_six_arms.json",
    },
    "bentheimer": {
        "gradient": ROOT / "outputs/baseline_campaign_20260822/pod/bentheimer_tol_1em08",
        "symmetric": ROOT / "outputs/dev_smoke_bentheimer_symmetric_20260822",
        "report": "bentheimer_six_arms.json",
    },
}

PROFILE_CASES = {
    "random27": {
        "discontinuous": ROOT / "outputs/baseline_campaign_20260822/time_step/random27_dt_0p25",
        "exponential": ROOT / "outputs/methodology_checks_20260822/initial_random27_exponential_l005",
        "report": "random27_six_arms.json",
    },
    "bentheimer": {
        "discontinuous": ROOT / "outputs/baseline_campaign_20260822/time_step/bentheimer_dt_0p25",
        "exponential": ROOT / "outputs/methodology_checks_20260822/initial_bentheimer_exponential_l005",
        "report": "bentheimer_six_arms.json",
    },
}


def _report(directory: Path, name: str) -> dict:
    return json.loads((directory / name).read_text(encoding="utf-8"))


def _row(report: dict, arm: str) -> dict:
    return next(item for item in report["results"] if item["arm"] == arm)


def _field(directory: Path, arm: str) -> dict[str, np.ndarray]:
    with np.load(directory / f"{arm}_final.npz") as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _vertex_weights(field: dict[str, np.ndarray]) -> np.ndarray:
    xyz = field["coordinates"][:, :3]
    cells = np.asarray(field["cells"], dtype=np.int32)
    tetra = xyz[cells]
    jacobian = tetra[:, 1:, :] - tetra[:, :1, :]
    volumes = np.abs(np.linalg.det(jacobian)) / 6.0
    return np.bincount(
        cells.reshape(-1), weights=np.repeat(volumes / 4.0, 4), minlength=len(xyz)
    )


def _relative_vector(candidate: dict, reference: dict) -> float:
    if not np.allclose(candidate["coordinates"], reference["coordinates"], atol=1e-12):
        raise RuntimeError("Compared fields use different vertex coordinates")
    weights = _vertex_weights(reference)
    delta = candidate["velocity"] - reference["velocity"]
    numerator = np.sum(weights * np.sum(delta * delta, axis=1))
    denominator = np.sum(weights * np.sum(reference["velocity"] ** 2, axis=1))
    return float(np.sqrt(numerator / denominator))


def _phase_metrics(field: dict[str, np.ndarray]) -> dict[str, float]:
    phi = np.asarray(field["phi"], dtype=float)
    weights = _vertex_weights(field)
    total = float(np.sum(weights))
    violation = np.maximum(-1.0 - phi, 0.0) + np.maximum(phi - 1.0, 0.0)
    return {
        "phi_min": float(np.min(phi)),
        "phi_max": float(np.max(phi)),
        "significant_undershoot_volume_fraction": float(
            np.sum(weights[phi < -1.001]) / total
        ),
        "bound_violation_l1_volume_mean": float(np.sum(weights * violation) / total),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    operator_rows: list[dict] = []
    for geometry, case in OPERATOR_CASES.items():
        reports = {
            form: _report(case[form], case["report"])
            for form in ("gradient", "symmetric")
        }
        fem_fields = {
            form: _field(case[form], "FEM-frozen")
            for form in ("gradient", "symmetric")
        }
        direct_change = _relative_vector(fem_fields["symmetric"], fem_fields["gradient"])
        gradient_fem_flux = float(_row(reports["gradient"], "FEM-frozen")["outlet_flux"])
        for form in ("gradient", "symmetric"):
            fem = _row(reports[form], "FEM-frozen")
            affine = _row(reports[form], "Affine-frozen")
            operator_rows.append({
                "geometry": geometry,
                "viscous_form": form,
                "fem_outlet_flux": float(fem["outlet_flux"]),
                "fem_flux_change_vs_gradient": (
                    float(fem["outlet_flux"]) / gradient_fem_flux - 1.0
                ),
                "fem_velocity_change_vs_gradient": 0.0 if form == "gradient" else direct_change,
                "affine_outlet_flux": float(affine["outlet_flux"]),
                "affine_flux_error_vs_fem": abs(
                    float(affine["outlet_flux"]) / float(fem["outlet_flux"]) - 1.0
                ),
                "affine_velocity_l2_error_vs_fem": float(
                    affine["velocity_l2_error_vs_fem"]
                ),
            })

    profile_rows: list[dict] = []
    for geometry, case in PROFILE_CASES.items():
        for profile in ("discontinuous", "exponential"):
            report = _report(case[profile], case["report"])
            row = _row(report, "FEM-frozen")
            metrics = _phase_metrics(_field(case[profile], "FEM-frozen"))
            initial = report.get("initial_phase_diagnostics", {})
            profile_rows.append({
                "geometry": geometry,
                "initial_profile": profile,
                "transition_length": 0.0 if profile == "discontinuous" else 0.05,
                "initial_max_cell_vertex_jump": float(
                    initial.get("phi_max_cell_vertex_jump", 2.0)
                ),
                "final_phi_min": metrics["phi_min"],
                "final_phi_max": metrics["phi_max"],
                "significant_undershoot_volume_fraction": metrics[
                    "significant_undershoot_volume_fraction"
                ],
                "bound_violation_l1_volume_mean": metrics[
                    "bound_violation_l1_volume_mean"
                ],
                "outlet_flux": float(row["outlet_flux"]),
                "mass": float(row["mass"]),
                "free_energy": float(row["free_energy"]),
            })

    _write_csv(REPORT_DIR / "viscous_form_summary.csv", operator_rows)
    _write_csv(REPORT_DIR / "initial_profile_summary.csv", profile_rows)

    by_operator = {(r["geometry"], r["viscous_form"]): r for r in operator_rows}
    by_profile = {(r["geometry"], r["initial_profile"]): r for r in profile_rows}
    lines = [
        "# 非 Korteweg 方法学检查（2026-08-22）",
        "",
        "所有检查保持当前黏度耦合 Stokes--CH 模型，不加入 Korteweg 力。",
        "",
        "## 黏性算子",
        "",
        "对照采用 frozen 初始黏度；每个算子内部的 FEM 与 Affine 使用完全相同弱式。",
        "Random-27 的 gradient 行取已验证生产基线，Bentheimer 的 gradient 行取 POD",
        "tol=1e-8 基线；frozen 流场不依赖后续 CH 时间步长。",
        "",
        "| geometry | form | FEM flux | FEM velocity change | Affine velocity error | Affine flux error |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for geometry in ("random27", "bentheimer"):
        for form in ("gradient", "symmetric"):
            r = by_operator[(geometry, form)]
            lines.append(
                f"| {geometry} | {form} | {r['fem_outlet_flux']:.8g} | "
                f"{r['fem_velocity_change_vs_gradient']:.3%} | "
                f"{r['affine_velocity_l2_error_vs_fem']:.3%} | "
                f"{r['affine_flux_error_vs_fem']:.3%} |"
            )
    lines += [
        "",
        "`2D(u)` 路径在两个几何的 FEM 与 Affine 上均成功装配和收敛。Random-27",
        "的 Affine 速度误差由约 5.55% 增至 9.49%，说明该几何对自然牵引下的算子",
        "选择较敏感；Bentheimer 则由约 18.74% 降至 17.61%，结论基本不变。旧基线",
        "仍以 gradient 为默认，论文必须准确写出所用弱式；是否改用 symmetric 应由目标",
        "连续模型决定，而不是按误差较小者事后选择。",
        "",
        "## 入口相容光滑初值",
        "",
        "对照均为 FEM-frozen、dt=0.25、t=1。光滑初值为",
        "`phi(x)=-1+2 exp(-x/0.05)`，在入口严格等于 +1。显著下冲定义为",
        "`phi<-1.001`，体积分数采用 P1 lumped-volume 权重。",
        "",
        "| geometry | profile | final phi_min | significant volume | L1 violation | flux |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for geometry in ("random27", "bentheimer"):
        for profile in ("discontinuous", "exponential"):
            r = by_profile[(geometry, profile)]
            lines.append(
                f"| {geometry} | {profile} | {r['final_phi_min']:.7f} | "
                f"{r['significant_undershoot_volume_fraction']:.3%} | "
                f"{r['bound_violation_l1_volume_mean']:.3e} | {r['outlet_flux']:.8g} |"
            )
    lines += [
        "",
        "光滑初值几乎消除了两个几何的下冲，强烈支持“入口网格尺度跳变是主要诱因”",
        "的判断。但它同时改变初始相含量、孔平均黏度和 frozen 流量，因此不能把新结果",
        "与旧生产基线混表。正式论文应先给出物理上希望表达的初始润湿/注入历史，再固定",
        "初值；若采用光滑初值，应报告 0.05 并补做过渡长度敏感性。",
        "",
        "机器可读数据见 `viscous_form_summary.csv` 与 `initial_profile_summary.csv`。",
    ]
    (REPORT_DIR / "METHODOLOGY_CHECKS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

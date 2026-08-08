# 20260727 说明

本目录已经从 `D:\hu\tongjiproj\FENICSX` 同步为最新版本，并清理了缓存、历史归档、smoke 输出和重复展示目录。

## 推荐展示顺序

```text
1. 真实 3D Berea Stokes 验证
2. 单相 Stokes-Tracer 验证
3. 两相流 P2-P1 HODDPNM 时间步原型
4. Adaptive audit 与后续改进方向
```

## 1. 真实 3D Berea

文件夹：

`fenicsx_real_porous_hoddpnm_berea_comparison`

优先打开：

`fenicsx_real_porous_hoddpnm_berea_comparison/outputs/03_figures_for_presentation`



- 图像风格是真实孔隙几何加 log-error 等值面。
- correctness exact-Schur case 可作为主正确性结果：速度相对误差约 `1.29e-9`，压力相对误差约 `1.38e-14`。
- balanced pressure-anchored case 可作为效率探索：FEM 约 `8.17 s`，HODDPNM 约 `5.25 s`，约 `1.56x` 加速。



- correctness exact-Schur case 不加速，HODDPNM 约 `24.72 s`，FEM 约 `8.12 s`。
- balanced case 虽然更快，但 GMRES 约 `237` 次，说明 Schur 预条件器还需要改进。
- physical diagnostics 只能说是 FEM/HODDPNM solver-comparison proxy。



## 2. 单相 Stokes-Tracer

文件夹：

`stokes_tracer_hoddpnm`

优先打开：

`stokes_tracer_hoddpnm/outputs/singlephase_tracer_hoddpnm_interface_pressure_t60_supg`



- HODDPNM Stokes active DOFs 从 `42532` 降到 `11521`，active ratio 约 `27.1%`。
- FEM Stokes solve 约 `9.627 s`，HODDPNM Stokes solve 约 `3.183 s`。
- Stokes 速度误差约 `7.39e-13`，压力误差约 `1.41e-15`。
- tracer breakthrough 相对 L2 误差约 `2.72e-13`，最终浓度场相对 L2 误差约 `4.99e-13`。
- `final_concentration_and_error.png` 现在不是点阵图，已经是连续面/体渲染风格。



- `breakthrough_curves.png`
- `tracer_error_summary.png`
- `mass_balance_validation.png`
- `final_concentration_and_error.png`



- Schur 迭代仍约 `251` 次，说明预条件器还不是最终版本。
- tracer 的优势主要来自 Stokes 求解阶段，transport 本身耗时差别不大。



- 已排除 legacy 和 experiments 归档输出。
- `outputs` 内只保留正式主 run。

## 3. 两相流 P2-P1 HODDPNM Timesteps

文件夹：

`fenicsx_twophase_p2p1_hoddpnm_stokes_timesteps`

优先打开：

`fenicsx_twophase_p2p1_hoddpnm_stokes_timesteps/outputs_presentation/05_selected_figures`

展示重点：

- 已经形成 `Sw -> mu_eff(Sw) -> FEniCSx P2-P1 Stokes -> HODDPNM Schur -> velocity -> Sw update` 的时间步循环。
- 主 case 是 27 holes、`cells_per_axis=4`、10 steps。
- active Schur DOFs 为 `31`，约占 mixed DOFs `2288` 的 `1.35%`。
- 速度误差约 `6.92e-9`，压力 mean-aligned 误差约 `5.36e-9`。
- 质量残差约 `1e-16`，CFL 约 `0.1106`。
- full Stokes solve 约 `0.0278 s`，HODDPNM Schur solve 约 `0.0031 s`。

推荐展示图：

- `03_main_twophase_step_0000.png`
- `04_main_twophase_step_0005.png`
- `05_main_twophase_step_0010.png`
- `06_main_mass_conservation_history.png`
- `07_main_hoddpnm_vs_full_error_history.png`
- `08_main_cfl_history.png`

谨慎表述：

- 当前 saturation transport 是 graph-edge Corey transport，不是最终严格守恒的 cell-wise finite-volume face-flux 离散。
- 可以称为“两相流时间步原型”或“框架验证”，不要称为完整物理两相流验证。
- 当前 10 步还不能展示完整 breakthrough。

本展示包清理：

- 已去掉 `_archive_smoke_and_old`。
- 已去掉重复目录 `01_reduction_validation`、`02_twophase_timesteps`、`04_reference_error`。
- 当前保留正式目录：`01_stokes_reduction_validation`、`02_twophase_main_case`、`03_minimal_twophase_validation`、`04_pressure_stabilization`、`05_selected_figures`。

## 4. Adaptive Audit And Fix

文件夹：

`stokes_audit_and_adaptive_fix`

优先打开：

`stokes_audit_and_adaptive_fix/REPORT_INDEX.md`

再看：

- `STOKES_AUDIT_REPORT.md`
- `EXPERIMENT_TAXONOMY.md`
- `outputs/final_state_figures`

展示重点：

- 这部分是对 adaptive/PNM/DDPNM/HODDPNM Stokes 线的审计和修正路线。
- 它明确区分了真正求解 FEniCSx Taylor-Hood P2-P1 Stokes 的结果，以及 graph pressure、tracer、smoke demo 等不能当 Stokes 验证的结果。
- 可以作为下一步算法路线：局部静态凝聚、局部 reduced Stokes bases、可计算后验误差估计器。

谨慎表述：

- adaptive 当前不是最终成功结果。
- true-error indicator 属于 oracle 诊断，因为用到了 FEM 真误差。
- computable posterior attempt 的验证速度误差仍约 `1.95e-2`。
- restricted solve 保留所有压力 DOFs，压缩效率声明要谨慎。

## 总体结论

本展示包适合定位为“文章雏形/阶段性算法验证”：

- 真实 3D：证明 HODDPNM/Schur 在真实孔隙几何上的 solver equivalence。
- tracer：证明 HODDPNM Stokes 速度场可支撑单相输运，breakthrough 和浓度误差都很好。
- 两相流：证明框架已经进入时间步耦合原型。
- adaptive：证明已经识别并修正了前期 Stokes 验证口径，下一步路线明确。

最不能说过头的地方：

- 不能说真实 3D correctness case 已经比 FEM 快。
- 不能说两相流已经是严格物理守恒 FV 验证。
- 不能说 adaptive 已经是最终算法。
- 不能说内存优势已经被完整证明；目前主要证据是自由度压缩和部分线性求解时间优势。


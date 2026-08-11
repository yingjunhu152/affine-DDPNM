# Affine-DDPNM 驱动的两相 Buckley--Leverett 验证

本项目在随机 27 球三维多孔介质上比较四个固定总速度场：完整 Taylor--Hood
FEM，以及 Classic-1、W1n-3、Affine-9 三档 DDPNM。每个速度场驱动同一个
非线性水饱和度输运模型，所有 DDPNM 输运误差均以 FEM 速度驱动结果为参考。

这里采用单向耦合：总速度由定常 Stokes 问题给出，饱和度不反向改变压力或总
速度。因此它是“固定速度驱动的两相输运验证”，不是全耦合两相流求解器。

## 模型与离散

方程为

\[
\phi\,\partial_t S_w + \nabla\!\cdot\!\left(f_w(S_w)\mathbf u_t\right)
-\kappa\Delta S_w=0.
\]

Corey 分流量采用

\[
S_e=\frac{S_w-S_{wr}}{1-S_{wr}-S_{or}},\qquad
f_w=\frac{S_e^{n_w}/\mu_w}{S_e^{n_w}/\mu_w+(1-S_e)^{n_o}/\mu_o}.
\]

默认参数为 `Swr=Sor=0.2`、`nw=no=2`、`mu_w=1`、`mu_o=5`。默认入口值为
`1-Sor=0.8`，饱和度限幅区间为 `[Swr, 1-Sor]`。

输运离散包括：

- P1 有限元与隐式欧拉；
- 分流量 Picard 迭代；
- 以 `fw'(Sw) u` 为特征速度的残差型 SUPG；
- 入口和出口的迎风数值通量；
- 球面和侧壁零通量；
- 保持一致质量矩阵积分的有界守恒限幅器。

质量验证独立比较水库存与边界通量账本：

\[
M_w^{n+1}-M_w^n+\Delta t\left(Q_{w,\mathrm{out}}-Q_{w,\mathrm{in}}\right)=0.
\]

## 运行

需要 FEniCSx、Gmsh、MPI、NumPy、SciPy、Matplotlib 和 PyVista。Windows
conda 环境示例：

```powershell
cd D:\hu\tongjiproj\20260727\20260824\affine_ddpnm_twophase
conda run -n fenicsx --no-capture-output python -u run_affine_ddpnm_twophase.py
```

默认启用 SUPG；可用 `--no-supg` 关闭。默认 `t_final=30` 可能早于含水率突破，
此时 `t10/t50/t90` 为 `NaN` 是合理结果，应同时检查报告中的最终注入孔隙体积
`final PVI`。如需突破曲线，应增加 `--t-final`。

不依赖 FEniCSx 的核心数值测试：

```powershell
python -m unittest discover -s tests -v
```

单步 FEniCSx 守恒诊断：

```powershell
conda run -n fenicsx --no-capture-output python -u debug_twophase_step.py
```

## 输出

完整运行写入 `outputs/benchmark_twophase/`：

- `watercut_curves.png`：出口含水率；
- `recovery_curves.png`：采收率；
- `mass_balance_validation.png`：水库存、边界通量账本及累计残差；
- `twophase_error_summary.png`：三类 DDPNM 输运误差；
- `final_saturation_and_error.png`：终态饱和度及 Affine 误差；
- `twophase_metrics.csv`、`twophase_history.csv`：汇总和逐步账本；
- `affine_ddpnm_twophase_report.json`、`TWOPHASE_VALIDATION_REPORT.md`；
- 四个 VTU、`twophase_fields.npz` 和网格文件。

## 修复说明

本版本修复了原型中的分流量导数反号、遗漏出口边界通量、SUPG 系数与右端符号、
初始采收率偏置、非物理饱和度区间、伪质量残差、节点求和式含水率、`tp.tt`
属性错误以及诊断脚本缺失依赖。旧 smoke 和失败 benchmark 已移入
`outputs/legacy_invalid_20260811/`，仅供追溯，不能作为验证结果。

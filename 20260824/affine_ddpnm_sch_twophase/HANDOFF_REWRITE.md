# 重写版交接（2026-08-22）

## 状态

随机 27 球的 FEM / Classic-DDPNM / Affine-DDPNM × frozen / SFI 六个算例已经
全部实现、烟测并完成 `dt=1, t_final=6` 正式运行。机器可读报告通过完整性校验：
6 行、每行 6 步、全部 `converged=true`、所有流动线性残差有限。

正式结果目录：

`outputs/random27_six_rewrite_20260822`

相反转 Bentheimer 也已接入并完成六臂单步初步基准（`dt=1, t_final=1`）：

`outputs/bentheimer_six_preliminary_20260822`

该网格含 48,078 个四面体、27 个 Cartesian 分区孔和 83 个连通界面片。六臂
均收敛，FEM/Classic/Affine 的 frozen 速度相对误差分别为 0、0.7871、0.1874。

在此基础上，非 Korteweg 正式 campaign 已完整结束：两个几何 × 两档生产时间步
(`dt=1,0.5`) × 六臂，共 24 行生产结果，全部推进到 `t=6` 并收敛。另有 6 个
时间步诊断点和 3 个 POD 阈值点。统一目录为：

`outputs/baseline_campaign_20260822`

机器生成的总报告为 `CAMPAIGN_REPORT.md`，汇总表为 `production_summary.csv`、
`time_step_summary.csv`、`pod_summary.csv`，图位于 `figures/`。

## 关键设计决定

旧版失败的根因是把局部 P1 化学势 trace 作为强 Dirichlet 端口数据，而多个端口
共享同一 P1 顶点自由度，造成不相容约束；组装出的 CH Schur 也奇异或不定。重写
版没有修补该系统，而是彻底移除它：所有臂共享一个全局 conforming CH solver，
三臂只替换速度来源。

这使比较口径变成：

- FEM：全局 P2--P1 Stokes；
- Classic：每界面 1 模态局部响应；
- Affine：每界面 9 模态局部响应；
- 三者：相同 P1--P1 CH 时间推进、入口相场条件与 Newton 容差。

SFI 使用孔内平均相场闭合孔内常黏度。第一轮为未阻尼预测器，后续修正
`omega=0.65`；这个改动把 FEM-SFI 单步从 10 次无意义的几何尾部迭代降为 2--4
次，同时保持最终解不变。

## 已验证事项

- Python compileall：通过。
- FEM-frozen 单步：通过，流动残差约 `3.7e-7`。
- Classic-frozen/SFI：通过，Schur 残差约 `1e-15`。
- Affine-frozen/SFI：通过，Schur 残差约 `1e-15`。
- FEM-SFI 持久化弱形式：通过，不会随 Picard 轮次重复 FFCx 编译。
- 六臂长跑：全部通过。
- 报告 JSON：读取、行数、步数、收敛标志、有限残差校验通过。
- Bentheimer 六臂单步：全部通过；机器可读结果齐全。
- Bentheimer Affine：原始 747 维 Schur 病态；缩放 POD 保留 720 维后，投影
  残差 `8.6e-16`，被截断方向残差约 `4.0e-15`。
- 正式 campaign：24/24 生产臂收敛；所有 JSON、history 和带网格元数据的 NPZ
  完整，最终场无 NaN/Inf。
- 时间精化：`dt=1` 对 `dt=0.5` 最终相场全场差约 `0.36%--0.45%`，SFI 速度
  全场差约 `1e-4`，Classic/Affine 精度排序稳定。
- POD 阈值：`1e-6,1e-8,1e-10` 全部保留 `720/747`，误差与流量不变。

## 不应忽略的限制

1. 当前 Stokes 方程没有 Korteweg 毛细体力；SFI 双向反馈目前只有相场到黏度。
2. 强入口跳变下 P1 相场约有 `phi_min=-1.35` 的下过冲。
3. 表内时间不包含 Classic/Affine 一次性响应库离线构建时间。
4. DDPNM 响应重缩放只以速度和通量为目标；混合向量中的局部压力值没有用于输出。
5. 时间步、POD 和 `t=6` 长时间检查已经完成；尚未做空间网格精化。
6. 相场过冲在时间步细化后不消失，Bentheimer 长跑中最坏约 `phi_min=-1.548`；
   这是当前连续 P1 Galerkin 对流和强入口跳变的已知空间非单调性，未做 clipping。

在加入 Korteweg 力前，这批结果应称为“黏度耦合 Stokes--CH 基线”，不是完整
Model H 最终结果。

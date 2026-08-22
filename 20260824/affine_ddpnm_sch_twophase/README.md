# Stokes--Cahn--Hilliard 六臂实验

这是 2026-08-22 从零重写的实现。旧版源码不再使用；旧数学文档、伪代码和
`HANDOFF_SCH.md` 仅作为历史记录保留。

## 六个算例

在随机 27 球或相反转 Bentheimer 三维多孔介质上运行：

1. FEM-frozen
2. FEM-SFI
3. Classic-DDPNM-frozen
4. Classic-DDPNM-SFI
5. Affine-DDPNM-frozen
6. Affine-DDPNM-SFI

三种方法使用同一个全局 conforming P1--P1 Cahn--Hilliard 输运离散。Classic
使用每界面 1 个法向常数模态，Affine 使用每界面 9 个仿射向量模态，FEM 使用
全局 Taylor--Hood P2--P1 Stokes。这样误差比较只反映流动降阶空间，不再混入
旧版病态的局部 CH trace Schur 系统。

SFI 的第一轮是未阻尼时间步预测器，后续耦合修正使用 `omega=0.65`。反馈量是
孔内体积平均相场对应的孔内常黏度。

## 运行

```powershell
cd D:\hu\tongjiproj\20260727\20260824\affine_ddpnm_sch_twophase
conda run -n fenicsx --no-capture-output python -u run_random27_six.py `
  --arms all --dt 1 --t-final 6 `
  --out-dir outputs\random27_six_rewrite_20260822
```

程序拒绝写入非空输出目录，避免覆盖历史结果。可用 `--arms FEM-frozen` 等名称
单独运行一个或多个臂。

Bentheimer 入口使用现成的反相数字岩心四面体网格：

```powershell
conda run -n fenicsx --no-capture-output python -u run_bentheimer_six.py `
  --arms all --dt 1 --t-final 1 `
  --out-dir outputs\bentheimer_six_preliminary_20260822
```

## 已完成结果

正式运行参数为 `dt=1, t_final=6`。网格为 28,214 个四面体、27 个孔、114 个
内部界面。六臂全部收敛。

| 算例 | phi 相对 L2 误差 | 速度相对 L2 误差 | 出口流量 | 在线时间/s |
|---|---:|---:|---:|---:|
| FEM-frozen | 0 | 0 | 0.00150622 | 23.77 |
| FEM-SFI | 0 | 0 | 0.00152249 | 254.71 |
| Classic-frozen | 0.037002 | 0.587548 | 0.00263670 | 11.56 |
| Classic-SFI | 0.037712 | 0.594800 | 0.00267678 | 34.62 |
| Affine-frozen | 0.002053 | 0.055534 | 0.00154863 | 11.86 |
| Affine-SFI | 0.002113 | 0.055862 | 0.00156576 | 32.90 |

结果文件：`outputs/random27_six_rewrite_20260822/random27_six_arms.csv` 和
`random27_six_arms.json`；每个臂另有 history CSV 与最终 `phi/velocity` NPZ。

### 相反转 Bentheimer 初步结果

网格为 48,078 个四面体、10,437 个点，Cartesian 3×3×3 分区得到 27 个孔和
83 个连通界面片。当前只跑了 `dt=1, t_final=1` 的六臂短时基准，不能代替时间步
收敛和长时间实验。

| 算例 | phi 相对 L2 误差 | 速度相对 L2 误差 | 出口流量 | 在线时间/s |
|---|---:|---:|---:|---:|
| FEM-frozen | 0 | 0 | 0.000261865 | 32.71 |
| FEM-SFI | 0 | 0 | 0.000262527 | 62.70 |
| Classic-frozen | 0.001637 | 0.787149 | 0.000578358 | 5.22 |
| Classic-SFI | 0.001642 | 0.787405 | 0.000579924 | 10.81 |
| Affine-frozen | 0.000281 | 0.187416 | 0.000291088 | 5.61 |
| Affine-SFI | 0.000282 | 0.187452 | 0.000291841 | 8.49 |

Bentheimer 的 9 模态 Affine 全空间包含近零相关方向，原始 747 维系统不适合直接
求解。程序以对角缩放 POD 删除 27 个数值零方向，保留 720 维；投影残差约
`8.6e-16`，被删除方向残差约 `4.0e-15`。结果位于
`outputs/bentheimer_six_preliminary_20260822`。表内在线时间与 Random-27 一样不含
一次性局部响应库构建。

## 当前物理边界

当前重写版已经包含 CH 表面能、化学势扩散、对流和相场到黏度的 SFI 反馈；
尚未把 Korteweg 毛细体力 `-phi grad(mu_ch)` 加入 Stokes 方程。因此这批结果是
稳定的“黏度耦合 Stokes--CH 基线”，不能冒充完整双向 Model H 毛细耦合结果。
相场在强入口跳变附近存在有限元过冲，本次最大约 `phi_min=-1.35`，结果文件中
完整保留了极值。后续若需要完整 Model H，应在现有清晰接口上新增毛细载荷，
而不是恢复旧版 CH trace Schur 实现。

## 非 Korteweg 正式 campaign（2026-08-22）

按当前“变黏度 Stokes--CH 基线”完成了以下验证，明确暂不加入 Korteweg 力：

- 两个几何的 `dt=1, 0.5, 0.25, t_final=1` FEM-frozen 时间步诊断；
- Bentheimer Affine 的 POD 阈值 `1e-6, 1e-8, 1e-10` 敏感性；
- 两个几何的 `dt=1, 0.5, t_final=6` 正式六臂运行，共 24 行生产结果；
- 全场时间精化误差、离线/在线成本、历史曲线和中央切片图。

全部运行收敛且最终 NPZ 场为有限值。`dt=1` 到 `dt=0.5` 的最终相场全场差为
约 `0.36%--0.45%`，SFI 速度差约 `1e-4`。相场下过冲随时间步细化没有消失：
随机球在 `t=1` 的 `phi_min` 为 `-1.346, -1.413, -1.453`；Bentheimer 为
`-1.328, -1.371, -1.397`。这指向强入口跳变下连续 P1 Galerkin 对流的空间
非单调性；程序没有用 clipping 隐藏它。

POD 三档阈值均保留 `720/747` 个 Bentheimer Affine Schur 方向，流量、速度误差
和残差逐位一致，默认 `1e-8` 位于稳定平台。完整报告见
`outputs/baseline_campaign_20260822/CAMPAIGN_REPORT.md`。

可断点续跑：

```powershell
conda run -n fenicsx --no-capture-output python -u run_baseline_campaign.py --stage all
conda run -n fenicsx --no-capture-output python analyze_baseline_campaign.py
```

## 新代码结构

- `run_random27_six.py`：随机 27 球命令行入口。
- `run_bentheimer_six.py`：相反转 Bentheimer 命令行入口。
- `run_baseline_campaign.py`：时间步、POD 和正式生产批处理，支持断点续跑。
- `analyze_baseline_campaign.py`：完整性校验、跨时间步场误差、CSV/JSON/图表和报告。
- `schbench/config.py`：六臂、物理和数值参数。
- `schbench/geometry.py`：随机 27 球加载、孔体积与场投影。
- `schbench/flow.py`：持久化 FEM、Classic、Affine 流动 solver。
- `schbench/transport.py`：全局凸分裂 CH Newton 与线搜索。
- `schbench/experiment.py`：frozen/SFI 编排、输出与误差参照。
- `schbench/metrics.py`：网格加权 L2 误差。

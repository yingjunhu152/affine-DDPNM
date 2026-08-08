# 三维 adaptive DDPNM 结果

## 1. 本次加入的界面空间

在原始 3D DDPNM 的同一张 34,341 四面体网格上，构造严格嵌套的三层界面空间：

| 层级 | 每条界面的牵引表示 | 每界面自由度 |
|---|---|---:|
| DDPNM | 常数法向牵引 | 1 |
| DDPNMT | 常数法向牵引 + 两个常数切向牵引 | 3 |
| HODDPNM | 法向和两个切向分量均乘以面内完整 P1 基 `{1,s,t}` | 9 |

局部 `[P2]^3-P1` Stokes 内部自由度首先被消元，随后只在界面牵引系数上装配 Schur 系统。这里的 HODDPNM 是面内仿射 P1 版本；它不是把每个界面网格节点全部作为独立未知量的完整 FE-trace Schur 系统。

## 2. adaptive 逻辑

目标层级容忍值取 `TOL=1%`，Dörfler 参数取 `theta=0.65`。整体 FEM 解不参加标记，只用于事后验证。

1. 所有 144 条界面从 DDPNM 开始。
2. 以完整 DDPNMT 为第一层嵌入比较解。经过 6 次升级后，73 条界面仍为 DDPNM、71 条为 DDPNMT，当前层级差为 0.545%，满足 1% 目标。
3. 以完整 HODDPNM 为第二层嵌入比较解，继续逐界面升级。
4. 最终有 142 条界面达到 HODDPNM，2 条保留为 DDPNMT；界面系统含 1,284 个未知量，完整 HODDPNM 为 1,296 个。
5. 最终解相对完整 HODDPNM 的层级差为：速度 0.212%，压力 0.052%。

因此，在当前规则三维几何和 1% 目标下，自适应最终几乎退化为全 HODDPNM。这不是程序失效，而是层级估计器表明绝大多数三维孔喉界面都需要面内线性模式。

## 3. 与同网格传统 FEM 的严格误差

误差用同一张网格上的全局 Taylor--Hood `[P2]^3-P1` 解计算，局部 DD-PNM 场保持分片，不在界面两侧做平均；体积分使用六阶求积。

| 方法 | 界面未知量 | 速度相对 L2 | 速度 broken-H1 | 压力原始相对 L2 | 出口流量误差 |
|---|---:|---:|---:|---:|---:|
| DDPNM | 144 | 21.86% | 45.31% | 2.88% | 18.72% |
| DDPNMT | 432 | 20.83% | 44.01% | 2.90% | 17.40% |
| HODDPNM | 1,296 | 6.72% | 20.14% | 1.43% | 2.37% |
| Adaptive | 1,284 | 6.73% | 20.15% | 1.43% | 2.37% |

与原始 DDPNM 相比，adaptive 将速度 L2 误差降低约 69%，将 broken-H1 误差降低约 56%，并把流量误差降低约 87%。常数双切向模式 DDPNMT 只带来小幅改善；真正显著的改进来自界面上的面内线性变化。

## 4. 如何理解剩余误差

- 速度 L2 已从 21.86% 降至 6.73%，说明界面 P1 富集确实抓住了主要模型误差。
- broken-H1 仍有 20.15%，并且误差图中高值集中在球体上下游与孔喉交汇处。这些位置的速度迹和梯度沿界面并非简单仿射变化。
- 压力原始 L2 为 1.43%，流量误差为 2.37%，说明平均压降和总通量已经明显可靠。
- 正式网格比粗网格更细，但 HODDPNM 速度误差仍在数个百分点，表明剩余误差主要来自界面空间阶数，而不是局部体网格分辨率。

若下一步要继续压低 broken-H1，应优先将 HODDPNM 从面内 P1 升到 P2，或使用完整界面节点/谱基的 FE-trace Schur 空间；仅继续细化体网格预计收益有限。

## 5. 运行与输出

```powershell
cd D:\hu\tongjiproj\20260727\ddpnm_3d_uniform_spheres
.\run_adaptive.ps1 --out-dir outputs\adaptive_hierarchy
```

主要输出位于 `outputs/adaptive_hierarchy`：

- `adaptive_report.json`：全部参数、守恒残差、误差和计时；
- `adaptive_history.csv`：每轮标记与界面层级；
- `method_error_metrics.csv`：四种方法的 FEM 误差表；
- `01_adaptive_convergence.png`：层级误差、界面数量和界面未知量；
- `02_adaptive_final_interface_hierarchy.png`：最终界面层级；
- `03_method_errors_to_fem.png`：方法误差比较；
- `04_adaptive_fem_error_fields.png`：FEM、adaptive 和误差四联图；
- `05_adaptive_algorithm_box.png`：论文风格算法框；
- `ADAPTIVE_3D_ALGORITHM.md`：算法文字版。

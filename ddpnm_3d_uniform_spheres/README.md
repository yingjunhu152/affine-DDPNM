# 3D DDPNM：27 球规则多孔介质

计算域为单位立方体 `[0,1]^3` 减去均匀排列的 `3×3×3=27` 个固体球。球心坐标取 `{0.20,0.50,0.80}^3`，半径为 `0.105`；入口为 `x=0`、压力 1，出口为 `x=1`、压力 0，其余立方体外表面和全部球面为无滑移固壁。

工程同时包含：

- 原始 DDPNM：每条内部界面 1 个常数法向牵引自由度；
- DDPNMT：增加两个常数切向牵引自由度；
- HODDPNM：三个牵引分量均使用界面局部坐标中的完整 P1 基 `{1,s,t}`；
- 两阶段 adaptive：`DDPNM → DDPNMT → HODDPNM`；
- 同一张四面体网格上的传统 Taylor--Hood `[P2]^3-P1` FEM 参考解与严格误差分析。

## 几何分区

净距函数同时考虑球面和立方体六个外边界：

```text
d(x)=min{x,1-x,y,1-y,z,1-z,min_j(||x-c_j||-r)}.
```

球心层平面把每个坐标方向分成四段，因此流体域中得到 `4×4×4=64` 个最大空球和 64 个子区域。连接六邻接最大球得到 144 条图边，即 144 个孔喉界面。

对当前等半径规则阵列，严格净距鞍点分隔面是九个解析平面：

```text
x=0.2,0.5,0.8;  y=0.2,0.5,0.8;  z=0.2,0.5,0.8.
```

这些平面嵌入 OpenCASCADE 流体体并统一 fragment，因此 64 个局部子网格来自同一张全局一致网格，接口三角形和节点严格匹配。该解析结论只适用于当前对称几何；真实随机三维介质需要数值最大球检测、净距 watershed 和一般曲面接口。

## 网格

Gmsh 使用三套 `Distance → Threshold` 尺寸场，并用 `Min` 合并：

1. 球面附近加密；
2. 立方体外边界附近加密；
3. 144 个孔喉界面附近加密。

正式网格含 34,341 个四面体、8,654 个顶点和 7,956 个接口三角形。传统 FEM 与所有 DD-PNM 层级共享这一个 `partition.mesh`，不存在网格转换误差。

## 方法层级

| 方法 | 每界面自由度 | 界面牵引空间 |
|---|---:|---|
| DDPNM | 1 | `P0` 法向 |
| DDPNMT | 3 | `P0` 法向 + 两个 `P0` 切向 |
| HODDPNM | 9 | 三分量 × `{1,s,t}` 完整面内 P1 |

每个局部 Stokes 矩阵只分解一次。局部有限元内部自由度被静态凝聚，最终只求解界面牵引系数构成的 Schur 系统。

adaptive 使用两层嵌入比较：先比较 DDPNM 与完整 DDPNMT，再比较当前混合阶解与完整 HODDPNM；当归一化速度或压力层级差超过目标容忍值时，按 Dörfler 准则标记界面，每条标记界面只升一级。传统 FEM 不参加标记，只用于事后验证。

## 运行

原始 DDPNM：

```powershell
cd D:\hu\tongjiproj\20260727\ddpnm_3d_uniform_spheres
.\run.ps1 --with-reference --out-dir outputs\fem_comparison
```

三维 adaptive：

```powershell
.\run_adaptive.ps1 --out-dir outputs\adaptive_hierarchy
```

默认 adaptive 参数为 `TOL=0.01`、Dörfler `theta=0.65`，每轮最多标记 12 条界面。FEniCSx 的 JIT 缓存位于工程内 `.cache`。

## 正式 adaptive 结果

1% 层级容忍值下，最终 142 条界面为 HODDPNM、2 条为 DDPNMT，界面未知量为 1,284；相对完整 HODDPNM 的层级差为 0.212%。

| 方法 | 速度相对 L2 | 速度 broken-H1 | 压力原始相对 L2 | 流量误差 |
|---|---:|---:|---:|---:|
| DDPNM | 21.86% | 45.31% | 2.88% | 18.72% |
| DDPNMT | 20.83% | 44.01% | 2.90% | 17.40% |
| HODDPNM | 6.72% | 20.14% | 1.43% | 2.37% |
| Adaptive | 6.73% | 20.15% | 1.43% | 2.37% |

详细解释见 `ADAPTIVE_3D_RESULTS.md`，算法框和全部数据位于 `outputs/adaptive_hierarchy`。

## 主要文件

- `ddpnm3d/geometry.py`：最大球、孔喉图、严格鞍点分区和加密网格；
- `ddpnm3d/solver.py`：原始 DDPNM 与传统 FEM；
- `ddpnm3d/hierarchy.py`：DDPNMT、HODDPNM 和 adaptive 界面 Schur 求解；
- `ddpnm3d/postprocess.py`：同网格严格误差、切片和数据输出；
- `run_ddpnm_3d.py` / `run.ps1`：原始流程；
- `run_adaptive_ddpnm_3d.py` / `run_adaptive.ps1`：adaptive 流程；
- `plot_results.py` / `plot_adaptive_results.py`：论文风格图；
- `verify_results.py` / `verify_adaptive_results.py`：自动数值审计。

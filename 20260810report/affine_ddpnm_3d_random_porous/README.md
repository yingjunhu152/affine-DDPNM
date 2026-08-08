# Affine-DDPNM 3-D on the random porous medium

本文件夹按照 `ddpnm_2d_random_porous` 的随机计算域生成逻辑，为
`affine_ddpnm_3d` 构建三维随机多孔介质算例：

- **随机实现**：固定种子 `20260804` 一次生成的 27 个随机球（18 个边界球
  球心刻意伸出单位立方体、被采样窗口裁剪；9 个内部球），坐标冻结在
  `random_porous.py::SPHERES`，每次运行使用同一张几何。半径范围与最小间隙
  满足 `r_max - r_min <= gap/2`，保证每个 Voronoi 孔喉面不穿过任何球。
- **孔喉界面**：与 2-D 相同，Delaunay 邻接对定义候选孔喉；对每个有效对
  构造两球壁之间的中心线间隙段，其等净距点为解析鞍点；间隙段穿过第三球
  的对被拒绝。2-D 的间隙段是 1 维线段，不能分割三维体积，因此 3-D 的界面
  是"过鞍点、垂直于连心线的平面中属于该通道的部分"，即两个球心的
  Voronoi 面（由相邻鞍点平面和立方体壁截断）——这是同一构造的
  3-D 推广（均匀算例的 9 张网格平面是其规则特例）。
- **共形网格**：孔喉面片嵌入 OpenCASCADE 流体体后统一 fragment，得到
  27 个子区域、一张全局一致的四面体网格，不存在网格转换误差。

## 方法对比（与 `affine_ddpnm_3d` 相同口径）

同一张网格上比较：

| 方法 | 每界面模态 |
|---|---:|
| Classic-DDPNM | 常数法向 `1*n`（1 模态） |
| Affine-DDPNM | 九模态 `{1,s,t} x {n,t1,t2}`（9 模态） |
| Exact FE-trace Schur | 完整界面有限元节点（正确性基线） |
| Monolithic FEM | 整体 Taylor-Hood P2-P1（参考解） |

## 运行

```powershell
conda run -n fenicsx --no-capture-output python run_random_benchmark.py --out-dir outputs\benchmark
```

结果表格与误差云图：

```powershell
conda run -n fenicsx --no-capture-output python plot_random_errors.py
```

输出位于 `outputs/benchmark`：`random_affine_report.json`（全部误差与计时）、
`random_affine_metrics.csv`、`random_benchmark_fields.npz`（切片场数据）以及
四张图（`01_slice_error_fields.png`、`02_geometry_and_partition.png`、
`03_2d_3d_error_ratio.png`、`04_methods_vs_exact_schur.png`）。

详细数值见 `RESULTS.md`。

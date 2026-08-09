# 缺失场数据清单（2026-08-09 盘点，明天跑）

> 今天的 fig6（2×4 误差云图）数据已齐，已进 B/A/A' 三个文档。
> 以下数据缺失 → 明天补跑。**代码已全部就位**，数据一到直接生成图。

## 1. Random-27 接口导出重跑（fig7 必需）【~3-5 min】

- 命令（fenicsx env，PATH 导出 Library/bin）：
  ```
  export PATH="/d/Miniconda3/envs/fenicsx/Library/bin:$PATH"
  /d/Miniconda3/envs/fenicsx/python.exe run_random_benchmark.py --out-dir outputs/benchmark_w1n
  ```
  （目录：`20260810report/affine_ddpnm_3d_random_porous/`）
- 产出：新 `random_benchmark_fields.npz` 追加接口键（`interface_q_fem/classic/normal_linear/affine`、
  `interface_st`、`interface_triangles`、`representative_interface_id`、`fem_interface_fluxes`、
  `interface_slice_segments`）——代码已在 `run_random_benchmark.py` 和 `visualization.py` 写好。
- 用途：**fig7 接口局部图（q=u·n）完整 2×4**；fig6 random 行补灰色接口轮廓线。
- 验证：csv 误差列应与基线逐位一致；timing 列有噪声，新 csv/report 移入
  `回收站/20260809_random_benchmark_w1n_timings/`，基线 csv/report 从 `_backup_random_benchmark_w1n/` 恢复。

## 2. Real-100（Heterogeneous-100）场数据（可选第三行）【~5 min】

- 前置：修 `real_porous_benchmark_3d` 的 `z_slice` 参数不匹配 bug（旧档 §4.1，未修）。
- 然后给 run 脚本加切片导出（照 uniform 的做法）重跑 benchmark_w1n。
- 用途：fig6 第三行（或独立一张）。论文口径 benchmark_w1n 目前**没存任何场**；
  旧的 `outputs/benchmark_voronoi_100/slice_fields.npz` 是不同分区（voronoi）且缺 W1n，**不能用于论文**。

## 3. （论文既有待办，与图无关，不用明天跑）

W0v 三几何、full C-DD ξ、mesh sensitivity、Random-27 多 realization、(g) 统一计时 ——
已在 B 文档 §10.3 limitations (a)–(g) 列出。

## 代码就绪清单（今天已写好，未跑）

- `ddpnm_3d_uniform_spheres/ddpnm3d/visualization.py`：
  `evaluate_fem_ddpnm_interface`、`fem_interface_fluxes`、`interface_contour_segments`
- `affine_ddpnm_3d/run_affine_ddpnm_benchmark.py`：切片 + 接口导出（已跑通，npz 已交付）
- `affine_ddpnm_3d_random_porous/run_random_benchmark.py`：接口导出（未跑，等明天）
- `numerical_section/scripts/make_figures.py`：`fig7_interfaces`（数据齐自动生成，缺数据会打印 skip）

## 今天已交付（存档）

- `20260810report/affine_ddpnm_3d/outputs/benchmark_w1n/affine_benchmark_fields.npz`（新）
- `numerical_section/figures/fig6_fields.png`（2×4 新版）+ deepseekoutput 两副本已同步
- B/A/A' tex：fig6 章节重写（新 caption 含共用色标/坐标/切片说明）+ 附录溯源更新
- 基线 csv/report 全部还原；新 timing 版本在 `回收站/20260809_uniform_benchmark_w1n_timings/`
- 备份目录 `_backup_uniform_benchmark_w1n/`、`_backup_random_benchmark_w1n/`（验证完可清）

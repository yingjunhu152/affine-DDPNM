# 2026-08-06 工作总结

## 输出文件

| 文件 | 内容 |
|---|---|
| `run.log` | 标准 Voronoi 100球 benchmark 完整日志 |
| `benchmark_report.json` | 离线/在线拆分报告 |
| `slice_errors_fields.png` | z=0.5 2×3 误差云图（速度+压力，无网格线） |
| `slice_fields.npz` | 切片原始数据 |

## 最终结果：标准 Voronoi 100球

| | FEM | Classic-DDPNM | Affine-DDPNM |
|---|---|---|---|
| 离线基函数 | — | **81 MiB** | **496 MiB** |
| 在线组装+求解 | — | **34 MiB** | **660 MiB** |
| 总内存 | 404 MiB | **115 MiB** | **1,156 MiB** |
| uk | — | 537 | 4,833 |
| L²(u) | ref | 95.1% | 15.7% |
| H¹(u) | ref | 115.6% | 30.2% |
| L²(p) | ref | 11.2% | 2.5% |
| 流量误差 | ref | 132.2% | 8.7% |
| 加速比 | 1× | 104× | 1.7× |

### 内存拆分解读

- **Classic**（离线 81 + 在线 34 = 115 MiB）：在线极轻（537² 稀疏），整体只有 FEM 的 28%。适合做"几何感知模态筛选"——关掉需要少的界面来省离线存储。
- **Affine**（离线 496 + 在线 660 = 1,156 MiB）：在线 660 MiB 中 ~187 MB 是 `toarray()` 稠密化，~187 MB 是 `eigvalsh` 全特征值分解（均为诊断用，可去掉）。真正的稀疏求解只需要 ~200 MiB。

## 代码改动清单

| 文件 | 修改 | 原因 |
|---|---|---|
| `ddpnm_core/assembler.py` | 稠密→稀疏 Schur（COO→CSR+spsolve） | 4833² 稠密浪费 187 MB |
| `ddpnm_core/fem_utils.py` | `parent_facet_lookup` 加 id(msh) 缓存 | 100×重建 90K 条目 dict→MemoryError |
| `ddpnm_core/library.py` | per-pore `gc.collect` + `del` 临时变量 | SuperLU C 堆耗尽 |
| `benchmark_voronoi_full.py` | 离线/在线分拆 tracemalloc | — |
| `geometry.py` | `_weighted_voronoi_throat_faces`（4D 外心法）+ `build_partition_weighted_voronoi_occ` | 加权 Voronoi 面生成+OCC 集成 |
| `plot_slice_errors.py` | grid+σ=4 强平滑+`imshow`，去网格线/子区域线 | — |

## 加权 Voronoi 进展

面生成已完成（362→248 面），OCC 嵌入成功（248/248），但 fragment 超时。下一步：
1. `validate_weighted_partition.py` — 仅跑 CAD+网格，不做 FEM
2. 把 fragment 改逐个 `cut` 循环
3. 验证通过后跑 `benchmark_weighted_voronoi_full.py`

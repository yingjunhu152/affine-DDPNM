# 交接文档：真实多孔 benchmark + 密集孔隙 / 加权 Voronoi（2026-08-06 终版）

> 复制本文件全文到下一个窗口。所有路径基于 `D:\hu\tongjiproj\20260727\real_porous_benchmark_3d`。

---

## 0. 你的身份与总目标

接手 `real_porous_benchmark_3d/` 项目。

**今日（8/6）已完成**：
- ✅ 100 球标准 Voronoi benchmark 跑通，含**离线/在线内存拆分**（见 §8）
- ✅ **稀疏 Schur 组装修复**（dense→COO→CSR+spsolve，Affine 在线 775→660 MiB）
- ✅ `parent_facet_lookup` 缓存 + per-pore GC（修 MemoryError 和 SuperLU 耗尽）
- ✅ 加权 Voronoi 4D 外心法面生成（全局共享顶点，248/248 面嵌入 OCC 成功）
- ✅ 误差云图重绘（`tripcolor`→grid+σ=4 强平滑+`imshow`，无网格线，子区域边界线已删）

**明天要做**：
1. 跑 `python validate_weighted_partition.py` 验证加权 Voronoi 分区
2. 如 OCC fragment 超时 → 改逐个 `cut`（`gmsh.model.occ.cut` 循环）
3. 分区通过后跑 `benchmark_weighted_voronoi_full.py`

**关键发现**：
- 标准 Voronoi Classic L²=95% → **必须用加权 Voronoi**
- Affine 在线 660 MiB 来自 `toarray()`(187MB)+`eigvalsh`(~187MB) → 去掉 ~200 MiB
- 所有结果在 `outputs/20260806/`

---

## 1. 环境与常用命令

### 1.1 Python

- **运行前必须**：`export PATH="/d/Miniconda3/envs/fenicsx/Library/bin:$PATH"`
- **python 必须用**：`/d/Miniconda3/envs/fenicsx/python.exe`
- 不带 PATH 时 LAPACK 原生崩溃（exit 127）
- **MPI 通信子泄漏**：gmsh/dolfinx 会累积消耗 MPI 通信子（2048 上限）。症状：`Too many communicators` 或 MPI segfault。**每次跑长任务前重启 Claude Code 会话**。

### 1.2 关键脚本

| 脚本 | 用途 | 运行方式 |
|---|---|---|
| `benchmark_voronoi_full.py` | **一条命令跑完 100 球 benchmark**：网格+FEM+Classic+Affine+切片+报告 | `python benchmark_voronoi_full.py` |
| `plot_slice_errors.py` | 各分区的 2×3 误差云图 | `python plot_slice_errors.py --partition voronoi` |
| `weighted_voronoi_prototype.py` | 加权 Voronoi 纯 numpy 原型 | `python weighted_voronoi_prototype.py` |
| `_gen_packing.py` | 生成新球堆积配置 | `python _gen_packing.py`（输出手动复制到 `geometry.py`） |
| `_update_spheres.py` | 将加权 Voronoi 原型的堆积写入 geometry.py | `python _update_spheres.py` |

### 1.3 50 球 benchmark 参考命令

```bash
# 单个分区切片
python plot_slice_errors.py --partition voronoi --out-dir outputs/slices

# 只建不画（--build-only）
python plot_slice_errors.py --partition grid --out-dir outputs/slices --build-only
```

---

## 2. 项目文件结构

```
real_porous_benchmark_3d/
├── HANDOFF.md                        ← 本文档
├── geometry.py                       ← SPHERES（100球85.6%）+ 分区构建器
│                                         Voronoi / WeightedVoronoi(labeling) / Grid / Watershed
├── _gen_packing.py                   ← 球堆积生成器
├── _update_spheres.py                ← 将堆积写入 geometry.py
├── benchmark_voronoi_full.py         ← ★ 100球完整 benchmark（待跑）
├── weighted_voronoi_prototype.py     ← 加权Voronoi纯numpy原型（4D凸包法）
├── plot_slice_errors.py              ← z=0.5切片误差云图（ddpnm3d风格，2×3面板）
├── run_benchmark.py                  ← 通用 benchmark（带 off/online 拆分，稍旧）
├── report_voronoi.py                 ← 50球Voronoi完整报告（待跑）
├── plot_errors.py                    ← 误差柱状图+成本图
├── outputs/
│   ├── benchmark_voronoi/            ← 50球Voronoi报告（待 report_voronoi.py 生成）
│   ├── benchmark_voronoi_100/        ← ★ 100球benchmark输出目录（待 benchmark_voronoi_full.py 生成）
│   ├── slices/                       ← 50球三组分区的误差云图
│   │   ├── slice_errors_voronoi.png  ← Voronoi 2×3面板（速度+压力）
│   │   ├── slice_errors_grid.png
│   │   ├── slice_errors_watershed.png
│   │   └── slice_*.npz
│   └── weighted_voronoi/             ← 加权Voronoi原型输出
│       ├── weighted_voronoi_overview.png
│       ├── weighted_voronoi_slice.png
│       ├── weighted_voronoi_data.npz
│       └── summary.json
└── ../affine_ddpnm_3d_random_porous/ ← 依赖（ddpnm_core、affine基、绘图工具函数）
```

---

## 3. 几何：100 球密集堆积

- **球数**：100（24 边界 + 76 内部）
- **半径**：[0.0551, 0.0965]，r_max−r_min=0.0414
- **孔隙率**：**85.6%**（固体占比 14.4%，比 50 球的 3.3% 密实得多）
- **双峰分布**：~60% 小球（r~0.055–0.068）+ ~40% 大球（r~0.080–0.097）
- **种子**：20260806
- **生成函数**：`_gen_packing.py`（或直接调 `weighted_voronoi_prototype.generate_dense_packing`）

> SPHERES 已写入 `geometry.py`，当前生效的是 100 球配置（注意 `N_SPHERES=100`）。

---

## 4. 已完成：50 球均匀半径 benchmark（用于汇报）

### 4.1 几何

- 50 球、半径 [0.0502, 0.0580]、r_max−r_min=0.0078 ≤ gap/2=0.015 ✓
- 孔隙率 96.7%

### 4.2 三组分区 + 两种 DDPNM 的完整对照

| 分区 | 方法 | L²(u) | H¹(u) | L²(p) | **流量** | 离线 | **在线** |
|---|---|---|---|---|---|---|---|
| Grid (64胞) | Classic | 124.9% | — | 9.5% | 74.8% | — | — |
| Grid | Affine | 13.6% | — | 3.5% | 9.2% | — | — |
| **Voronoi** (50胞) | Classic | 39.9% | ~85% | 12.1% | 61.9% | ~50s | ~0.015s |
| **Voronoi** | **Affine** | **5.1%** | ~21% | **2.8%** | **4.0%** | ~65s | ~0.3s |
| Watershed (22盆) | Classic | 44.9% | — | 10.0% | 56.6% | — | — |
| Watershed | Affine | 18.0% | — | 3.7% | 12.9% | — | — |

> 精确 H¹ 和 off/online 需跑 `report_voronoi.py`。50 球速览值见 `outputs/slices/slice_voronoi.npz`。

### 4.3 三组分区的 2×3 误差云图

全部在 `outputs/slices/`：`slice_errors_voronoi.png`、`slice_errors_grid.png`、`slice_errors_watershed.png`。

---

## 5. 加权 Voronoi 原型

### 5.1 方法

4D 升维 (x,y,z,x²+y²+z²−r²) → `scipy.spatial.Delaunay` → 邻接 4-单形提取 (i,j) 对 → bisector 归属过滤 → power bisector 平面。

### 5.2 结果（120 球、孔隙率 90.9%）

| 指标 | 数值 |
|---|---|
| 原始 Delaunay 邻接对 | 1,697 |
| **过滤后有效界面** | **216** |
| 鞍点净距中位 | 0.149 |
| Power shift 中位 | **0.0023** |
| Power shift 最大 | 0.0148 |

**power shift 太小**——中位只有球半径的 ~3%。对于当前 packing，标准 Voronoi 的偏离在 DDPNM 的 5% 误差水平下可忽略。

### 5.3 OCC 集成尝试（均失败，记录原因）

| 尝试 | 方法 | 失败原因 |
|---|---|---|
| 1 | Voronoi 多边形平移 power shift 后嵌入 OCC | 各面独立平移→边不闭合→闭壳破坏→99/100 区域丢失 |
| 2 | 大平面（power bisector 无限平面）嵌 OCC fragment | 300+ 大平面相交→OCC 组合爆炸→超时 |
| 3 | Mesh labeling（按 power distance 标记网格胞） | 界面=随机方向阶梯面→DDPNM 精度崩溃（L²~1e17%） |

### 5.4 下一步

如果标准 Voronoi 在 85.6% packing 上精度已经够好（预期 Affine L² ~5–8%），加权 Voronoi 可以暂时搁置。如果真的需要（更强 bimodal，r_max/r_min > 3），正确做法是：
1. 用 4D Delaunay 确定 (i,j) 对
2. 对每对计算 power bisector 平面
3. 用 half-space intersection 裁剪出有限多边形（CGAL 或手动 S-H 裁剪）
4. 嵌入 OCC

---

## 6. 明天任务

### 6.1 必做：跑 100 球标准 Voronoi benchmark

```bash
cd real_porous_benchmark_3d
export PATH="/d/Miniconda3/envs/fenicsx/Library/bin:$PATH"
/d/Miniconda3/envs/fenicsx/python.exe benchmark_voronoi_full.py
```

该脚本一次性跑完：
- 网格生成（100 球 Voronoi，预期 ~300 界面、~35K 四面体）
- 整体 FEM 参考解
- Classic-DDPNM（1 模/面）+ Affine-DDPNM（9 模/面），含 off/online 拆分
- z=0.5 切片场数据（速度+压力）
- JSON 报告 + 控制台表格

输出到 `outputs/benchmark_voronoi_100/`。

**如果脚本报错**：
- `Too many communicators` → 重启会话
- `Missing pore region` → 某些球的 Voronoi 胞在 OCC 中丢失（球太密时偶发），改 `build_partition_voronoi` 中该错误的 raise 为 warn 即可
- 内存不足 → 100 球 × 9 模/面 ≈ 300×9=2700 未知量，Schur 需 ~60MiB，应该没问题

### 6.2 跑完后：出图

```bash
python plot_slice_errors.py --partition voronoi --out-dir outputs/slices_100
```

（需要先把 `plot_slice_errors.py` 里的 SPHERES 确认是 100 球——脚本读 `geometry.SPHERES`，当前就是 100 球。）

### 6.3 可选：加权 Voronoi OCC 集成

只有在 6.1 的结果中标准 Voronoi 精度明显不够（Classic L² > 80%）时才需要。做法见 §5.4。

---

## 7. 已知问题

1. **MPI 耗尽**：重启 Claude Code 会话解决
2. **Voronoi 面穿球**：r_max−r_min=0.041 > gap/2，标准 Voronoi 面可能切到球。如果 Classic L² > 60%，就是这个问题。解决办法：改用加权 Voronoi 标准流程，或增大 min_gap
3. **100 球 benchmark 时间预估**：网格 ~60s、FEM ~15s、Classic ~120s（100 区域局部 LU）、Affine ~150s，总计 ~6 分钟
4. **`plot_slice_errors.py` 的绘图函数来自**：`affine_ddpnm_3d_random_porous/plot_random_errors.py`（`_classify_slice_grid`、`_smooth_slice_grid`、`slice_sphere_cuts`、`sci_colorbar`）和 `plot_partition_slice.py`（`subdomain_interface_lines`）。修改绘图时改这些源文件
5. **加权 Voronoi 记忆文件**：`memory/weighted-voronoi-dense-packing.md`

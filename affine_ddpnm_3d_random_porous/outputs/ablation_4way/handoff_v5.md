# 下一个窗口的交接文档：DDPNM 三维随机球——公平加速比对照与可视化（2026-08-05 第五版）

> 复制本文件全文到下一个窗口。接手者不能只读结论，必须按文中的数学对象、
> 代码位置与验收条件执行。本文承接 `deepseekoutput/watershed_partition_handoff.md`
> 第四版：**公平加速比对照（第四版 5.2）与三档网格收敛（5.4-1）已完成**，
> 并新增两张图（4 组合误差云图、分区截面图）与一个绘图 bug 修复（界面
> 迹线 0 条问题）。所有路径基于 `D:\hu\tongjiproj\20260727`。

---

## 0. 你的身份与本轮总目标

接手 `affine_ddpnm_3d_random_porous/` 中的 DDPNM 三维随机球项目。
第四版第 5 节的待办已执行一部分，本轮产出：

1. **公平加速比对照**：watershed 在 0.10 族（11,404 胞）重跑，与冻结
   Voronoi（15,249 胞）同分辨率对比；
2. **watershed 三档网格收敛**（0.16/0.13/0.10 族，4,311/6,586/11,404 胞）；
3. **RESULTS.md 表 8** + 表 7 归因修正（"网格更粗贡献退化"已证伪）；
4. **两张图**：4 组合速度误差云图（01 风格）、分区截面图（只要边和球）；
5. **绘图 bug 修复**：切片网格非共形导致界面迹线 0 条（见 3.4，勿回退）；
6. **结果评估口径**：哪些数字能站住、怎么讲（见 2.6）。

禁止沿用旧文档的错误表述：孔数≠27 是预期、Voronoi 面≠物理孔喉、
"网格分辨率贡献 Affine 退化"已被证伪（见 3.1）。

---

## 1. 环境与常用命令

- **运行前必须**：
  `export PATH="/d/Miniconda3/envs/fenicsx/Library/bin:$PATH"`，且
  **python 必须用 fenicsx 环境的解释器**
  `/d/Miniconda3/envs/fenicsx/python.exe`（`python` 默认指向 base）。
- 不带 PATH 时 `np.linalg.eigvalsh`/`scipy.linalg.eigh` 原生崩溃（bash
  报 exit 127、缓冲输出全丢）——先加 flush 再判。
- 单元测试：`cd affine_ddpnm_3d_random_porous && /d/Miniconda3/envs/fenicsx/python.exe test_watershed_partition.py`
  （5 个测试：3 个图基 + 2 个网格基浮游盆合并）
- 基准重跑（0.10 族示例）：`... python run_random_benchmark.py --partition
  watershed --mesh-size 0.10 --sphere-size 0.05 --boundary-size 0.065
  --sphere-band 0.115 --boundary-band 0.10 --out-dir outputs/<dir>`
- **绘图**：
  `... python plot_four_way_errors.py` → `outputs/ablation_4way/four_way_speed_error.png`
  `... python plot_partition_slice.py` → `outputs/ablation_4way/partition_slice.png`
  （两个脚本共享 `subdomain_interface_lines`，位于 plot_partition_slice.py）
- 其余脚本见第四版第 1 节（compare_partitions_formal.py、verify_basis_suitability.py、
  ablation_4way.py、run_watershed_exact_schur.py 等，均未动）。
- 后台跑命令别用 `| tee <不存在的目录>/log`（tee 先死、输出全丢、任务
  报 exit 1 但 python 实际完成）——**先 mkdir 或干脆不用 tee**（harness
  自带 stdout 捕获）。

---

## 2. 本窗口已完成的工作

### 2.1 公平加速比对照（第四版 5.2）

**参数族**：0.10 族 = 冻结 Voronoi 0.13 族等比缩放 ×0.769（mesh 0.10 /
sphere 0.05 / boundary 0.065 / sphere_band 0.115 / boundary_band 0.10；
interface 尺寸场对 watershed 不适用）。输出 `outputs/fair_speedup/watershed/`。

结果（11,404 胞 / **50 盆** / 257 界面 / FEM 65,278 dofs / 92.9 s）：

| 指标 | V×Classic | V×Affine | W0.10×Classic | W0.10×Affine |
|---|---:|---:|---:|---:|
| 界面未知量 | 114 | 1,026 | 257 | 2,313 |
| 速度相对 L2 | 65.32% | 6.51% | 46.73% | 14.32% |
| broken-H1 | 85.85% | 17.56% | 68.44% | 34.42% |
| 压力相对 L2 | 10.30% | 2.03% | 6.51% | 3.13% |
| 出口流量误差 | 72.72% | 2.71% | 46.61% | 11.39% |
| 首次求解 | 12.2 s | 15.2 s | 15.0 s | 24.5 s |
| 相对 FEM 加速 | 9.55× | 7.66× | 6.17× | 3.78× |

- 表 7 两结论同分辨率下均成立：Classic 画法优势（46.7% < 65.3%）；
  Affine 退化 ~2.2×（14.3% vs 6.5%）。
- 加速比同分辨率下约减半（3.78× vs 7.66×）：残余差距来自界面数
  257 vs 114——离线库近似线性于界面数（15.0 s vs 12.2 s），是画法
  本身的离线代价，不是伪象。

### 2.2 watershed 三档网格收敛（第四版 5.4 第 1 条）

0.16/0.13/0.10 族（阈值 0.02/0.05 固定），输出
`outputs/fair_speedup/watershed_016/`（0.13 族复用 `ablation_4way`）：

| 档 | 胞数 | 盆/界面 | Classic L2 | Affine L2 | Classic 加速 | Affine 加速 |
|---|---:|---:|---:|---:|---:|---:|
| 0.16 | 4,311 | 77/326 | 54.99% | 10.89% | 0.96× | 0.80× |
| 0.13 | 6,586 | 82/369 | 43.55% | 11.31% | 1.59× | 1.10× |
| 0.10 | 11,404 | 50/257 | 46.73% | 14.32% | 6.17× | 3.78× |

**无 h 收敛**：两方法误差在 4.3k–15k 胞区间由画法模型误差主导。

### 2.3 RESULTS.md 更新

- 新增**表 8**（同分辨率公平对照 + 三档收敛，含口径脚注）；
- 修正表 7 归因："网格更粗也贡献退化"删除，改指表 8；加速比条目指表 8；
- 表 7 后补图引用行（four_way_speed_error.png）。

### 2.4 可视化（新增）

- **`outputs/ablation_4way/four_way_speed_error.png`**（`plot_four_way_errors.py`）：
  2×4 布局照 `outputs/benchmark/01_slice_error_fields.png` 风格——顶行四组合
  DDPNM 速度 |u|（viridis 共享色标）、底行四组合速度误差
  ||u_dd|−|u_fem||（turbo 每格独立 98 分位色标，标题带 rel L2 实测数字）；
  每格有界面迹线（黑）与球切面（灰）。
- **`outputs/ablation_4way/partition_slice.png`**（`plot_partition_slice.py`）：
  1×2 纯几何——z=0.5 截面上子区域边界线 + 球切面，无场数据；(a) Voronoi
  27 子区域直边分割、(b) watershed 82 盆折线分界；标题自动带子区域/界面数。

### 2.5 绘图 bug 修复：界面迹线 0 条（重要，勿回退）

**现象**：分区截面图、4 组合误差图上没有子区域边界线（只有球）。

**根因**：`plot_random_errors.py` 的 `_subdomain_boundary_lines` 只收集
"恰好被 2 个切片三角形共享且两三角形子区域 label 不同"的边。但**切片
三角形网格是每个子区域独立三角化的非共形拼合**——跨子区域缝两侧各是
轮廓边（只被 1 个三角形使用）、无共享边；所有 len==2 共享边都在同一
子区域内（label 相同）。实测 Voronoi 4484 条共享边全部同 label → 0 条
边界。四组合图之前"看起来有线"是误读（球盘边缘）。

**修复**：`plot_partition_slice.py` 新增 `subdomain_interface_lines
(slice_points, slice_triangles, spheres, radii, z_value)`：收集所有
**轮廓边**（cnt==1），裁剪掉①立方体外壁段（两端都在 x=0/1 或 y=0/1
同一壁上）②球孔段（段中点在某个球切圆盘内，容差 1e-9）；剩余即子区域
间边界（非共形缝两侧重合轮廓边各画一遍 = 一条实线）。实测 Voronoi
4,282 段、Watershed 1,916 段。`plot_four_way_errors.py` 从
plot_partition_slice import 该函数（**两脚本有依赖，勿拆散**）。

### 2.6 结果评估口径（数字怎么讲）

- **能站住的数字**：V×Affine 压力 2.03% / 流量 2.71% / L2(u) 6.51%；
  守恒全程舍入级；**在线加速 136×（0.86 s vs 117 s）**——多工况口径，
  离线 15.2 s 付一次。
- **要防的数字**：首次求解 7.66×（一次性成本摊进每次，平庸）；
  小网格 0.80×（离线反超 FEM）；无 h 收敛（6.5% 是画法+模态的误差地板）。
- **叙事**："单工况 7.7×，多工况下每工况 136×"；watershed 退化 =
  弯折界面超出仿射模态（有法向离散度证据），不是网格问题；
  "无收敛 → 模型误差主导 → 升阶方向"（broken-H1 34% 是靶子）。

---

## 3. 本窗口的关键发现（必须理解）

### 3.1 无 h 收敛，分辨率不贡献退化（修正第四版 2.3 归因）

三档加密（4,311→6,586→11,404 胞）后 Affine L2 为 10.9% → 11.3% →
14.3%，不改善反略升；Classic 55.0% → 43.6% → 46.7% 无系统趋势。
误差由画法模型误差主导。**不要写"网格更粗贡献退化"。**

### 3.2 盆数随网格非单调（77/82/50）

阈值 0.02/0.05 固定的持久性过滤对采样密度敏感。分区拓扑非网格不变；
Voronoi 的 27 盆由球排布固定。任何 watershed 收敛性论证须按网格重标
阈值或把盆数变化当方法固有属性。

### 3.3 加速比低端交叉点

4,311 胞处 W 加速 < 1×（Classic 0.96× / Affine 0.80×）：FEM 仅
14.4 s，离线库（15.0/16.4 s，≈线性于界面数）反超。DDPNM 加速需 FEM
参考足够大摊薄离线成本；写加速比声明须注明网格规模。

### 3.4 其余继承（第四版，仍有效）

浮游盆合并 `merge_floating_basins` 不可删（种子守卫不变式）；0.13 族
是冻结基准参数；Hausdorff 配对三个实现坑勿回退；dispersion 双口径按
代码公式引用；界面线修复见 2.5（勿回退到 `_subdomain_boundary_lines`）。

---

## 4. 当前状态总结

| 项 | 状态 |
|---|---|
| 5.1 几何对照 / 5.2 基核验 / 5.3 消融 / 5.4 精确 Schur | ✅ 均未动 |
| 浮游盆合并 + 单元测试 | ✅ 5 个全过 |
| **公平加速比对照** | ✅ `outputs/fair_speedup/watershed/` |
| **三档网格收敛** | ✅ `outputs/fair_speedup/watershed_016/` + 复用 ablation |
| RESULTS.md 表 8 + 表 7 归因修正 | ✅ |
| **4 组合误差图 / 分区截面图** | ✅ `outputs/ablation_4way/` 两张 png |
| 界面线 0 条 bug | ✅ 修复（subdomain_interface_lines） |
| 交接文档 | 🔄 本窗口（第五版，输出在 ablation_4way/） |

## 5. 接下来要做什么（按序）

### 5.1 升阶实验（第一优先，结果评估的补强）

面内 P1→P2 或残差驱动界面富集，压低 broken-H1——watershed 上 Affine
的 34.4%（0.10 族）是现成靶子。做出来：①误差数字上台阶 ②把"无 h
收敛"变成"误差地板可移动"的正面叙事。参考 2D 项目 `ddpnm2d/basis_2d.py`
的 `PolynomialNormalBasis`（interface_order ∈ {0,1}）。

### 5.2 种子扫描（一般性支撑）

3+ 个随机排布（改 `random_porous.py` 的 SPHERES 表或加种子参数），重跑
V×Affine 与 W×Affine 各一档（0.13 族即可）——支撑"6.5% 不是单算例
运气"。

### 5.3 Voronoi 收敛 + watershed 阈值重标

Voronoi 目前只有 0.13 族一档；补 0.10/0.16 两档成三档曲线。watershed
收敛须先按网格重标持久性阈值（盆数 77/82/50 非单调是障碍）。

### 5.4 其余（第四版继承）

浮游盆商空间局部求解（改 `ddpnm_core/stokes_operator.py`，风险大须
回归）；匹配 2D/3D 基准组（`ddpnm_2d_random_porous` 有 STRICT
2D/3D 对比基础设施）；画法 B 等净距双曲面界面。

## 6. 验收标准（本窗口已达成）

- ✅ 同分辨率对照：分辨率差距 2.3× → 1.34×；两画法结论不变（Classic
  46.7% < 65.3%；Affine 14.3% > 6.5%）；
- ✅ 三档收敛齐全（4,311/6,586/11,404 胞），"无 h 收敛、模型误差主导"
  结论写入 RESULTS.md 表 8；
- ✅ 旧归因"网格更粗贡献退化"已修正并注明；
- ✅ 盆数网格依赖（77/82/50）与加速比低端交叉点（<1×）已记录；
- ✅ 两张图产出且风格统一（01 同款），界面迹线修复后 4,282/1,916 段；
- ✅ 守恒诊断舍入级、Schur 对称误差 0（三档均）；
- ✅ 单元测试全过、Voronoi 基线未动、`_diag_*.py` 保留。

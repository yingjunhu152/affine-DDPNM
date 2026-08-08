# 下一个窗口的交接文档：DDPNM 三维升维——科研叙事与补强路线（2026-08-05 第七版）

> 复制本文件全文到下一个窗口。本文是第六版（`handoff_v6.md`，同目录）的
> 全面更新版：**继承了第六版全部执行记录与 A–H 待办，新增分区方法数学文档
> （`deepseekoutput/ddpnm_3d_partition_methods.tex`，已编译）、LaTeX 环境修复
> 记录、教材章节核实表与"给导师的三句话"口径**。接手者不能只读结论，必须按
> 文中的数学对象、代码位置与验收条件执行。所有路径基于
> `D:\hu\tongjiproj\20260727`。

---

## 0. 你的身份与总目标

接手 `affine_ddpnm_3d_random_porous/` 中的 DDPNM 三维随机球项目。
**两个目标**：

1. **论文目标**：把现有数值结果组织成"升维机理 + 模态设计 + 半理论误差
   界"的数值方法论文。定位：**CMAME 现实目标、JCP 冲刺目标、C&F/IJNMF/AWR
   稳妥落点**（见第 2 节三个验收条件）。
2. **执行目标**：按第 6 节待办 A–H 补强实验与理论，全部有验收标准。

禁止沿用旧表述：孔数≠27 是预期、Voronoi 面≠物理孔喉、"网格分辨率贡献
Affine 退化"已证伪、"硬搬 2D 产生灾难性后果"是稻草人（见 2.2 的正确表述）。

---

## 1. 环境与常用命令

### 1.1 Python（与第六版相同，勿改）

- **运行前必须**：
  `export PATH="/d/Miniconda3/envs/fenicsx/Library/bin:$PATH"`，且
  **python 必须用 fenicsx 环境的解释器**
  `/d/Miniconda3/envs/fenicsx/python.exe`。
- 不带 PATH 时 `np.linalg.eigvalsh`/`scipy.linalg.eigh` 原生崩溃（exit 127、
  缓冲输出全丢）——先加 flush 再判。
- 单元测试：`cd affine_ddpnm_3d_random_porous && /d/Miniconda3/envs/fenicsx/python.exe test_watershed_partition.py`
  （5 个测试全过）。
- 基准重跑（0.10 族示例）：`... python run_random_benchmark.py --partition
  watershed --mesh-size 0.10 --sphere-size 0.05 --boundary-size 0.065
  --sphere-band 0.115 --boundary-band 0.10 --out-dir outputs/<dir>`
- 绘图：`plot_four_way_errors.py` → `four_way_speed_error.png`；
  `plot_partition_slice.py` → `partition_slice.png`（两脚本共享
  `subdomain_interface_lines`，勿拆散）。
- 均匀球对照实验在独立目录 `affine_ddpnm_3d/`（不是 random_porous）：
  `run_affine_ddpnm_benchmark.py`，结果在
  `affine_ddpnm_3d/outputs/benchmark/` 与 `affine_ddpnm_3d/RESULTS.md`。
- 2D 基准在 `ddpnm_2d_random_porous/`（STRICT_2D_3D_COMPARISON.md 现成）。
- 后台跑命令别用 `| tee <不存在的目录>/log`（tee 先死输出全丢）——
  先 mkdir 或不用 tee。

### 1.2 LaTeX（第七版新增，本机已修好）

- 本机 texlive 2025 为精简安装。**第七版会话已修复**：
  - `tlmgr install tcolorbox cleveref pdfcol oberdiek`（tcolorbox 依赖链）；
  - pgf 缺 `tikzfill.image.sty`（tcolorbox skins 库需要、仅加载不调用）→
    已放 stub：`~/texmf/tex/generic/pgf/utilities/tikzfill.image.sty`，
    并 mktexlsr。**勿删**。
- 编译：`cd deepseekoutput && xelatex -interaction=nonstopmode -halt-on-error <file>.tex`
  **跑两遍**（目录/交叉引用）。
- **已知不能本机编译**：参考文档 `deepseekoutput/ddpnm_3d_math_document.tex`
  需要 `multirow`（未装）。如需编译它：`tlmgr install multirow` 后
  跑两遍。
- 长 `\texttt{}` 文件名导致 Overfull 的修法：`\emergencystretch=2em` +
  `\allowbreak{}` 断点 + 表格用 `tabularx` 的 `X` 列。改文档时沿用。

---

## 2. 科研叙事主线（论文骨架，四幕）

### 2.1 第一幕：升维的几何断裂

2D 界面是连心线上的 1D 线段（端点由两个圆支撑，
`ddpnm_2d_random_porous/ddpnm2d/geometry.py` `analytic_throat_cuts`）；
3D 里 1D 线段不能分割体积，子区域构造必须重新发明——Voronoi 胞/垂直
平分面 vs 净距盆/鞍面，且半径不均匀（0.079–0.133）时两者不重合
（鞍点位移中位 0.0073/2.8%、Hausdorff 中位 0.0858，
`outputs/watershed_formal/`）。

### 2.2 第二幕：升维的代数断裂

把 2D 的"每界面一个常数法向系数"硬搬到 3D = Classic-DDPNM：
L2 误差 21%（均匀球）~65%（随机球）。
**正确表述（勿写稻草人）**："硬搬"指**常数模态**的搬运——2D 界面是 1D
线段（面内变化 1 个方向），常数系数丢 1 个方向的线性信息；3D 界面是
2D 曲面（面内变化 2 个方向），丢 2 倍。**表述定稿前必须先做待办 A**
（核实 2D Classic 数字，若 2D 也 ~20–30% 则改为"缺口结构不同"而非
"缺口翻倍"）。

### 2.3 第三幕：双假说判别（核心章节，实验作证）

升维后为什么难？假说 A（压缩比）：3D 每界面 1 代表点代替 ~100 真实节点
（2D 只代替 ~10）；假说 B（模态形状）：单模态表达力不足。
**四线判别（全部有数据）**：

| 判别线 | 数据 | 结论 |
|---|---|---|
| 模态数 1→9 | 21.1%→5.63%（均匀，`affine_ddpnm_3d/RESULTS.md`）；65%→6.51%（随机，表 7） | 模态形状是瓶颈 |
| **点数 9→15** | Uniform-DDPNMT 15 采样点/面 5.47% vs Affine 9 模态 5.63%，**打平**（+40% 未知量 2,160 vs 1,296） | **对冲压缩比失败，A 出局** |
| 网格加密三档 | 10.9→11.3→14.3%（4,311/6,586/11,404 胞）无改善 | 离散误差不是瓶颈 |
| 弯折度 | 246/369 界面 >30° 竖面（5.2）；W 差于 V | **主因 = 模态形状与界面几何的匹配** |

### 2.4 第四幕：解法与边界

Affine-DDPNM（每面 9 模态 {1,s,t}×{n,t1,t2}，`affine_face_basis.py`）：
平面界面上 5.6–6.5%、流量误差 <3%（V×Affine 2.71%）、压力 2.03%、
在线 136×（0.86 s vs 117 s）、守恒舍入级、Schur 对称误差 0；
弯折界面（watershed）退化（14.3%）不是失败，是主因的可控验证。
浮游盆合并（`merge_floating_basins`）是数值稳定性修复，不可删。

### 2.5 期刊定位与三个验收条件

| 层 | 期刊 | 条件 |
|---|---|---|
| 冲刺 | JCP | 误差界严格（有证明）+ 控制算法带收敛性论证 |
| 现实 | CMAME | 半理论界（严格或启发式）+ 完整数值 + 控制算法 + 2D/3D 匹配对比 |
| 稳妥 | C&F / IJNMF / AWR | 理论启发式或控制算法缺失 |

**决定档次的三个条件**：①界严格性（弯折阶梯界面的误差界是否证明过）；
②"控制 H1 误差"有算法（残差驱动指示器 → 增选模态 → H1 单调下降且被界
控制，带论证）——只有解释没有算法会被审稿人挑战；③2D/3D 匹配对比严格
（STRICT 基础设施现成）。

---

## 3. 关键发现（必须理解，勿回退）

1. **无 h 收敛**：三档加密 Affine L2 10.9→11.3→14.3%，Classic
   55.0→43.6→46.7% 无系统趋势——误差由画法模型误差主导（RESULTS.md
   表 8）。"网格更粗贡献退化"已证伪，勿再写。
2. **盆数随网格非单调**（77/82/50）：watershed 分区拓扑非网格不变；
   换种子/换网格必须重做阈值扫描（0.02/0.05 的稳定区间）。
3. **加速比低端交叉点**：4,311 胞处 W 加速 <1×（0.80×）——离线库
   （≈线性于界面数：12.2 s/114、15.0 s/257、15.0 s/326）反超 FEM
   （14.4 s）。写加速比声明必须注明网格规模；叙事用"单工况 7.7×，
   多工况每工况 136×"。
4. **界面线 bug（已修）**：切片网格每子区域独立三角化（非共形），
   `plot_random_errors._subdomain_boundary_lines` 返回 0 条；
   `plot_partition_slice.subdomain_interface_lines`（轮廓边 cnt==1，
   裁剪外壁段 + 球孔段）是正确实现，勿回退。
5. **浮游盆合并**（种子守卫不变式）与 Hausdorff 配对三坑（faces/
   used_pairs 索引错位、球对过滤不可行、点到多边形距离）——勿回退。
6. **dispersion 双口径**：按代码公式引用（Σ(1−n_f·n̄)/A），勿按 docstring。

---

## 4. 已完成的工作（第六版继承 + 第七版新增，全部验收）

| 项 | 状态 |
|---|---|
| 5.1 几何对照 / 5.2 基核验 / 5.3 消融 / 5.4 精确 Schur | ✅ 均未动 |
| 公平加速比对照（watershed 0.10 族，11,404 胞） | ✅ `outputs/fair_speedup/watershed/` |
| 三档网格收敛（0.16/0.13/0.10） | ✅ `outputs/fair_speedup/watershed_016/` |
| RESULTS.md 表 8 + 表 7 归因修正 | ✅ |
| 4 组合误差图 / 分区截面图 | ✅ `outputs/ablation_4way/*.png` |
| 界面线 bug 修复 | ✅ `subdomain_interface_lines` |
| 均匀球对照实验（独立目录） | ✅ `affine_ddpnm_3d/`（Classic 21.1%→Affine 5.63%，15 点打平 9 模态） |
| 单元测试 | ✅ 5 个全过 |
| **分区方法数学文档**（第七版新增） | ✅ `deepseekoutput/ddpnm_3d_partition_methods.tex` + PDF（17 页，xelatex 两遍编译过） |
| **LaTeX 环境修复**（第七版新增） | ✅ tlmgr 装 tcolorbox/cleveref/pdfcol/oberdiek + tikzfill stub（见 1.2） |
| **教材章节核实**（第七版新增） | ✅ 见 §8 附表（写论文引用可直接用） |
| **"给导师的三句话"口径**（第七版新增） | ✅ 见 §9（watershed vs Voronoi 选择说明） |

**结果评估口径（数字怎么讲）**：能站住的——V×Affine 压力 2.03%/
流量 2.71%/L2 6.51%、守恒舍入级、在线 136×；要防的——首次求解 7.66×
（用在线口径替代）、小网格 0.80×、无 h 收敛（模型误差主导叙事）。

---

## 5. 第七版新增成果详情

### 5.1 分区方法数学文档（论文第 2 章的深化素材）

`deepseekoutput/ddpnm_3d_partition_methods.tex`（与 `ddpnm_3d_math_document.tex`
同目录、同版式：ctexart + keybox/algobox/定义-命题环境）。内容：

1. **问题设置**：分区定义（含"每子区域必须触固壁"的代数原因：浮游盆 →
   刚体模态零空间 → Schur 爆炸）；净距场；支撑数维数计数
   $\dim\mathcal M_k=4-k$（3D：k=2 面 / k=3 线 / k=4 点）；
2. **四种分区方法**：规则解析 64 胞（半径公式
   $\sqrt{b(0.2-b_0)^2+(3-b)0.15^2}-0.105=b_0$，b=边界轴数）、球心 Voronoi
   27 胞（鞍点位移命题 $|r_i-r_j|/2$）、等净距双曲面（不构成全局分区）、
   持久性 watershed 82 盆；
3. **五个手算例**（全部数字已逐项验证）：
   - 例 1：连通细胞图上的 watershed 全流程——**注意**：1D 几何里内部固体
     区间会切断流体域（胞图不连通），因此例 1 不用几何、直接给
     (胞邻接图, d_K) 剖面——这正是代码的输入形式；阈值
     τ_abs∈{0.02,0.06,0.11} → 3/2/1 盆；
   - 例 2：两不等半径球（r=1,2, D=4）：Voronoi 面 x=2 vs 等净距轴交 x=1.5，
     位移 0.5=|r1−r2|/2；2D 间隙线段中心恰在鞍点 vs 3D 平面偏离；
   - 例 3：支撑数维数（ℝ³ 三点等距直线 (0.5,0.5,z)、四点外接球心）；
   - 例 4：规则 3×3×3 阵列——四类最大空球半径 0.1548/0.1213/0.1006/0.0884、
     喉道鞍点 4 球支撑（dim M_4=0）、**解析界面 {0.2,0.5,0.8} vs 球心
     Voronoi {0.35,0.65}，64 vs 27——方法身份之差**；
   - 例 5：浮游盆判别思维练习；
4. **分区-求解器接口**、**延伸阅读**（教材章节见 §8）。

**与叙事的关系**：文档把"升维的几何断裂"（2.1）写成可手算验证的
对象；例 4 的 64 vs 27 是随机项目 82 vs 27 的正则对应物；例 2 的
$|r_1-r_2|/2$ 位移与命题 2 一致。

### 5.2 已确认的关键理解（勿再怀疑）

- **"理想分区"（过球心坐标平面族切出 4×4×4 胞）就是规则项目的现有实现**
  （`ddpnm_3d_uniform_spheres/ddpnm3d/geometry.py`：`PARTITION_PLANES =
  SPHERE_GRID`）。不存在"未实现"待办。两个精确边界事实：界面平面穿过
  球心（每球被 3 张平面劈成 8 片，共 216 patch）；边界胞界面 x=0.2 不是
  两最大空球的垂直平分面（平分面在 x=0.2356），内部胞界面恰是平分面。
- **盆（basin）的严格定义**：超水平子图滤波（出生 β、鞍层死亡 δ、
  持久性 ρ=β−δ）→ 双阈值剪枝 + 全局最大幸存者 → 降序标签传播（等值
  平台成批、平局取最小标签、升序回填至不动点）得到的胞图划分；界面 =
  异标签公共面集按边连通分量拆分。见文档 §2.4 与 §3 例 1。
- **watershed 相对 Voronoi 的定位**：不是"更正确"，是"回答不同问题"——
  盆由净距极大值组织（82），Voronoi 胞由球心组织（27=N_s）；选 watershed
  是选"孔体=空隙"的语义。

---

## 6. 后续补强（A–H，按执行顺序，全部有验收标准）

### A. 核实 2D 对照数据（先行，约半小时）
读 `ddpnm_2d_random_porous/RESULTS.md` + STRICT_2D_3D_COMPARISON.md：
2D Classic 的 L2 水平、interface_order 0/1（P0/P1）数字、模态数→误差
曲线。**验收**：2D/3D Classic 对照表 + 2.2 表述定稿（决定"缺口翻倍"还是
"缺口结构不同"）。

### B. 表述修正（1 天）
按 2.2 改主线 1；"原因 1/2"改为"**精度主因 = 模态形状-几何匹配；成本
瓶颈 = 未知量阶数 O(h⁻²)**"；broken-H1 进误差结构分析（见 F）。
**验收**：第 2 节四幕定稿，无稻草人、无原因-后果错位。

### C. 种子扫描（1–2 天，可挂后台）
3+ 排布：参数化 `_gen_spheres.py`（9 内部球 + 18 壁球、无重叠、孔隙率/
配位数相近）；每种子 V×{Classic,Affine}、W×{Classic,Affine}（0.13 族）。
**关键坑：每种子重做 watershed 阈值扫描**（0.02/0.05 的稳定区间）。
**验收**：3+ 种子 × 4 组合 report json + 均值±标准差（6.5%/14% 量级散布）。

### D. 误差分解实验（1–2 天）
三块分离：离散误差（精确 FE-trace Schur 为基准——`run_watershed_exact_schur.py`
现成；Voronoi 舍入级 / watershed 3.2e-7）；画法误差（光滑化/理想界面
对照）；模态误差（Classic vs Affine 差分）。**验收**：误差分解表
（三项 × 四组合），主因叙事从排除法升级为可验证分解。

### E. 谱分析半理论（1–2 个月，最大块，范围锁定）
1. 平面界面仿射模态完备性（平凡可证）；
2. 模态空间谱 → 模态投影误差 ||u − P_modal u|| → L2/H1 误差地板的
   理论预测，与实测 6.5%/14% 对照；
3. 弯折（阶梯）界面法向跳变进入误差界——严格界做不出就退启发式界 +
   数值吻合（仍是 CMAME 级）；
4. broken-H1 与 L2 是同一谱的两个范数，一并处理。
**不做**：完整先验误差分析（角点奇异性引理证明会失控）。
**验收**：理论预测地板 vs 实测地板对照表（L2 与 H1 各一组）+ 对四线
判别的逐一解释（理论解释四个观察，而非独立估计）。

### F. broken-H1 处理（2–3 天 + 视情况）
**必做（解释）**：误差结构分析——broken-H1 高是界面薄层导数误差集中
（误差云图证据：高亮贴界面迹线），与主因同步（V 17.56% vs W 34.42%，
加密无改善），工程量（流量 2.71%）不受影响。**可选（压它）**：面内 P2
富集一档（{1,s,t,s²,st,t²}×{n,t1,t2} = 18 模态/界面，V 上跑），预计
17.6%→~10%——验证"误差地板可移动"，参考 2D `ddpnm2d/basis_2d.py`
`PolynomialNormalBasis`。**验收**：broken-H1 有解释、有空间证据、有
（或明确没有）P2 对照。

### G. 2D/3D 匹配对比（1–2 周，STRICT 基础设施现成）
同孔隙率/配位数匹配的 2D vs 3D 基准（`ddpnm_2d_random_porous` strict
框架）：Classic/Affine 的 L2/H1/通量/未知量/加速比对照。
**验收**：跨维度对照表 + "界面维度 1D→2D"的定量支撑。

### H. 升阶实验（可选，与 F 合并考虑）
HODDPNM 式残差驱动界面富集：同时压 broken-H1（34% 靶子）与未知量阶数
O(h⁻²)→O(h⁻¹)（承接成本瓶颈叙事）。**验收**：broken-H1 单调下降 +
未知量/精度权衡曲线。C、E 之后的增强项，非主线必需。

---

## 7. 时间线建议与论文结构草案

| 周 | 工作 |
|---|---|
| 1 | A + B + C 启动 |
| 2 | C 收尾 + D |
| 3–4 | F 必做 + G 启动 |
| 5–10 | E（谱分析）+ G 收尾 |
| 11–12 | H（若做）+ 写作 |

论文结构：1 引言 → 2 子区域构造的维度断裂 → 3 界面模态（常数→仿射 +
浮游盆稳定性）→ 4 误差机理判别（四线 + 误差分解）→ 5 模态空间谱与
误差地板 → 6 控制 H1 误差（若做）→ 7 数值验证（均匀 + 随机多种子 +
2D/3D 匹配 + 成本）→ 8 结论。

**论文第 2 章的写作素材**：`ddpnm_3d_partition_methods.tex` §2–§3
（四种分区方法 + 手算例）可直接改编；§2.1 的"几何断裂"论述建议从
例 2（两不等半径球）切入。

---

## 8. 教材章节核实表（第七版新增，写引用直接用）

| 主题 | 材料 | 章节 |
|---|---|---|
| Voronoi/Delaunay | de Berg et al., Computational Geometry (3rd ed., Springer 2008) | Ch 7（p.147 起）、Ch 9（p.191 起） |
| 加权 Voronoi/中轴 | Okabe–Boots–Sugihara–Chiu, Spatial Tessellations (2nd ed., Wiley 2000) | Ch 2（基础）、Ch 3（广义：§3.1 加权、§3.1.2 加性加权=Apollonius、§3.5.4 中轴） |
| 持久性/合并树 | Edelsbrunner–Harer, Computational Topology (AMS 2010) | Ch VI（Morse 函数）、Ch VII（§VII.1 持久同调 p.149、§VII.3 扩展持久性）、Ch VIII（稳定性） |
| watershed | Soille, Morphological Image Analysis (2nd ed., 2003) | Ch 9 Segmentation（p.267，watershed 所在；**不是 Ch 12**）；Ch 7 测地变换 |
| 孔网提取 | Blunt, Multiphase Flow in Permeable Media (Cambridge 2017) | Ch 2 §2.2 孔尺度网络与拓扑描述 |
| 域分解/Schur | Toselli–Widlund, Domain Decomposition Methods (Springer 2005) | Ch 4（§4.3 Schur 补系统、§4.4 离散调和延拓、§4.5 条件数）、Ch 5（原始迭代子结构化） |

经典文献（bibitem 已在分区文档 thebibliography 就位）：Blum 1967；
Vincent–Soille 1991（TPAMI 13(6):583）；Lindquist et al. 1996（JGR 101:8297）；
Silin–Patzek 2006（Physica A 371:336）；Carr–Snoeyink–Axen 2003（contour
trees）；Dong–Blunt 2009（PRE 80:036307，最大内切球孔网提取——与项目
最大空球构造同源）；Sun et al. 2026（arXiv:2510.13429，DDPNM 参考论文）。

---

## 9. "给导师的三句话"口径（第七版新增，答辩/组会可直接用）

> 子区域我比较了两种分区方法：**球心 Voronoi 平面**（以固体球心为生成元，
> 每个球对应一个胞）和**净距 watershed**（以空隙空间到固体的距离场的局部
> 极大值为种子，把空隙分成盆）。我最终选了 watershed，因为 Voronoi 把孔体
> 绑定到固体颗粒上——孔体数必须等于球数（27 个），而空隙拓扑实际给出 82
> 个孔体，且球半径不等时 Voronoi 界面会系统性偏离真实喉道鞍点
> $|r_i-r_j|/2$；watershed 的孔体对应真实空隙、界面落在两盆相遇的最窄处。
> 消融实验也支持这个选择：在相同界面模态下，watershed 分区比 Voronoi 的
> 误差显著更低（Classic 基：65%→44%）。

**追问防线**（不主动说）：①"82 比 27 更'对'吗？"→ 不是严格意义更对，
是两种方法定义孔体的方式不同，选 watershed 是选"孔体=空隙"的语义；
②"界面为什么锯齿？"→ 离散 watershed 的阶梯法向跳变是已知局限，正是
仿射基在 W 上退化（11.3%）的原因，下一步是分片法向/多 patch 模态，
不是退回 Voronoi；③"盆数随网格变？"→ 固定阈值下 77/82/50，换网格/种子
重扫阈值，列入待办（C），如实报告。
**禁用**："网格越细误差越大是网格问题"（已证伪，无 h 收敛）；
"watershed 比 Voronoi 更正确"（表述过强）。

---

## 10. 验收标准（本窗口继承 + 论文补强）

- ✅ 已验收：公平对照、三档收敛、表 8、两张图、界面线修复、单元测试、
  分区方法数学文档（编译通过）、LaTeX 环境修复、教材章节核实；
- 🔄 论文补强按 §6 A–H 逐项验收（每项有独立验收标准）；
- 🔄 期刊三条件（§2.5）：界严格性、控制算法、2D/3D 匹配——每完成一个
  在论文中形成对应章节；
- 风险退路：界不严格→CMAME 放弃 JCP；无控制算法→落 C&F/IJNMF；
  种子散布大→阈值重标 + 如实报告；时间不够→E 只做完备性 + 数值地板对照。

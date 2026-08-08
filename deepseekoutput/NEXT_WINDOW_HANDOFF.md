# Affine-DDPNM 下一步工作大纲 --- 2026-08-07

> 基于 HANDOFF_20260807.md、handoff_v8.md、paper_roadmap.md 及 2026-08-07
> 对话中用户指出的理论缺口，合并为统一优先级路线。本文件覆盖上一版
> NEXT_WINDOW_HANDOFF.md（2026-08-04）的内容并更新。

---

## 零、已完成的维护（本日，勿回退）

| 项 | 文件 | 内容 |
|---|---|---|
| 交换恒等式修正 | `affine_ddpnm_math_v7.tex:735` | `R̂_i^u P_i^aff = P_aff R_i^u` → `R̂_i^u P_aff = P_i^aff R_i^u`，两边均映射 R^{9m} → R^{r_i} |
| 列顺序越界修复 | `affine_ddpnm_method_v6.tex:120–150` | 行内公式 → `\[ \begin{aligned} \]` 显示公式，追加置换矩阵 Π 关系 |
| "uniquely determined"修正 | `affine_ddpnm_math_v7.tex:584–588` | 改为 "unique only modulo ker(Ŝ)" |
| ε=0 声明限定 | `affine_ddpnm_math_v7.tex:179–185` | 限定为局部响应用 ε=0，Exact FE-trace 比较器用 ε=10⁻¹⁰ |
| 维数表补全 | `affine_ddpnm_math_v7.tex:684–728` | 引入 r_i^u, r_i^k，补全 N_i^u, N_i^k, R̂_i^u, P_i^aff, P_i^k 维数 |
| "open question"措辞 | `affine_ddpnm_math_v7.tex:796` | 改为 "not established or tested in the present work" |
| 代码命名描述 | `affine_ddpnm_method_v6.tex:358–360` | max_mass_residual 已覆盖九模态，命名为 legacy |
| 编译验证 | 两个 tex | **math_v7: 0 Error / 0 Undefined / 3 minor overfull（预存）；method_v6: 0 Error / 0 Undefined / 0 Overfull** |

---

## 一、理论补强（主线，按数学依赖排序）

### 1.1  嵌套空间正交分解（新定理）

**要证明**：设 Λ₀ ⊂ Λ₁ ⊂ Λ₂ 为 Level 0/1/2 的全局未知量空间（嵌套性由
`basis_3d.py:global_keys` 保证），λ_k 为 Level k 的 Galerkin 解，则

\[
\|\lambda_h - \lambda_k\|_S^2
= \|\lambda_h - \lambda_{k+1}\|_S^2
+ \|\lambda_{k+1} - \lambda_k\|_S^2 .
\]

**证明路线**：Galerkin 正交性（Theorem A.3 已有）+ 嵌套性 → Pythagoras。
难度 **低**（标准 Galerkin 推论，约半页证明）。

**前置**：无（Theorem A.3 现成）。

---

### 1.2  体误差与迹误差分离（新定理）

**要证明**：

\[
\|u - u_k\|_X \le \|u - u_h\|_X + C \|\lambda_h - \lambda_k\|_S .
\]

其中 u = PDE 真解，u_h = C-DD 全迹解重建的速度场，u_k = Level-k DDPNM 重建的速度场。

**拆为两步**：

**(a)** `‖u − u_k‖_X ≤ ‖u − u_h‖_X + ‖u_h − u_k‖_X`（三角不等式，trivial）

**(b)** `‖u_h − u_k‖_X ≤ C ‖λ_h − λ_k‖_S`（核心，需证明）

**步骤 (b) 的分范数情况**：

| 体范数 X | C 的值 | 难度 |
|---|---|---|
| A-范数 (Σ ‖·‖_{A_i}²)^{1/2} | C = 1（Theorem A.3 式(A.4) 直接给出）| 已证 |
| L2 | 需 Friedrichs/Poincaré 逐孔 → C = C_P / √ν | 中 |
| broken-H1 半范数 | 需 Korn + 迹逆不等式逐孔 | 中-高 |

**前置**：Theorem A.3（已有）。

---

### 1.3  谱分析半理论（paper_roadmap.md 待办 E，1–2 月）

四个子任务，按数学依赖排序：

1. **平面界面仿射模态完备性**（引理，几天）
   - 证明 {1, s, t} × {n, t₁, t₂} 在平面三角形面片上 Span 完整仿射牵引空间
   - 注意：这是迹空间的完备性，不是体空间的完备性

2. **模态截断谱 → 显式误差地板**（核心，2–4 周）
   - 定义模态投影算子 P_k : Ŝ-迹空间 → Λ_k
   - 刻画 Ŝ 的谱在模态截断下的衰减
   - 导出 ‖ξ_full − P_k ξ_full‖_Ŝ 的显式上界
   - 与实测地板对照：L2 ~6.5%, broken-H1 ~17.6%

3. **弯折界面法向跳变 → 误差界附加项**（2–4 周）
   - Watershed 的 per-facet normal dispersion 如何贡献 ‖·‖_Ŝ 额外项
   - **退路**：若严格界做不出 → 启发式界 + 数值吻合（CMAME 级）

4. **broken-H1 与 L2 同谱处理**（随 2 一并做）
   - 两者是同一模态截断谱在不同范数下的表现
   - 不是两个独立理论

**明确不做**：完整先验误差分析（角点奇异性常数的引理证明会失控）。

---

### 1.4  自适应富集收敛性（paper_roadmap.md 待办 H，依赖 1.1–1.3）

若做完 1.1–1.3，自然地：

- 1.1 给出 ‖λ_{k+1} − λ_k‖_S = 第 k→k+1 层的**增益**
- 1.2 把增益翻译成体范数误差减小量
- `estimate.py` 的残差指示器作为局部增益的代理 → Dörfler 标记 → 单调下降保证

---

## 二、数值实验（支线，可与理论并行）

### 2.1  种子扫描（handoff_v8.md 待办 C，1–2 天）

3+ 随机球排布，每种子 V×{Classic,Affine}、W×{Classic,Affine}。
**关键坑**：每种子必须重做 watershed 阈值扫描。

**验收**：均值±标准差（~6.5%/14% 量级散布）。

---

### 2.2  误差分解实验（handoff_v8.md 待办 D，1–2 天）

三块分离：
- 离散误差：FE-trace Schur oracle（`run_watershed_exact_schur.py` 现成）
- 画法误差：V vs W 的界面几何差异
- 模态误差：Classic vs Affine 差分

**验收**：误差分解表（三项 × 四组合）。

---

### 2.3  broken-H1 处理（handoff_v8.md 待办 F，2–3 天）

- **必做**：误差结构分析——broken-H1 高来自界面薄层导数集中（误差云图）
- **可选（压它）**：面内 P2 富集 {1,s,t,s²,st,t²}×{n,t₁,t₂} = 18 模态/界面，验证地板可移动

---

### 2.4  2D/3D 匹配对比（handoff_v8.md 待办 G，1–2 周）

同孔隙率/配位数匹配的 2D vs 3D，STRICT 框架现成。

---

### 2.5  表述修正（handoff_v8.md 待办 B，1 天）

按 handoff_v8 §5.3 定稿措辞改主线文档（待用户拍板）。

---

## 三、几何方法探索

### 3.1  Delaunay 四面体分区可行性（handoff_v8.md 待办 I，半天）

纯 numpy scipy.spatial.Delaunay，对冻结 SPHERES 统计：
- 四面体数 / 共享面数
- 被第 4 球穿透的界面数
- 无流体通道的界面数
- 鞍点净距分布 vs Voronoi 的中位 0.0073

**验收**：数字表 → 决定原型做不做。待用户拍板。

---

## 四、论文写作（后期，理论+数值就位后启动）

按 paper_roadmap.md §4 八章结构：

1. 引言（升维动机）
2. 子区域构造的维度断裂（几何）
3. 界面模态：常数 → 仿射（代数）
4. 误差机理判别（四线实验 + 分解表）
5. **模态空间谱与误差地板**（= 理论 1.1–1.3，论文核心章）
6. 控制 H1 误差（若做自适应富集）
7. 数值验证（均匀 + 随机多种子 + 2D/3D 匹配 + 成本）
8. 结论

**论文第 2 章的写作素材**：`ddpnm_3d_partition_methods.tex` §2–§3（四种分区方法 + 手算例）可直接改编。若待办 I 结果正面，第 2 章可增第五种分区方法（过球心面族 Delaunay 四面体剖分）。

---

## 五、执行顺序与依赖图

```
已完成 ────────────────────────────────────────────────────
  tex 八项修复 ✅
  Theorem A.3（最佳逼近）✅
  Corollary A.4（富集保证）✅
  Lemma A.2（代数等价性）✅

第一优先（可在本周启动）────────────────────────────────
  ├─ 1.1 嵌套正交分解 ← 不依赖任何新实验，纯纸笔
  ├─ 2.5 表述修正 ← 待用户拍板 §5.3 措辞
  ├─ 2.1 种子扫描 ← 可挂后台
  └─ 3.1 Delaunay 可行性 ← 半天

第二优先（理论核心 + 误差解剖）────────────────────────
  ├─ 1.2 体-迹误差分离 ← 依赖 1.1 的嵌套引理
  ├─ 2.2 误差分解实验 ← 可与 1.2 并行
  └─ 2.3 broken-H1 解释（必做部分）

第三优先（长周期理论 + 匹配对比）────────────────────
  ├─ 1.3 谱分析半理论 ← 1–2 月，依赖 1.1 + 1.2
  ├─ 2.4 2D/3D 匹配 ← 1–2 周
  └─ 2.3 broken-H1 P2 富集（可选部分）

第四优先（冲刺，依赖 1.3）────────────────────────────
  ├─ 1.4 自适应富集收敛性
  └─ 论文写作
```

---

## 六、风险退路（不变）

| 关卡 | 退路 |
|---|---|
| 弯折界面严格界做不出 | 启发式界 + 数值吻合 → CMAME |
| 控制算法做不出 | 只写"解释"，标题避开"控制" → C&F / IJNMF |
| 种子间散布大 | 阈值重标 + 如实报告散布 |
| 时间不够 | E 只做完备性 + 数值地板对照 |
| Delaunay 结果负面 | 如实报告"平凡方法在随机几何不可行"（可发表的负结果）|

---

## 参考文件索引

| 文件 | 用途 |
|---|---|
| `deepseekoutput/affine_ddpnm_math_v7.tex` | 数学性质（Version 7，已修复）|
| `deepseekoutput/affine_ddpnm_method_v6.tex` | 方法论（Version 6，已修复）|
| `deepseekoutput/HANDOFF_20260807.md` | 2026-08-07 交接文档 |
| `affine_ddpnm_3d_random_porous/outputs/ablation_4way/handoff_v8.md` | 第八版交接（完整科研叙事+待办 A–I）|
| `affine_ddpnm_3d_random_porous/paper_roadmap.md` | 论文路线图 |
| `affine_ddpnm_3d_random_porous/outputs/ablation_4way/taskA_2d_verification.md` | 2D 对照核实记录 |
| `ddpnm_core/assembler.py` | 全局 Schur 装配（S += G_uu, rhs -= G_uk @ p_known）|
| `ddpnm_core/library.py` | 响应库：G = BᵀR |
| `ddpnm_3d_uniform_spheres/ddpnm3d/basis_3d.py` | 3D 基（Level 0/1/2 的 global_keys）|
| `ddpnm_core/estimate.py` | 残差指示器 |
| `affine_ddpnm_3d_random_porous/watershed_partition.py` | Watershed 分区 |

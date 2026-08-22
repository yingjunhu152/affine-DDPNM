# 交接文档：affine_ddpnm_sch_twophase（Stokes–Cahn–Hilliard 两相流实验）

日期：2026-08-22（v2，接替同日 v1）。状态：**文档体系完备（math v1 + 伪代码 v2）；随机 27 球 FEM 臂已验证；Classic-frozen 臂被孔 9 的 CH-Newton 不收敛阻塞（调试中，已有复现脚本与残差追踪）；Bentheimer 几何分支已写入主驱动但未验证；正式 12 臂实验未跑。**

---

## 1. 任务定义

12 个数值实验 = 2 几何 × 3 方法 × 2 耦合：

| 编号 | 几何 | 方法 | 耦合 |
|---|---|---|---|
| E1/E2 | 随机27球 | classic DDPNM（流动 W0n 1 模态 + CH W0ch 1 模态） | frozen / SFI |
| E3/E4 | 随机27球 | Affine-DDPNM（流动 W1v 9 模态 + CH W1ch 3 模态） | frozen / SFI |
| E5–E8 | 反转 Bentheimer | 同 E1–E4 换几何 | — |
| E9–E12 | 两几何 | 单体 FEM（标准 Taylor–Hood P2–P1 + P1 相场，凸分裂） | frozen / SFI |

- frozen = Algorithm S1（流动在 φ⁰ 解一次即冻结，CH 推进）；SFI = Algorithm S2（每步阻尼 Picard ω=0.65，双向耦合）。
- 输出：**相对同耦合 FEM benchmark 的误差表（φ/u 的 broken L²、水切曲线、回收率、出口流量）+ 成本表（墙钟分解、峰值内存、迭代数）**。画图明确暂缓。
- 数学规格：`docs/affine_ddpnm_sch_twophase_math.{md,tex,pdf}`（公式 (S1)–(S23)）。
- 伪代码（v2，矩阵元素级，与代码逐条对齐）：`pseudocode/pseudo_{1..4}_*.tex/.pdf`。

## 2. 文件地图

| 文件 | 内容 |
|---|---|
| `run_sch_experiments.py` | 主驱动：几何分支、dt 调度、6 臂编排、误差/成本表（CSV/JSON/控制台） |
| `sch_core/stokes_operator.py` | 孔内应变率 Stokes（2μ∫D:D，P2–P1），splu 分解复用 |
| `sch_core/flow_library.py` | 算法 B/D：响应库、G=BᵀR、重缩放 (S12)、毛细列 (S11)、COO 三重组 Schur (S13)+对称化、G-行通量提取、核诊断 |
| `sch_core/ch_parts.py` | 算法 C：P1 M/K、端口形状矩阵 {1,s,t}、bvec 扩散通量模板、conv 对流形式、Dirichlet 掩码 |
| `sch_core/ch_step.py` | 算法 E：孔内 CH-Newton（凸分裂 Jac+阻尼线搜索）、反应通量 (S19)、切线列、G^ch=−M_c·bvec·δw、CH-Schur (S21)、η-Picard（full/single） |
| `sch_core/fem_solver.py` | FEM 臂：全局 Stokes（孔内常黏度闭合+毛细力）+ 全局 CH-Newton；PARDISO/splu |
| `sch_core/metrics.py` | broken L² 误差、能量 (S23)、账本 (S22)、PoreRestrictor |
| `_dbg_pore9.py` | **孔 9 失败复现+诊断脚本**（本轮新增） |
| `dolfinx_jit_options.json` | 必须在工作目录（FFCx timeout=900s） |

依赖（sys.path 引入，不改）：`../affine_ddpnm_twophase` 的 `random_porous.py`、`digital_core_partition.py`、`affine_face_basis.py`、`ddpnm3d/basis_3d.py`、`ddpnm_core/`。

## 3. 本轮（v2 会话）改动

1. **伪代码 4 份升级 v2**：补齐矩阵元素级（H_i 显式、G=BᵀR 元素式、COO 三重组、½(S+Sᵀ) 对称化、G-行通量、CH 块消去 M⁰⁰/M⁰Γ、b_adv/N(φ)、G^ch 负定构造、入口反应通量账本、μ_w 参考场 dt），与代码逐行对齐。注意 `pseudo_1_ddpnm_frozen.pdf` 旧版曾被阅读器锁定，v2 存为 `pseudo_1_ddpnm_frozen_v2.pdf`（tex 已是 v2）。
2. **主驱动加几何分支**：`--geometry {random27,bentheimer}`、`--mesh-file`、`--group-pair-patches`；Bentheimer 默认网格 `REF_DIR/outputs/experiment_v2/bentheimer_inverted_cartesian_mesh_c6/bentheimer_voxel_pore_mesh.msh`（27 孔/48078 tets/64 相邻标签对/入口 726 出口 762 三角形，见该目录 VOXEL_MESH_REPORT.json）。`build_experiment_partition()` 走 `build_digital_core_partition`（split_disconnected_interfaces 默认 True，与 BL 单相协议一致）。**已过语法检查，未实际运行验证。**
3. **CFL 默认 0.9→0.45**（cfl 与 cfl-adv）：理由是 dt 调度用 FEM μ_w 参考场，而 DDPNM 重构场局部超限。**但实测 CFL 0.45 未能解决孔 9 失败（见 §5），该默认值是否保留待孔 9 定因后再定。**
4. **ch_step.py 加调试钩子**：环境变量 `SCH_NEWTON_TRACE=1` 时打印每孔每次 Newton 的 |r|∞。

## 4. 已验证状态

**FEM-frozen（随机27球）两轮通过**：
- dt=1.2×5 步：R 0→0.32，能量单调降 6.04→4.28，账本 −1e−4~−1.5e−3，max|φ| 1.09→1.21；
- dt=0.67×9 步：R→0.318，max|φ|→1.23（降 dt 并未压住过冲，过冲主要来自入口突变初值的 CH 松弛，非纯 dt）。
- 几何：15249 tets / 27 孔 / 114 界面；Classic 离线库 14.5s、ChParts 16.8s、primitives=246。

**Classic-frozen：被孔 9 阻塞**（两轮、两种 CFL 均失败于第 1 个 CH 步）。

**FEM-SFI / Classic-SFI / Affine 两臂：未跑**（等待孔 9 解决后统一跑）。

## 5. 孔 9 失败：现象与排查记录（当前工作焦点）

复现（约 2–3 分钟，跳过 FEM 臂）：
```bash
cd D:\hu\tongjiproj\20260727\20260824\affine_ddpnm_sch_twophase
conda run -n fenicsx --no-capture-output python -u _dbg_pore9.py
```

现象（SCH_NEWTON_TRACE 输出，见 `outputs/_dbg_pore9_full.log` 若存在；上一轮前台 tail 截断了前半段）：
- 孔 10–26：第 1 次迭代残差即 ~1e−21（纯油、∇φ⁰=0、η=0，退化解）；
- 孔 0–8（入口区活跃孔）：2–6 次迭代二次收敛，正常；
- **孔 9：|r|∞ 从 25.45 起，15 次迭代卡在 ~17 的常数地板不降**（阻尼线搜索无效 ⇒ 方向在残差承载方向上零分量 ⇒ 怀疑 Jac 奇异/不相容约束，而非强非线性）。

已排除：全局 dt/CFL（0.9→0.45 无效）；η 初值（首轮 η=0）；荷载/Jac 符号（其它孔同代码收敛）。

**下一步排查清单（按优先级）**：
1. 重跑 `_dbg_pore9.py` 完整落盘（`> outputs/_dbg_pore9_full.log 2>&1`），看被截断的前半段：各孔 |u*| 统计表、`global max|u*|`（对照 FEM 参考 ~0.0196）、入口孔列表——确认孔 9 是否为入口孔且 |u*| 异常。
2. 对孔 9 打印残差分量归因：失败迭代中 |r|∞ 的 argmax 落在 r1 还是 r2、该 dof 是否属于 inlet_dofs / port_dofs / 内部；若 argmax 在**被行替换的 Dirichlet 行**上仍不降，说明有两处约束写同一行或 lift 冲突。
3. 检查孔 9 的端口几何特例：入口面与内部界面共享顶点（dof 同属 inlet_dofs 与 port_dofs[j] 是合法的，但若同一 dof 属于**两个** interface 端口，r[n+dof]=w−lift 会被后写的 j 覆盖 → 两个不同 lift 的不相容约束；首轮 η=0 时两 lift 都=0 不冲突，但若失败发生在第二轮 η-Picard 则必现——**先确认失败在第几轮**：trace 显示孔 9 只跑了一批，应是第一轮，暂排除）。
4. 检查 splu 是否对近奇异 J 返回了劣解：解完后回代验证 ‖J·Δ+r‖；若线性残差大，加微小对角正则或改用 MINRES/LU-pivot 检查。
5. 对照实验：把孔 9 的 u* 换成 FEM 场（或置零）再跑 Newton——若收敛，则问题是 Classic W0n 重构场在孔 9 局部异常（尖峰使 b_adv 过大），解法是 dt 调度纳入 DDPNM 场的最坏界或对 u* 限幅（并如实报告）。
6. 若确认是重构场尖峰：公平协议改为 dt = min over {FEM, Classic, Affine} 参考场的 CFL 界（三臂同 dt，报告口径不变）。

## 6. 运行方法（环境备忘）

```bash
cd D:\hu\tongjiproj\20260727\20260824\affine_ddpnm_sch_twophase
# 必须 fenicsx 环境（dolfinx 0.10.0 / scipy 1.18 / numpy 2.5）；KMP_DUPLICATE_LIB_OK=TRUE 已在脚本内设置
conda run -n fenicsx --no-capture-output python -u run_sch_experiments.py --arms Classic-frozen --t-final 6 --out-dir outputs/smoke_x   # 冒烟
conda run -n fenicsx --no-capture-output python -u run_sch_experiments.py --out-dir outputs/rand27                                  # 随机27球 6 臂
conda run -n fenicsx --no-capture-output python -u run_sch_experiments.py --geometry bentheimer --out-dir outputs/bentheimer         # Bentheimer 6 臂
```
- `dolfinx_jit_options.json` 必须在 cwd；form 缓存已建立大半，冷启动 JIT 较慢属正常。
- 输出：`sch_metrics.csv` + `sch_report.json` + 控制台表格（误差列对 DDPNM 臂自动以同耦合 FEM 臂为参考；FEM 臂误差列为 NaN/自比）。

## 7. Bentheimer 注意事项（接入后必看）

- 协议（docs/INVERSE_BENTHEIMER_AND_TIMING_RESULTS.md）：128³ 体素反转 + 最大贯通分量 + 3×3×3 Cartesian 分区 + coarsen=6 体素一致四面体网格；27 孔 / 48078 tets。
- **已知风险**：粗体素接触面上完整 9 模态空间可能病态（单相实验中子体 02/04 完整 Affine 相对残差 1.3e−1/3.2e2，当时用 POD tol=1e-8 稳定化）。若 Affine 臂 Schur 出现大残差/失稳，参照 `run_bentheimer_pod_ddpnm.py` 的谱截断方案处理，并在报告中如实标注。
- eps = eps_factor×h_avg 会随体素网格 h 自动放大（协议一致）。
- FEM 臂规模约 3× 随机 27 球，全局 P2–P1 分解预计 3–6 分钟/次，SFI 臂每 3 步 refresh 一次分解。

## 8. 已修复 bug 清单（v1 会话 14 项，摘要）

1. dolfinx Vector 合并 → numpy；2. cfl 参数名不匹配；3. PoreRestrictor 方向反；4. dt 用 μ_o 慢场 → 改 μ_w 参考场；5. FFCx 10s 超时 → json 配 900s；6. **通量符号反**（m=loads·u=−∫u·n；Q^bnd=−m）；7. DG0 form 编译挂起 → φ̄ 纯代数；8. 账本不闭合 → 入口反应通量 J_in=conv−Σr1/Δt；9. 水切负值 → 对流通量口径；10. sparse norm；11. Newton 首迭代收敛时 lu 未建；12. **孔 9 加阻尼线搜索（未根治，见 §5）**；13. OMP 冲突；14. _factor 元组解包。

## 9. 下一步（按优先级）

1. **定位并修复孔 9**（§5 清单；这是当前唯一阻塞点）。
2. Classic-frozen 冒烟通过 → 6 臂随机 27 球全跑（`--out-dir outputs/rand27`）。
3. Bentheimer 分支验证（先 `--arms FEM-frozen --t-final 6` 冒烟几何/端口/边界标签，再 6 臂）。
4. 汇总两几何误差/成本表交用户（不做图）。
5. max|φ|≈1.2 超界问题：凸分裂允许 O(Δt) 过冲，当前协议接受并在表中标 max|phi| 列；若需压界再调 cfl_adv（注意 §5 定因前不要动 dt 协议）。
6. 遗留清理：`pseudo_1_ddpnm_frozen.pdf` 旧锁解除后用 v2 覆盖回原名；调试脚本（probe/_warmup/_diag/_dbg/verify_signs）实验跑完后归档或删除。

## 10. 时间估算（孔 9 修复后）

随机 27 球 6 臂 ≈ 40–60 分钟；Bentheimer 6 臂 ≈ 1.5–3 小时（FEM-SFI 占大头）。

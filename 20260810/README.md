# DDPNM 3D — Affine-DDPNM 理论与数值验证工程

> 自包含可运行目录。2026-08-10 归档。

## 一、目录结构

```
20260810/
├── README.md                          ← 本文件
│
├── docs/                              ← 三份核心理论文档 (tex + pdf)
│   ├── affine_ddpnm_math_v7.pdf       ← 数学性质：局部核、全局 SPD、Galerkin 最优性
│   ├── affine_ddpnm_method_v6.pdf     ← 方法论：牵引 Ansatz、算法、成本
│   └── affine_ddpnm_theory_revised.pdf ← 理论：最佳逼近、嵌套分解、体-迹分离
│
├── ddpnm_core/                        ← 维度无关核心管线 (15 个 .py)
│   ├── stokes_operator.py             ← 局部 P2-P1 Stokes 算子 + 分解
│   ├── library.py                     ← 响应库：B, R=K⁻¹B, G=BᵀR
│   ├── assembler.py                   ← 全局 Schur 装配：S += G_uu
│   ├── basis.py                       ← 界面基协议 (PrimitiveMode / InterfaceBasis)
│   ├── estimate.py                    ← 残差指示器 (跳变/通量/切向矩/非活跃模)
│   ├── trace_schur.py                 ← 精确 FE-trace Schur (oracle 基线)
│   ├── reconstruction.py / validation.py / fem_utils.py / ...
│   └── ...
│
├── postprocess/                       ← 后处理 (5 个 .py)
│   ├── metrics.py                     ← broken-domain FE 误差分析
│   ├── fields.py                      ← 场重建 (P1 顶点值)
│   └── ...
│
├── ddpnm_3d_uniform_spheres/          ← 3D 均匀 27 球算例
│   ├── ddpnm3d/                       ← 3D 求解器模块
│   │   ├── geometry.py                ← 分区构造 (4×4×4=64 胞)
│   │   ├── basis_3d.py                ← 九模放射击基 (HierarchyBasis)
│   │   ├── solver.py                  ← DDPNM (1 模/面)
│   │   ├── hierarchy.py               ← 自适应层级 (0→1→2)
│   │   ├── trace_schur.py             ← 精确 FE-trace Schur
│   │   └── ...
│   ├── run_*.py                       ← 运行脚本
│   ├── plot_*.py / verify_*.py        ← 画图 & 验证
│   ├── *.md                           ← 算法 & 结果文档
│   └── outputs/                       ← 输出图表 + 数据
│
├── affine_ddpnm_3d/                   ← 均匀球 Affine-DDPNM 对照
│   ├── affine_face_basis.py           ← 单实体每面九模
│   ├── run_affine_ddpnm_benchmark.py  ← Classic vs Affine vs FEM
│   ├── RESULTS.md
│   └── outputs/benchmark/
│
├── affine_ddpnm_3d_random_porous/     ← 3D 随机球 + Voronoi/Watershed 分区
│   ├── random_porous.py               ← Voronoi 分区 (球心垂直平分面)
│   ├── watershed_partition.py         ← 净距 watershed 分区
│   ├── affine_face_basis.py           ← 仿射面基
│   ├── run_random_benchmark.py        ← 四组合消融 (V/W × Classic/Affine)
│   ├── plot_*.py                      ← 四组合误差图 + 分区截面图 + 场误差图
│   ├── ablation_4way.py / compare_partitions_formal.py
│   ├── RESULTS.md / paper_roadmap.md
│   └── outputs/
│       ├── four_way_speed_error.png    ← 四组合速度-误差-成本
│       ├── partition_slice.png         ← 分区截面
│       └── benchmark/                  ← 正式基准输出 (4 张图 + json + csv)
│
└── failed_experiments/                ← 面上自由度增加的两个失败案例
    ├── uniform_point/                 ← 每面 15 均匀采样点 (打平 9 模)
    ├── cross_ddpnmt/                  ← Cross-DDPNMT (交叉向量扩展)
    └── grid_refinement/               ← 三档网格加密 (无 h-收敛)
```

## 二、核心理论 (docs/)

三份文档构成完整理论链：

| 文档 | 核心内容 |
|---|---|
| **math_v7** | 局部 Stokes 算子 ($H_i$, $G_i$, kernel)、条件性 SPD 证明、Appendix A: broken-pressure C-DD 系统 → Galerkin 投影 → 最佳逼近 + 富集保证 |
| **method_v6** | 牵引 Ansatz 符号约定、9 模设计、column ordering、算法框 (离线/全局/重构/验证)、经典方法关系、计算代价 |
| **theory_revised** | Theorem 2.1 (最佳逼近)、Theorem 3.1 (嵌套正交分解)、Corollary 3.2 (三层增益分解)、Theorems 5.1–5.4 (体-迹误差双边分离)、Remark 3.3 (四空间钻石诊断) |

关键结论：
- 仿射解是 C-DD 全迹解在 `Ŝ`-半范数下的最佳逼近
- 嵌套迹空间中的误差缩减严格正交（Pythagorean 恒等式）
- 体误差 = C-DD 离散化误差 + 迹模态截断误差（双边控制）
- `Δ_T`, `Δ_N` 是可控的 error-reduction contrasts，不是唯一因果归因

## 三、3D 均匀球基准 (ddpnm_3d_uniform_spheres/)

**几何**：27 个等半径球 (r=0.105)，4×4×4=64 胞，144 条内部界面。

**运行**：
```bash
export PATH="/d/Miniconda3/envs/fenicsx/Library/bin:$PATH"
/d/Miniconda3/envs/fenicsx/python.exe run_ddpnm_3d.py
/d/Miniconda3/envs/fenicsx/python.exe run_adaptive_ddpnm_3d.py
```

**核心结果（34,341 四面体，混合 DOF 155,907）**：

| 方法 | 界面未知量 | 速度 L² | broken-H¹ | 出口流量误差 |
|---|---|---|---|---|
| Classic-DDPNM (1 模/面) | 144 | 21.08% | 44.83% | 16.46% |
| DDPNMT (3 模/面) | 432 | — | — | — |
| HODDPNM (9 模/面) | 1,296 | 6.72% | — | — |
| Adaptive (142/144 面→HODDPNM) | ~1,278 | 6.73% | — | — |
| Monolithic FEM | 155,907 | 0% (reference) | — | — |

## 四、3D 随机球基准 (affine_ddpnm_3d_random_porous/)

**几何**：27 个随机球（种子 20260804，9 内+18 壁），半径 0.079–0.133。

**两种分区**：
- **Voronoi (V)**：27 胞/114 界面，球心垂直平分面
- **Watershed (W)**：82 盆/369 界面，净距持久性 watershed

**运行**：
```bash
# Voronoi 基准
/d/Miniconda3/envs/fenicsx/python.exe run_random_benchmark.py \
  --partition voronoi --mesh-size 0.10 --sphere-size 0.05 \
  --out-dir outputs/benchmark

# Watershed 基准
/d/Miniconda3/envs/fenicsx/python.exe run_random_benchmark.py \
  --partition watershed --mesh-size 0.10 --sphere-size 0.05 \
  --out-dir outputs/watershed

# 四组合消融
/d/Miniconda3/envs/fenicsx/python.exe ablation_4way.py
```

**核心结果（15,249 四面体，混合 DOF 75,954）**：

| 组合 | 界面未知量 | 速度 L² | broken-H¹ | 压力 L² | 出口流量 | 在线时间 | 加速比 |
|---|---|---|---|---|---|---|---|
| V×Classic (1 模) | 114 | 65.32% | 85.85% | 10.30% | 72.72% | 1.28 s | 9.55× |
| V×Affine (9 模) | 1,026 | **6.51%** | **17.56%** | **2.03%** | **2.71%** | 15.2 s | 7.66× |
| W×Classic (1 模) | 369 | 43.55% | 63.96% | — | 40.15% | — | — |
| W×Affine (9 模) | 3,321 | **14.3%** | **34.42%** | — | — | — | — |
| Exact FE-trace Schur | 21,970 | 1.05e-12 | — | — | — | 532 s | 1× (ref) |
| Monolithic FEM | 75,954 | 0% (ref) | — | — | — | 117 s | — |

**离线/在线时间分离（V×Affine, 15,249 四面体）**：

| 阶段 | 时间 | 说明 |
|---|---|---|
| 离线 (library build) | 12.2 s | 114 界面 × 9 模 = 1,026 次局部 Stokes 求解 + G 装配 |
| 在线 (Schur solve + reconstruct) | 3.0 s | 1,026×1,026 稠密 Schur (当前) / 稀疏 |
| 总计 (首次) | 15.2 s | vs FEM 117 s = 7.66× |
| 多工况增量 | 3.0 s/工况 | 离线一次 → 在线 136× (vs FEM 117 s) |

**四线消融判别（回答"升维后为什么难"）**：

| 判别线 | 数据 | 结论 |
|---|---|---|
| 模态 1→9 | 21.1%→5.63% (均匀), 65%→6.51% (随机) | 模态形状是瓶颈 |
| 点数 9→15 | 5.47% vs 5.63% (打平) | 压缩比假说出局 |
| 网格加密三档 | 10.9→11.3→14.3% (无改善) | 离散误差非瓶颈 |
| 弯折度 | 246/369 界面 >30° 竖面 | 主因=模态-几何匹配 |

## 五、两个失败案例 (failed_experiments/)

### 5.1 Uniform-DDPNMT：每面 15 均匀采样点

**假设**：每面 1 个代表点代替 ~100 真实节点（压缩比）是误差主因 → 增加点数应改善。

**实验**：在均匀球上，每面 15 个均匀采样点 (Uniform-DDPNMT, 2,160 未知量) vs Affine 9 模 (1,296 未知量)。

**结果**：5.47% vs 5.63% — **打平**（+40% 未知量）。压缩比假说被排除，确认主因是模态形状而非代表点数量。

**文件**：`failed_experiments/uniform_point/run_uniform_point_ddpnmt_3d.py`

### 5.2 网格加密：三档无 h-收敛

**假设**：误差随网格加密应系统性下降。

**实验**：watershed 分区，mesh_size = 0.16/0.13/0.10（4,311/6,586/11,404 四面体），跑 Classic + Affine 全组合。

**结果**：Affine L² = 10.9% → 11.3% → 14.3% — **无系统趋势**。误差由画法模型误差主导，离散误差不是瓶颈。

**文件**：`failed_experiments/grid_refinement/outputs/`

## 六、环境要求

- **Python**：`D:\Miniconda3\envs\fenicsx\python.exe`（FEniCSx 环境）
- **PATH**：运行前必须 `export PATH="/d/Miniconda3/envs/fenicsx/Library/bin:$PATH"`
- **依赖**：dolfinx, gmsh, numpy, scipy, matplotlib, ufl, basix, mpi4py
- **LaTeX**：texlive 2025（如需重编译文档）

## 七、关键发现总结

1. **升维的几何断裂**：2D 的 1D 切口在 3D 无法分割体积 → Voronoi 胞 vs 净距盆地
2. **升维的代数断裂**：2D Classic 5.17% L² → 3D Classic 21–65% → Affine 6.5%
3. **双假说判别**：四线实验证实主因 = 模态形状与界面几何的匹配
4. **理论支柱**：Galerkin 最佳逼近 + 嵌套正交分解 + 体-迹误差双边分离
5. **自适应展望**：残差驱动界面富集 → 下一篇工作（reliability/efficiency + contraction theorem）

# W1n 法向线性三模态对照实验（2026-08-10 归档）

## 目的

机制分析：Affine-DDPNM-9（`span{1,s,t}⊗{n,t₁,t₂}`）相对 Classic-DDPNM-1
（`span{1·n}`）的误差改善，主要来自**法向线性模**（`span{1,s,t}⊗n`，即
W₁ₙ）还是**切向线性模**（`span{1,s,t}⊗{t₁,t₂}`），或两者的联合效应。

## 方法

新增 `NormalLinearFaceBasis`（`code/affine_face_basis.py`）：与
`AffineFaceBasis` 相同的代表实体构造，但每界面只保留
**方向 = 法向 × 标量 {P0, P1_s, P1_t}** = 3 个广义模（W₁ₙ 控制空间，
界面中心为 s=t=0 的代表点）。覆盖 `active_indices`（全激活）与
`global_keys`（3 键，与层级 level 无关）。

三个几何各跑一遍同网格多方法 benchmark：

| 项目 | 脚本 | 输出 |
|---|---|---|
| 均匀 27 球 | `run_affine_ddpnm_benchmark.py` | `results/affine_ddpnm_3d/` |
| 随机 27 球 | `run_random_benchmark.py` | `results/random_porous/` |
| 真实多孔（grid） | `run_benchmark.py --skip-hoddpnm` | `results/real_porous/` |

环境：`D:\Miniconda3\envs\fenicsx\python.exe` + 导出 PATH +
`PYTHONPATH=/d/hu/tongjiproj/20260727`（ddpnm_core / postprocess 依赖）。
16 GiB 内存纪律：**禁止并行跑两个 FE 任务**。

## 结果（速度相对 L2，同网格 vs 单片 FEM）

| 几何 | 未知量 Classic→W₁ₙ→Affine | Classic | W₁ₙ | Affine | 法向模贡献 |
|---|---:|---:|---:|---:|---:|
| 均匀 27 球（12203 胞 / 144 界面） | 144→432→1296 | 21.08% | **8.16%** | 5.63% | **83%** |
| 随机 27 球（15249 胞 / 114 界面） | 114→342→1026 | 65.32% | **30.75%** | 6.51% | **59%** |
| 真实多孔 grid（32611 胞 / 144 界面） | 144→432→1296 | 92.54% | **47.38%** | 19.98% | **62%** |

完整指标（broken-H1 / pL2 / 流量误差 / 成本 / 内存）见各 `results/*/*.csv` 与
`*.json`。随机球算例含精确 FE-trace Schur 正确性基线（vs 单片 FEM 1.05e-12）。

## 结论

1. **法向线性模是普适主导项**（占绝对改善 59~83%）：经典瓶颈是「界面上
   法向牵引只能取常数」（压力梯度的面内投影无法表达），不是代表实体个数。
2. **切向模贡献随几何不规则度上升**：均匀球 17% → 随机球 41% → 真实 38%；
   均匀球上切向模只做收尾（流量 2.4%→0.32%），随机/真实几何下近半改善靠它。
3. **性价比**：W₁ₙ 未知量为 Affine 1/3、离线时间 ≈ Classic、首解加速
   10~13×；精度预算 ~8%（规则）或 ~30%（随机）时 W₁ₙ 更优；要 0.3% 流量
   精度必须上 Affine 九模。

## 附加修复

`real_porous_benchmark_3d/run_benchmark.py` 的 Classic/W₁ₙ 响应库构建后
未释放内存（累积驻留 + 本机 commit 挤压导致 Affine 步骤 `np.column_stack`
1.52 MiB 分配失败），已加 `del lib; gc.collect()` 修复（`code/run_benchmark.py`）。

## Git 记录

- `a23db0b` feat: W1n normal-linear 3-mode control-space experiment（均匀球）
- `d1e04d5` feat: extend W1n ... to random and real porous geometries（三几何）

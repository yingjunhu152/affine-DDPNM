# 当前模型与离散规范（实现真源）

版本：2026-08-22。本文只描述 `schbench/` 当前实际执行的算法，是代码、实验报告和
后续论文公式的唯一实现真源。`docs/`、`pseudocode/` 和 `HANDOFF_SCH.md` 记录的是
重写前设计，其中的局部 CH trace Schur 与 Korteweg 力均未进入当前实现。

## 1. 模型范围

当前模型是 **viscosity-coupled Stokes--Cahn--Hilliard baseline**。相场通过孔平均
黏度影响 Stokes 速度，速度再对流相场。Stokes 动量方程没有
`-phi grad(mu_ch)` 或等价 Korteweg 应力，因此当前结果不是完整 Model H。

计算域按单元标签分成孔子域 `Omega_i`。Random-27 和反转 Bentheimer 均使用 27
个孔；孔间连通界面由网格拓扑识别。三个流动方法共用同一网格、物性和全局 CH
求解器，因而六臂比较主要隔离流动界面空间的误差。

## 2. 相场到黏度的闭合

对 P1 顶点相场 `phi_h`，代码先用四面体精确体积和单元四顶点均值计算孔平均

```text
phi_bar_i = (1 / |Omega_i|) integral_{Omega_i} phi_h dx.
```

仅在构造黏度时令 `phi_hat_i = clip(phi_bar_i, -1, 1)`，再取

```text
c_water_i = (1 + phi_hat_i) / 2,
mu_i = c_water_i mu_water + (1 - c_water_i) mu_oil.
```

默认 `mu_water=1`、`mu_oil=5`。这不会裁剪或修改输运相场本身；输出中的过冲和
下冲被完整保留。每个孔内的 `mu_i` 是常数，而不是点值 `mu(phi_h(x))`。

## 3. Stokes 流动层

### 3.1 当前连续算子与边界

当前代码对应

```text
-div(mu grad(u)) + grad(p) = 0,       in Omega,
 div(u) = 0,                          in Omega,
 u = 0,                               on solid walls,
(mu grad(u) - p I)n = -p_in n,        on inlet,
(mu grad(u) - p I)n = -p_out n,       on outlet.
```

默认 `p_in=1`、`p_out=0`。默认黏性算子是 `grad(u)` 形式；命令行可用
`--viscous-form symmetric` 切换到 `2 sym(grad(u))`。全局 FEM 与局部 DDPNM
响应算子总是使用同一选择，因此方法间比较在这一点上是内部一致的。旧生产基线
均为 `gradient`。

### 3.2 全局 FEM 弱式

Taylor--Hood 空间为 `u_h in [P2]^3`、`p_h in P1`。对测试函数 `(v_h,q_h)`，求

```text
integral mu grad(u_h):grad(v_h)
       - p_h div(v_h) - q_h div(u_h) - delta_p p_h q_h dx
= -integral_inlet p_in n.v_h ds - integral_outlet p_out n.v_h ds.
```

默认压力正则 `delta_p=1e-10`，因此离散连续性方程严格说是
`div(u_h)+delta_p p_h=0`。固壁速度强制为零；入口、出口是自然压力牵引边界。

### 3.3 Classic 与 Affine DDPNM

每个孔仍在局部 Taylor--Hood 空间中求 Stokes 响应，但全局只装配孔间界面牵引
系数。Classic 在每个连通界面片保留一个常数法向模态：

```text
t|_Gamma = lambda_Gamma n.
```

Affine 使用面内归一化坐标 `(s,t)` 和局部向量基 `(n,t1,t2)`，保留

```text
{1,s,t} x {n,t1,t2},
```

即每界面最多 9 个牵引模态。局部响应库在 `mu=1` 下离线构造；由于每孔黏度为
常数，速度/柔度响应按 `1/mu_i` 精确缩放。Bentheimer Affine 系统默认使用对角
缩放 POD，阈值 `1e-8`，删除数值零方向后求解。

## 4. Cahn--Hilliard 输运层

三个流动方法使用同一个全局 conforming `P1--P1` 混合空间 `(phi_h,mu_ch,h)`。
在时间步 `n -> n+1`，当前代码求

```text
(phi^{n+1}-phi^n)/dt + u.grad(phi^{n+1})
    = div(M grad(mu_ch^{n+1})),

mu_ch^{n+1} = sigma/epsilon ((phi^{n+1})^3-phi^n)
              - sigma epsilon Laplacian(phi^{n+1}).
```

第一式的对流项对新相场是隐式的，不是旧文档中的显式 `u.grad(phi^n)`。默认
`M=2e-4`、`sigma=2e-3`、`epsilon=1.5 h_mean`。入口强制 `phi=+1`；其余边界对
化学势扩散和相场梯度使用弱式自然零通量条件。非线性系统用 Newton、稀疏直接
线性求解和残差下降回溯；默认 Newton 容差 `1e-8`。

离散自由能诊断为

```text
E(phi) = integral sigma[(phi^2-1)^2/(4 epsilon)
                        + epsilon |grad(phi)|^2/2] dx.
```

入口 Dirichlet 会引入相质量，故总质量不应被误解为封闭域守恒常数。

默认初值仅在入口顶点取 `+1`，其余顶点取 `-1`。可选的入口相容光滑初值为
`phi(x)=phi_initial+(phi_inlet-phi_initial) exp(-x/ell)`，通过
`--initial-profile exponential --initial-transition-length ell` 启用；它改变初始
相含量和黏度场，不能与默认初值结果混作同一个物理算例。

## 5. Frozen 与 SFI

Frozen 在初始相场上计算一次孔黏度并只解一次流动；以后固定该速度推进 CH。

SFI 在每个时间步固定旧时间层 `phi^n`，执行：

```text
iterate -> pore-average viscosity -> Stokes velocity
        -> CH candidate at n+1 -> relaxed iterate.
```

第一轮是未阻尼预测器；从第二轮起用默认 `omega=0.65` 松弛。只有在第二轮及以后
满足 `max|iterate_new-iterate_old| <= 1e-4` 才收敛，最多 12 轮。收敛后保存最后
一个 CH candidate。当前反馈路径只有 `phi -> mu_i -> u -> phi`。

## 6. 输出、参照与已知限制

每臂输出逐步 Newton/SFI 次数、质量、自由能、相场极值和出口流量，以及最终
顶点相场、顶点速度、坐标、单元和孔标签。相场诊断还包含 P1 lumped-volume
越界体积分数、`phi<-1.001` 显著下冲体积分数、越界 L1 体积均值、最小值坐标/
孔号/距入口距离、入口邻域占比、最大单元顶点跳变及孔平均裁剪计数。对同一
coupling，Classic/Affine 的
相场与速度相对 L2 误差以 FEM 为参照；积分权重来自四面体体积。

当前已知限制：

- 没有 Korteweg 毛细体力，不能称为 full SCH/Model H；
- 连续 P1 Galerkin 加强入口跳变不是保界格式，已有 `phi<-1` 下冲；
- 默认初值仅入口顶点为 `+1`、其余域为 `-1`，形成网格尺度跳变；
- 默认黏性算子是 `grad(u)`，与常见的 `2D(u)` 形式需单独做敏感性对照；
- 压力零空间用 `1e-10` 质量项正则化，而非显式零均值约束。

以上限制必须在论文方法和结果解释中明确陈述；未经新增实现和验证，不得把旧设计
文档中的局部 CH-DDPNM、Korteweg 载荷或显式对流公式写成已完成方法。

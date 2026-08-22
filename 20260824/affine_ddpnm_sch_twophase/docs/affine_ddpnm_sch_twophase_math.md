# Stokes–Cahn–Hilliard 两相流的 DDPNM / Affine-DDPNM：连续模型、离散与守恒（数学文档 v1）

> **历史设计警告（2026-08-22）：** 本文包含未实现的 Korteweg 力和已经弃用的
> 局部 CH trace Schur 路线，不描述当前 `schbench/`。当前实现只以
> [`../MODEL_SPEC.md`](../MODEL_SPEC.md) 为准；此文件及其 PDF/TEX 仅供追溯。

> 项目：`D:\hu\tongjiproj\20260727\20260824\affine_ddpnm_sch_twophase`
> 日期：2026-08-22。前驱：单相 1/3/9 模态层级见 `20260810report/docs/affine_ddpnm_method_extension`；
> BL/Corey 两相版见 `20260824/docs/affine_ddpnm_twophase_method`（本文件继承其流动层机制，替换输运层为 Cahn–Hilliard）。
> 记号尽量与前驱一致：局部双线性、定向方向 $d_{i,k,\beta}$、九模态 $\mathcal M$、柔度 $G_i$、Schur 装配。

---

## 0. 定位与总览

本方法把已验证的单相 DDPNM / Affine-DDPNM 流动层扩展为**标准 Stokes–Cahn–Hilliard（SCH, Model H，匹配密度）两相流**：

- **流动层**（每步/每迭代）：变黏度 Stokes + 毛细体积力，DDPNM（$W_{0n}$，每界面 1 个牵引标量）或 Affine-DDPNM（$W_{1v}$，每界面 9 个仿射模态）求解。机制、重缩放命题、守恒通量提取与 BL 版**逐字相同**，仅新增一项毛细强迫右端。
- **输运层**（每步）：相场 $\varphi$ 与化学势 $w$ 满足 Cahn–Hilliard（四阶、扩散、守恒），**替换** BL 的 Buckley–Leverett/Corey 分数流。CH 也按 DDPNM 风格离散：孔内 P1 有限元 + 端口化学势迹模态耦合（classic 层级每界面 1 个常数迹，affine 层级每界面 3 个仿射迹），全局为一个 SPD 的 CH-Schur 系统。
- **benchmark**：单体 Taylor–Hood + P1–P1 的标准 FEM SCH 求解器（同一共形网格、同一时间格式），用于误差与成本对比。

与 BL 版的本质区别（先说清楚，防止误读）：

| | BL 版（已有） | SCH 版（本文件） |
|---|---|---|
| 输运变量 | 每孔 1 个饱和度 $S_a$（P0） | 相场 $\varphi$、化学势 $w$（每孔 P1 场） |
| 输运方程 | $\varphi\partial_t S+\nabla\cdot(f_w(S)u)=0$，upwind 分数流 | $\partial_t\varphi+\nabla\cdot(u\varphi-M_c\nabla w)=0$，$w=\lambda(\epsilon^{-1}f(\varphi)-\epsilon\Delta\varphi)$ |
| 毛细 | 关闭（$p_e=0$） | 内生（混合能 $\lambda,\epsilon$；锐界面极限 $\sigma_s=\tfrac{2\sqrt2}{3}\lambda\epsilon$） |
| 孔内黏度 | $\mu_a=1/\lambda_t(S_a)$（Corey） | $\mu_a=\mu(\bar\varphi_a)$（混合律，Assumption S2） |
| 流动层机制 | 相同 | 相同（+毛细强迫列） |
| 守恒 | 相质量账本（29） | 相质量账本（S-CH 台账）+ 能量耗散律 |

---

## 1. 连续模型（Model H，匹配密度，无重力）

### 1.1 未知量与自由能

$\bm u$（总速度）、$p$（压力）、$\varphi$（相场，$\varphi=+1$ 润湿相/水、$\varphi=-1$ 非润湿相/油）、$w$（化学势）。混合能

$$
E(\varphi)=\int_\Omega \lambda\Big(\frac{\epsilon}{2}|\nabla\varphi|^2+\frac{1}{\epsilon}F(\varphi)\Big)\,dx,
\qquad F(\varphi)=\tfrac14(\varphi^2-1)^2,\quad f=F'=\varphi^3-\varphi. \tag{S1}
$$

### 1.2 强形式

$$
-\nabla\cdot\big(2\mu(\varphi)D(\bm u)\big)+\nabla p \;=\; w\,\nabla\varphi,
\qquad \nabla\cdot\bm u=0 \qquad\text{in }\Omega, \tag{S2}
$$

$$
\partial_t\varphi+\nabla\cdot\big(\bm u\,\varphi-M_c\nabla w\big)=0 \qquad\text{in }\Omega, \tag{S3}
$$

$$
w=\lambda\Big(\epsilon^{-1}f(\varphi)-\epsilon\,\Delta\varphi\Big) \qquad\text{in }\Omega. \tag{S4}
$$

$D(\bm u)=\tfrac12(\nabla\bm u+\nabla\bm u^{\!\top})$；混合黏度 $\mu(\varphi)=\tfrac{1+\varphi}{2}\mu_w+\tfrac{1-\varphi}{2}\mu_o$（或对数混合；孔级取值见 Assumption S2）；$M_c>0$ 迁移率。

**等价形式（Remark S1.1）.** 毛细力以体积力出现：$w\nabla\varphi=-\nabla\cdot\mathbb K$，Korteweg 应力
$\mathbb K=\lambda\big(\epsilon\nabla\varphi\otimes\nabla\varphi-(\tfrac{\epsilon}{2}|\nabla\varphi|^2+\epsilon^{-1}F(\varphi))I\big)$，
差为一个可吸入 $p$ 的梯度项。因此**入口/出口仍是水动伪牵引** $\sigma_h\bm n=-p_b\bm n$（$\sigma_h=-pI+2\mu D$，切向为零），与前驱单相/BL 版边界机制完全一致；Korteweg-应力形式只在 $\partial_n\varphi\neq0$ 的边界上与体积力形式差一个切向牵引重整，本协议入口取 $\varphi\equiv+1$（$\nabla\varphi=0$），差异可忽略。

### 1.3 边界条件

$$
\bm u=0,\quad \partial_n\varphi=0,\quad \partial_n w=0 \qquad \text{on }\Gamma_{\mathrm{wall}}, \tag{S5}
$$

$$
\sigma_h\bm n=-p_{\mathrm{in}}\bm n,\quad \varphi=+1 \quad\text{on }\Gamma_{\mathrm{in}};
\qquad
\sigma_h\bm n=-p_{\mathrm{out}}\bm n,\quad \partial_n\varphi=0\ \text{（对流外流）} \quad\text{on }\Gamma_{\mathrm{out}}. \tag{S6}
$$

壁面 $\partial_n\varphi=0$ 即中性润湿（90° 接触角）；$\partial_nw=0$ 为壁面零相质量通量。初始 $\varphi^0\equiv-1$（油充满），入口注入水。

### 1.4 守恒律与能量律（连续层面）

- **总质量**：$\nabla\cdot\bm u=0$ ⇒ $\int_{\partial\Omega}\bm u\cdot\bm n\,dS=0$。
- **相质量**：对 (S3) 积分，壁面零通量 ⇒ $\frac{d}{dt}\int_\Omega\varphi\,dx=-\int_{\Gamma_{\mathrm{in}}\cup\Gamma_{\mathrm{out}}}(\bm u\varphi-M_c\partial_nw)\cdot\bm n\,dS$（仅边界交换）。
- **能量耗散（热力学一致性）**：

$$
\frac{dE}{dt}=-\int_\Omega 2\mu|D(\bm u)|^2\,dx-M_c\int_\Omega|\nabla w|^2\,dx\;\le\;0. \tag{S7}
$$

证明两行：$\tfrac{dE}{dt}=\int w\,\partial_t\varphi$（用 (S4) 分部）；代入 (S3) 得 $\tfrac{dE}{dt}=-\int w\,\bm u\cdot\nabla\varphi-M_c\|\nabla w\|^2$；(S2) 用 $\bm u$ 检验并 $\nabla\cdot\bm u=0$ 给 $\int w\,\bm u\cdot\nabla\varphi=2\mu\|D\|^2$。两式相消即 (S7)。

### 1.5 锐界面极限与无量纲参数

- 平衡界面剖面 $\varphi=\tanh(z/(\sqrt2\,\epsilon))$；界面张力与 Young–Laplace：

$$
\sigma_s=\frac{2\sqrt2}{3}\lambda\,\epsilon,
\qquad [p]=\sigma_s\kappa. \tag{S8}
$$

- 参数组（$L$ 特征长度、$U$ 特征速度）：Cahn 数 $\mathrm{Cn}=\epsilon/L$、毛细数 $\mathrm{Ca}=\mu_u U/\sigma_s$、迁移 Péclet $\mathrm{Pe}=U L/(M_c\lambda/\epsilon)$。实验报告 $(\epsilon,\lambda,M_c,\mu_w/\mu_o,\Delta p,\mathrm{Ca})$。

---

## 2. 孔级闭合假设（对齐 BL 版 Assumption 2）

**Assumption S2（孔内常物性闭合）.** 每孔一个相自由度用于黏度取值：$\bar\varphi_a=\int_{\Omega_a}\varphi\,dx/V_a$，且

$$
\mu_a=\mu(\bar\varphi_a)\quad\text{在 }\Omega_a\text{ 内为常数}. \tag{S9}
$$

作用：(i) 使重缩放命题（Prop S3.1）精确成立；(ii) FEM 主协议采用同一闭合以隔离界面建模误差（§7）。
声明：这是**建模近似**（数值尺度由实验标定，量级预期与 BL 版反馈项同阶），不是定理；逐点 $\mu(\varphi)$ 的 FEM 作绝对参考另行报告。

---

## 3. 流动层：DDPNM / Affine-DDPNM + 毛细强迫

### 3.1 界面模态（与单相版逐字相同，摘录）

孔对共享平面界面 $\gamma_k$，全局正交标架 $\{n_k,t_{k,1},t_{k,2}\}$，定向内牵引方向
$d_{i,k,n}=n_{i,k}$、$d_{i,k,t_\nu}=-\sigma_{i,k}t_{k,\nu}$、两侧反对称 $d_{j,k,\beta}=-d_{i,k,\beta}$；
归一化面内坐标 $s_k,t_k$，标量形 $\varphi_0=1,\varphi_s=s,\varphi_t=t$，九模态 $\mathcal M=\{0,s,t\}\times\{n,t_1,t_2\}$；
四空间 $W_{0n}(\dim1)/W_{0v}(3)/W_{1n}(3)/W_{1v}(9)$，代码列序置换 $p=(0,3,6,1,4,7,2,5,8)$。
本实验取两端：**classic DDPNM $=W_{0n}$，affine DDPNM $=W_{1v}$**。

### 3.2 牵引 Ansatz 与荷载（水动应力）

$$
-\sigma_h(\bm u_i,p_i)\,n_{i,k}=\sum_{(\alpha,\beta)\in\mathcal M_X}\lambda_{k,\alpha,\beta}\,\varphi_\alpha\,d_{i,k,\beta},
\qquad
\big[f_{i,k}^{(\alpha,\beta)}\big]_j=-\int_{\gamma_k}\varphi_\alpha\,d_{i,k,\beta}\cdot\phi_j^{(i)}\,dS. \tag{S10}
$$

荷载不含 $\mu$（重缩放前提，同 BL）。

### 3.3 毛细强迫（新增）

动量右端的体积力项在孔 $i$ 上离散为一条额外荷载列（滞后显式或 Picard 当前迭代值 $\varphi^\*,w^\*$）：

$$
\big[f_i^{\mathrm{cap}}\big]_j=\int_{\Omega_i} w^\*\,\nabla\varphi^\*\cdot\phi_j^{(i)}\,dx. \tag{S11}
$$

### 3.4 重缩放命题（推广到任意荷载）

**Prop S3.1.** 设 Assumption S2（$A_i(\mu_a)=\mu_a A_i(1)$）。对**任意**速度荷载 $f$（界面牵引模态或体积力 (S11)），

$$
P(\mu_a)=P(1)\ \text{（压力响应不变）},\qquad U(\mu_a)=\tfrac{1}{\mu_a}U(1),\qquad G_i(\mu_a)=\tfrac{1}{\mu_a}G_i(1). \tag{S12}
$$

证明与 BL 命题 4.1 逐字相同（仅用鞍点结构与 $A\propto\mu$ 齐次性，与荷载形式无关）。离线库仍在参考黏度 $\mu_{\mathrm{ref}}=1$ 建一次；毛细响应列同样按 $1/\mu_a$ 缩放；**每步/每迭代仅一次额外局部回代求解**（复用存储的参考黏度分解），无新分解。

### 3.5 全局 Schur（含毛细强迫）

局部解 $\bm U_i=H_i\big(L_i\bm\lambda_i+f_i^{\mathrm{cap}}\big)$；界面矩 $W(U_i)=-G_i\lambda_i-\underbrace{L_i^{\!T}H_if_i^{\mathrm{cap}}}_{=:\ \bm c_i}$；矩连续 $W_i+W_j=0$ 装配：

$$
\mathbf S(\bm\mu)\,\bm\lambda=\mathbf F(\bm\mu)-\sum_i (R_i^u)^{\!T}\bm c_i. \tag{S13}
$$

**矩阵**与单相/BL 完全相同（$\mathbf S=\sum_i(R_i^u)^T\tfrac{\mu_{\mathrm{ref}}}{\mu_i}G_i^{uu}R_i^u$），SPD 条件（局部稳定、局部一维核 $\mathrm{span}\{1_{0n}\}$、图锚定）不变；**毛细只进右端**，不影响对称性与正定性。

### 3.6 流动层守恒（继承，不变）

- 每孔质量平衡 $\sum_{\gamma}W_{i,\gamma}^{(0,n)}=0$（$B_iU_i=0$ + 常压测试；毛细体积力不进入不可压约束）；
- 界面净通量单值 $Q_\gamma=\tfrac12(m_b-m_a)$、边界外向通量 $Q_a^{\mathrm{bnd}}=-m_a^{\mathrm{bnd}}$（同 BL 式 (24)，符号协议含 2026-08-21 修正）；
- 全局 $\int_{\partial\Omega}\bm u\cdot\bm n=0$。各模态层级均精确（保留 $(0,n)$）。

### 3.7 与 BL 的速率控制差别（Remark S3.2）

BL 的线性重标 $\lambda\mapsto\alpha\lambda$ 在含毛细体积力时不再保持物理（除非同时缩放 $f^{\mathrm{cap}}$，但那会篡改毛细物理）。协议改为**压力驱动**：固定 $\Delta p=p_{\mathrm{in}}-p_{\mathrm{out}}$，报告 $Q(t)$；如需恒速驱替可加 $\alpha$-Picard 外环（数值决策点）。

---

## 4. 输运层：Cahn–Hilliard 的 DDPNM 式离散

### 4.1 空间与端口迹模态

$\varphi,w$ 取每孔网格上的 $P_1(\Omega_i)$（孔网格为整体共形网格的逐胞分解，界面两侧节点匹配）。**端口化学势迹**用标量仿射层级（CH 是标量场 → 层级为 1/3，对照流动层 1/9）：

$$
W^{\mathrm{ch}}_0(\gamma_k)=\mathrm{span}\{1\}\ (\text{classic，每界面 }1),\qquad
W^{\mathrm{ch}}_1(\gamma_k)=\mathbb P_1(\gamma_k)=\mathrm{span}\{1,s,t\}\ (\text{affine，每界面 }3). \tag{S14}
$$

界面未知量：$\eta_{k,\alpha}$（$\alpha\in\{0\}$ 或 $\{0,s,t\}$），孔 $i$ 侧端口迹 $w|_{\gamma_k}=\sum_\alpha\eta_{k,\alpha}\varphi_\alpha$。**对偶结构完全镜像流动层**：流动层未知量=界面**牵引**（Neumann 数据）、响应=速度矩、方程=矩连续；CH 层未知量=界面**化学势迹**（Dirichlet 数据）、响应=相通量矩、方程=通量矩连续。

### 4.2 时间离散（孔内，凸分裂 + 显式对流 + 隐式扩散）

$$
w^{\,n+1}=\lambda\Big(\epsilon^{-1}\big[(\varphi^{\,n+1})^3-\varphi^{\,n}\big]-\epsilon\,\Delta\varphi^{\,n+1}\Big), \tag{S15}
$$

$$
\frac{\varphi^{\,n+1}-\varphi^{\,n}}{\Delta t}+\bm u^{\*}\cdot\nabla\varphi^{\,n}=\nabla\cdot\big(M_c\nabla w^{\,n+1}\big). \tag{S16}
$$

凸分裂 $f=f_c-f_e$，$f_c'=\varphi^3$、$f_e'=\varphi$（$F_c=\varphi^4/4$，$F_e=\varphi^2/2$），无条件能量稳定（孔内）；对流显式（$\bm u^{\*}$ = 上一步或 Picard 当前的 DDPNM 重构速度场），扩散隐性（无 $M_c,\epsilon$ 时间步约束）。入口 $\varphi^{n+1}=+1$ 强约束。

### 4.3 局部 CH 步（孔 $i$，端口 $w$-Dirichlet）

记 $w=L\bm\eta+w_0$（$L\bm\eta$ 端口迹提升，$w_0$ 端口零迹）。求 $(\varphi_i^{n+1},w_{0,i}^{n+1})\in P_1\times P_{1,0}$：

$$
\big(\varphi^{\,n+1},v\big)+\Delta t\,M_c\big(\nabla w^{\,n+1},\nabla v\big)_{\Omega_i}
=\big(\varphi^{\,n},v\big)-\Delta t\big(\bm u^{\*}\cdot\nabla\varphi^{\,n},v\big)_{\Omega_i}
\quad\forall v\in P_1(\Omega_i), \tag{S17}
$$

$$
\big(w^{\,n+1},q\big)_{\Omega_i}=\lambda\Big[\epsilon^{-1}\big(((\varphi^{\,n+1})^3),q\big)-\epsilon^{-1}(\varphi^{\,n},q)+\epsilon\big(\nabla\varphi^{\,n+1},\nabla q\big)_{\Omega_i}\Big]
\quad\forall q\in P_{1,0}(\Omega_i). \tag{S18}
$$

（$w$-方程只在内部检验；端口迹自由度由 $\bm\eta$ 给定。$\varphi^3$ 项使 (S17)–(S18) 非线性 → 每孔小规模 Newton。）**局部适定**：质量项 $(\varphi^{n+1},v)$ 消除纯 Neumann 核，无需锚定（见 Prop S4.1）。

### 4.4 端口相通量矩（响应泛函）

局部解出后，对每端口 $k$、每保留迹模态 $\alpha$ 计算**总相通量矩**（精确求积，两侧共享同一平面界面）：

$$
J_{i,k}^{(\alpha)}=\int_{\gamma_k}\varphi_\alpha\,\big(\bm u_i^{\*}\varphi^{\,n}-M_c\nabla w^{\,n+1}\big)\cdot n_{i,k}\;dS. \tag{S19}
$$

（等价地由 (S17) 对端口支集检验函数的残差——反应通量——计算。$\alpha=0$ 是界面相质量流量。）

### 4.5 全局 CH-Schur：通量矩连续

界面方程（对每内部界面 $k$、每 $\alpha$）：

$$
J_{i_k,k}^{(\alpha)}+J_{j_k,k}^{(\alpha)}=0. \tag{S20}
$$

对 $\bm\eta$ 线性化（Picard）：$J_i\approx J_i(\bm\eta^{\mathrm{p}})+G_i^{\mathrm{ch}}\big(\bm\eta-\bm\eta^{\mathrm{p}}\big)$，其中 $G_i^{\mathrm{ch}}$ 为**局部 CH-DtN**（端口迹单位扰动 → 通量矩响应，由 (S17)–(S18) 的切线问题组装；切线问题对称：质量 + 刚度 + $3(\varphi^{\mathrm p})^2$ 乘子均为自伴）。装配：

$$
\mathbf S^{\mathrm{ch}}\,\bm\eta=\mathbf F^{\mathrm{ch}},
\qquad
\mathbf S^{\mathrm{ch}}=\sum_i (R_i^{\mathrm{ch}})^TG_i^{\mathrm{ch}}R_i^{\mathrm{ch}}. \tag{S21}
$$

**Prop S4.1（局部负定 / 全局 SPD，条件性）.** 由能量恒等式，$\bm\eta_i^{\,T}G_i^{\mathrm{ch}}\bm\eta_i=-\Delta t^{-1}\big(\delta\varphi,\delta\varphi\big)_{\Omega_i}-M_c\big(\nabla\delta w,\nabla\delta w\big)_{\Omega_i}\le0$ 型二次型（沿 (S17)–(S18) 切线的耗散）；且 $\varphi$ 的质量项使常数迹扰动产生非零响应（不同于纯 Laplace DtN 的一维核），故 $G_i^{\mathrm{ch}}$ 负定、$\mathbf S^{\mathrm{ch}}$ SPD，无需图锚定。实现以最小特征值与对称缺陷作数值诊断（对齐前驱"实用代数诊断"）。

### 4.6 相质量守恒账本（精确）

(S17) 取 $v\equiv1$：$\big(\varphi^{n+1}-\varphi^{n},1\big)_{\Omega_i}=-\Delta t\sum_k J_{i,k}^{(0)}$（壁面零通量）。内界面 (S20) 的 $\alpha=0$ 行使 $J^{(0)}$ 单值成对抵消，求和得全局账本：

$$
r^n=\sum_a\big(\varphi^{n+1}-\varphi^{n},1\big)_{\Omega_a}+\Delta t\sum_{a}J_a^{\mathrm{bnd}}=0
\quad(\text{未截断时逐位恒零}). \tag{S22}
$$

CH 隐式扩散无需 BL 的 clip（凸分裂控制 $|\varphi|\lesssim1$，允许 $O(\Delta t)$ 过冲）。classic 与 affine 两层级都保留 $\alpha=0$ 行 ⇒ **相质量守恒不因迹模态富集而破坏**（镜像流动层定理）。

### 4.7 能量诊断（非定理，报告项）

$$
E^{n}=\sum_i\int_{\Omega_i}\lambda\Big(\tfrac{\epsilon}{2}|\nabla\varphi^{n}|^2+\epsilon^{-1}F(\varphi^{n})\Big)dx\ \text{应单调下降（孔内无条件稳定；界面耦合与显式毛细引入 }O(\Delta t)\text{ 耦合误差）}. \tag{S23}
$$

---

## 5. 耦合算法

**Algorithm S1（frozen-SCH，单向）**：初始流度 $\mu_a(\bar\varphi^0)$ 下解一次流动 (S13)（含初始毛细列）→ 冻结 $\bm u^{\*}$、$\{Q_\gamma\}$ → 每步只推进 CH：(S17)–(S18) 局部步 + (S21) CH-Schur（对 $\eta$ 的 Picard 内环可选）。特点：每步 0 次流动求解、1 组局部 CH 回代 + 1 个小 Schur。

**Algorithm S2（SFI-SCH，双向阻尼 Picard）**：每步 $k=1,\dots,K_{\max}$：
(i) $\mu_a\leftarrow\mu(\bar\varphi_a^{(k)})$；(ii) 重缩放库 (S12)、装配方程 (S13)（含 $\bm c_i(\varphi^{(k)},w^{(k)})$）解 $\bm\lambda$；(iii) (S24-式通量提取) 取 $Q_\gamma$；(iv) CH 局部步 + (S21) 解 $\bm\eta$、更新 $\varphi^{(k+1)}=0.65\,\Phi+0.35\,\varphi^{(k)}$；(v) $\delta=\|\Phi-\varphi^{(k)}\|_\infty\le10^{-8}$ 收敛。特点：每迭代 = 代数重缩放 + 1 次流动 Schur + 局部 CH 回代 + 1 次 CH-Schur；**无新 Stokes 分解**。

**时间步约束**：对流 CFL $\Delta t\le\mathrm{cfl}\cdot\min_a V_a/\sum_{i\ni a}\max(\pm Q_i,0)$（同 BL `cfl_time_step`）；扩散隐性无 $M_c$ 约束；显式毛细耦合的稳定性由阻尼 Picard 吸收。

**指标**：能量衰减 (S23)、相质量账本 (S22)、$\int\varphi$ 回收曲线、出口突破曲线、界面位置、与 FEM 的差。

---

## 6. FEM benchmark

**单体标准 FEM**：全局共形网格（= 孔网格之并），$[\mathbb P_2]^3\times\mathbb P_1$（$\bm u,p$，Taylor–Hood）+ $\mathbb P_1\times\mathbb P_1$（$\varphi,w$）；时间格式与 DDPNM 臂**逐字相同**（(S15)–(S16) 凸分裂 + 显式对流 + 隐性扩散）；无界面闭合。每步解一个全局线性系统（凸分裂下 $f_c'$ 非线性 → 单体 Newton 或一阶线性化）。

**公平协议**：同一网格、同一 $\Delta t$/CFL、同一凸分裂格式 ⇒ DDPNM 臂与 FEM 臂差异**只来自**：(i) 流动层界面牵引模态限制（$W_{0n}$/$W_{1v}$）；(ii) CH 层端口迹模态限制（$W^{\mathrm{ch}}_0/W^{\mathrm{ch}}_1$）；(iii) 孔内常黏度闭合（主协议 FEM 同闭合，消除该项；逐点 $\mu(\varphi)$ FEM 为绝对参考另跑）。

**输出**：误差（$\|\varphi-\varphi_{\mathrm{FEM}}\|_{L^2}$、$\|\bm u-\bm u_{\mathrm{FEM}}\|_{L^2}$、$p$、出口通量/突破曲线）+ 成本（墙钟、峰值内存、每步求解次数、离线/在线分解）+ 场图与误差云图（风格仿 `01_slice_error_fields.png`）。

---

## 7. 定理级 / 假设级 / 经验级的划分（防止过度声明）

- **有限维代数结论（定理级）**：流动层 SPD、唯一解、精确质量守恒（继承单相理论；毛细列不触及矩阵）；CH 层局部负定、CH-Schur SPD（条件性，附数值诊断）、相质量账本 (S22) 逐位为零。
- **闭合假设（Assumption S2）**：孔内常黏度 —— 数值尺度待标定；逐点黏度 FEM 参考给出其真实影响。
- **时间离散**：孔内凸分裂无条件稳定；显式对流 CFL；显式毛细耦合 $O(\Delta t)$——非能量稳定定理，能量衰减作为**诊断**报告 (S23)。
- **经验量**：相对单体 FEM 的 $L^2$ 误差沿层级不必单调（BL 版实测同款现象）；$\epsilon$-网格分辨敏感性（要求 $h\lesssim\epsilon$）。

## 8. 开工前待确认决策

1. 参数集：$\epsilon$（建议 $4\text{–}6\,h_{\mathrm{avg}}$）、$\lambda$（由目标 $\mathrm{Ca}$ 定）、$M_c$（$\mathrm{Pe}\sim10\text{–}10^2$）、$\mu_w/\mu_o$（沿用 1/5？）、$\Delta p$、步数/PVI。
2. 方法配对：classic =（流动 $W_{0n}$ + CH $W^{\mathrm{ch}}_0$），affine =（流动 $W_{1v}$ + CH $W^{\mathrm{ch}}_1$）——两层同时升降级（建议），或固定一层只动另一层（消融）。
3. 主报告算法：frozen 与 SFI 都跑，主表 SFI（建议）。
4. CH 对流是否加 SUPG/限制器（P1+显式若有振荡再加）。
5. 几何：随机 27 球 + 反转 Bentheimer（复用 `random_porous.py` / `prepare_bentheimer_inverted_core.py` 管线）。

## 参考文献

1. S. Sun, Z. Wang, L. Zhang, J. Zhao. A domain decomposition approach to pore-network modeling of porous media flow. arXiv:2510.13429v2, 2026.（经典 DDPNM）
2. `20260810report/docs/affine_ddpnm_{math,method}_extension.{md,tex,pdf}`：单相 1/3/9 模态公式链与定理。
3. `20260824/docs/affine_ddpnm_twophase_{math,method}.md`：BL/Corey 两相版（本文件流动层机制来源）。
4. J. Shen, X. Yang. Energy stable schemes for Cahn–Hilliard phase-field model of two-phase incompressible flows. J. Comput. Phys. 228 (2009).（凸分裂时间格式）
5. H. Abels, H. Garcke, G. Grün. Thermodynamically consistent, frame indifferent diffuse interface models for incompressible two-phase flows with different densities. Math. Models Methods Appl. Sci. 22 (2012).（Model H）

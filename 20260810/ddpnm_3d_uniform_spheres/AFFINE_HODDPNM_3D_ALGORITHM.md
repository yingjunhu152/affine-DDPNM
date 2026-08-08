# 算法：三维完整向量仿射界面 HODDPNM-P1(9)

**输入：** 多孔介质区域 \(\Omega\)，最大球子区域 \(\{\Omega_i\}\)，严格孔喉鞍点界面 \(\{\Gamma_{ij}\}\)，局部 Taylor--Hood 网格，黏度 \(\mu\)，入口/出口压力。

**输出：** 各界面的九个牵引系数、各子区域速度 \(\boldsymbol u_i\) 与压力 \(p_i\)。

1. 对每个界面 \(\Gamma_{ij}\) 计算中心 \(\boldsymbol c_{ij}\)、单位法向 \(\boldsymbol n_{ij}\) 以及一致定向的两个单位切向量 \(\boldsymbol t_{1,ij},\boldsymbol t_{2,ij}\)。
2. 定义无量纲面内坐标
   \[
   s=\frac{(\boldsymbol x-\boldsymbol c_{ij})\cdot\boldsymbol t_{1,ij}}{h_{1,ij}},
   \qquad
   t=\frac{(\boldsymbol x-\boldsymbol c_{ij})\cdot\boldsymbol t_{2,ij}}{h_{2,ij}}.
   \]
3. 在每个界面上同时采用完整向量仿射牵引空间
   \[
   \mathcal T_{ij}^{(9)}
   =\operatorname{span}\{\boldsymbol n,s\boldsymbol n,t\boldsymbol n,
   \boldsymbol t_1,s\boldsymbol t_1,t\boldsymbol t_1,
   \boldsymbol t_2,s\boldsymbol t_2,t\boldsymbol t_2\}.
   \]
   其中包含三个常数向量模式、两个线性法向模式和四个线性切向模式，共九个自由度。
4. 对每个子区域只分解一次局部离散 Stokes 矩阵；依次施加与该子区域相邻的全部界面基牵引，保存九类局部速度—压力响应。
5. 将局部有限元内部自由度静态凝聚，组装九模式界面 Schur 系统
   \[
   S_9\boldsymbol\lambda_9=\boldsymbol b_9.
   \]
6. 求解 \(\boldsymbol\lambda_9\)。该方程在每个共享界面上同时强制速度的常数矩以及两个面内一阶矩连续；三个向量方向均参与耦合。
7. 用求得的九模式系数线性组合局部响应，重构全部 \((\boldsymbol u_i,p_i)\)，并检查界面矩残差、入口—出口质量平衡和全局线性残差。

> 注：HODDPNM-P1(9) 是九维低阶界面迹空间上的约化 Schur 补；它不等于保留全部有限元界面节点自由度的精确 FEM 迹 Schur 补。

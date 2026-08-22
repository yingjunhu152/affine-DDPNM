# 非 Korteweg 方法学检查（2026-08-22）

所有检查保持当前黏度耦合 Stokes--CH 模型，不加入 Korteweg 力。

## 黏性算子

对照采用 frozen 初始黏度；每个算子内部的 FEM 与 Affine 使用完全相同弱式。
Random-27 的 gradient 行取已验证生产基线，Bentheimer 的 gradient 行取 POD
tol=1e-8 基线；frozen 流场不依赖后续 CH 时间步长。

| geometry | form | FEM flux | FEM velocity change | Affine velocity error | Affine flux error |
|---|---|---:|---:|---:|---:|
| random27 | gradient | 0.0015062177 | 0.000% | 5.553% | 2.816% |
| random27 | symmetric | 0.0015066102 | 5.723% | 9.490% | 6.291% |
| bentheimer | gradient | 0.0002618647 | 0.000% | 18.742% | 11.160% |
| bentheimer | symmetric | 0.00025332426 | 5.887% | 17.612% | 11.325% |

`2D(u)` 路径在两个几何的 FEM 与 Affine 上均成功装配和收敛。Random-27
的 Affine 速度误差由约 5.55% 增至 9.49%，说明该几何对自然牵引下的算子
选择较敏感；Bentheimer 则由约 18.74% 降至 17.61%，结论基本不变。旧基线
仍以 gradient 为默认，论文必须准确写出所用弱式；是否改用 symmetric 应由目标
连续模型决定，而不是按误差较小者事后选择。

## 入口相容光滑初值

对照均为 FEM-frozen、dt=0.25、t=1。光滑初值为
`phi(x)=-1+2 exp(-x/0.05)`，在入口严格等于 +1。显著下冲定义为
`phi<-1.001`，体积分数采用 P1 lumped-volume 权重。

| geometry | profile | final phi_min | significant volume | L1 violation | flux |
|---|---|---:|---:|---:|---:|
| random27 | discontinuous | -1.4531055 | 13.430% | 6.752e-03 | 0.0015062177 |
| random27 | exponential | -1.0009746 | 0.000% | 5.616e-07 | 0.0015327999 |
| bentheimer | discontinuous | -1.3973067 | 10.123% | 5.477e-03 | 0.0002618647 |
| bentheimer | exponential | -1.0000000 | 0.000% | 0.000e+00 | 0.00026895477 |

光滑初值几乎消除了两个几何的下冲，强烈支持“入口网格尺度跳变是主要诱因”
的判断。但它同时改变初始相含量、孔平均黏度和 frozen 流量，因此不能把新结果
与旧生产基线混表。正式论文应先给出物理上希望表达的初始润湿/注入历史，再固定
初值；若采用光滑初值，应报告 0.05 并补做过渡长度敏感性。

机器可读数据见 `viscous_form_summary.csv` 与 `initial_profile_summary.csv`。

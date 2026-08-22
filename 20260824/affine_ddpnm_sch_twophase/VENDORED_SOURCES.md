# 内置源码来源

为使本目录离开原研究树后仍能运行，2026-08-22 从相邻的
`../affine_ddpnm_twophase` 内置了当前六臂实验实际依赖的最小源码集合：

- `ddpnm_core/`：局部 Stokes 响应、界面 Schur 装配及 FEM 工具；
- `postprocess/fields.py`：混合有限元场的 P1 顶点重构；
- `ddpnm3d/basis_3d.py` 与 `affine_face_basis.py`：Classic/Affine 界面基；
- `random_porous.py` 与 `digital_core_partition.py`：两个几何的构造/读取；
- `data/bentheimer_voxel_pore_mesh.msh`：反转 Bentheimer 固定输入网格。

这些文件属于同一 `affine-DDPNM` 研究代码库，不是外部第三方发布包。内置时保持
源码内容不变；后续针对本基准的改动由本目录 Git 历史追踪。Bentheimer 网格的
SHA-256 为：

```text
B5ABCD0ED294D6615EC53102B3D6A3055348F871B3069CA7CCDCBAEFC427D765
```

运行时不再修改 `sys.path`，也不再读取任何相邻项目目录。

# "平凡分区方法"可行性检查：Delaunay 四面体剖分（最终版）

> 对应 handoff §5.4。本文档供论文第 2 章"机械搬用二维逻辑"叙事引用。
> v2 细化版：2026-08-05，几何正确的 4th 球侵入筛选。

---

## 方法

规则阵列的"过球心平面族"在随机几何的推广 = 球心点集的 **Delaunay 四面体剖分**：

- **胞** = 一个 Delaunay 四面体（4 顶点全是球心）
- **界面** = 相邻两胞共享的三角形面（3 端点都是球心 ✓、平面穿过这 3 个球心 ✓）
- **正确性筛选**：对每面采样 25×25 网格（~325 点），计算对所有 27 球的 clearance；
  找正 clearance 区域的连通分量；4th 球只在交圆与三角形真正重叠时算 intruder。

## 基本数字

| 指标 | 数值 |
|---|---|
| 球心数 | 27 |
| Delaunay 四面体 | 84 |
| 共享界面（real–real） | 151 |
| 带 8 角 dummy 的四面体 | 131 |
| real–real 界面（dummy 版） | 111 |
| real–dummy 边界界面 | 46 |

## 细化的几何正确性筛选

| 类别 | 数量 | 占比 |
|---|---|---|
| **clean**（viable，无 intruder） | **150** | 99.3% |
| fragmented（>1 连通分量） | 1 | 0.7% |
| blocked（<1% 开放面积） | 0 | 0% |
| **4th 球 intruder** | **0** | **0%** |

> ⚠️ **v1 粗口径"151/151 穿面"是完全错误的**：v1 只判了球心到平面距离 < 半径，
> 但交圆离三角形很远时不影响流体通道。几何正确的交集检查后，**零个面被 4th 球侵入**。

## 唯一 fragmented 面

| 属性 | 值 |
|---|---|
| 面 ID | 96 |
| 顶点球 | 5, 10, 20 |
| 涉及球 | {4, 5, 10, 16, 20} |
| 开放面积比 | 39.9% |
| 最大 clearance | 0.0591 |
| 最小正 clearance（constriction） | 0.0061 |
| 连通分量数 | 2 |

> 两个正 clearance 区域被 3 个顶点球圆盘夹出的窄桥隔开。对孔网模型而言，
> 一个面有两个独立通道是物理上合理的（等效于两个并联喉道），不是致命缺陷。

## Clearance 分布

| 指标 | Delaunay（球心平面） | Voronoi（垂直平分面） | 比值 |
|---|---|---|---|
| 鞍点净距（max clearance）中位 | **0.1571** | 0.0073 | **21.5×** |
| constriction（min positive clearance）中位 | **0.00048** | — | — |
| 正 clearance 面积比中位 | 76.4% | — | — |

### 解读

- **鞍点净距 21× 差距**：Delaunay 界面穿过球心（孔体内部，宽敞），Voronoi 界面
  在两球正中间（孔喉，最窄）。这是"界面被钉在球心晶格上"的代价——handoff §5.4④。
- **constriction 极薄（中位 0.00048）**：正 clearance 区域的边界贴着 3 个顶点球的
  圆盘边缘——通道在球表面处收窄到近乎关闭。这意味网格生成需要局部 h < 0.0005，
  比当前 0.10 的尺寸场细 200×，**网格化不可行**。
- **但连通性还在**：每个面都有一条（虽然极细的）通道连接两个胞。

## 三个已知坑的最终判定

| 坑 | v1 粗口径 | v2 细化 | 判定 |
|---|---|---|---|
| ① 凸包覆盖 | 0/8 角 | — | ❌ **必须补 8 个角 dummy** |
| ② 4th 球穿面 | 151/151 | **0/151** | ✅ **不存在**（v1 是误报） |
| ③ 无流体通道 | 0/151 | 0/151 | ✅ 不存在 |
| **④ 界面偏离真喉道** | 21× | 21× | ❌ **本质缺陷** |

## 结论：Delaunay 四面体剖分在随机几何是否可行？

**几何正确性**：✅ 通过。151 个界面中 150 个有连通的流体通道，无 4th 球侵入。

**但是**：
1. ❌ **界面不在喉道**——球心平面上的通道是孔体最宽处（~0.16），
   不是喉道最窄处（~0.007）。DDPNM 的模型假设（界面 = 流动瓶颈）被打破。
2. ❌ **constriction 极薄**——通道在球表面处近乎关闭（中位 0.00048），
   无法在当前网格尺寸下解析。
3. ❌ **凸包不覆盖立方体**——必须补 8 个角 dummy，边界胞的界面端点不再是球心。

**总体判定**：`outputs/delaunay_feasibility/delaunay_summary_v2.md` 列出的四个指标
（胞数 84、穿透面数 0、无通道面数 0、净距分布 vs 0.0073）表明 Delaunay 方法
在几何层面是自洽的——**子区域可以画出来**。但"界面偏离喉道"的本质缺陷意味着
这种分区的 DDPNM 求解精度不会好于 Voronoi（可能更差，因为 Classic 基在孔体
内部近似更差）。建议不作为原型开发。

---

## 图的输出位置

| 图 | 路径 | 内容 |
|---|---|---|
| v1 骨架 + 凸包 + 界面 | `outputs/delaunay_feasibility/delaunay_partition.png` | Delaunay wireframe、红方块=外面角、界面着色 |
| v1 dummy + 鞍点直方图 | `outputs/delaunay_feasibility/delaunay_dummies_and_saddles.png` | 8 角 dummy 网格 + 鞍点分布 vs Voronoi |
| v1 球 + 界面 3D | `outputs/delaunay_feasibility/delaunay_spheres_interfaces.png` | 半透明球线框 + 界面 + 鞍点绿点 |
| **v2 细化分类** | **`outputs/delaunay_feasibility/delaunay_refined_v2.png`** | 绿=clean 黄=holed 橙=frag 红=blocked |
| JSON 数据 v2 | `outputs/delaunay_feasibility/delaunay_report_v2.json` | 每面详细指标 |
| 摘要 v2 | `outputs/delaunay_feasibility/delaunay_summary_v2.md` | 分类统计 |
| **本文档** | **`outputs/delaunay_feasibility/delaunay_trivial_partition_report.md`** | 最终分析 + 判定 |

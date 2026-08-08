# Delaunay tetrahedron partition — refined geometric screening (v2)

Sampling grid: 25×25 per triangular face
Delaunay tetrahedra: 84
Shared interfaces: 151

## Interface classification

| class | count | pct |
|---|---|---|
| **clean** (viable, no intruders) | 151 | 100.0% |
| **holed** (viable, with intruders) | -1 | -0.7% |
| **fragmented** (>1 positive-clearance component) | 1 | 0.7% |
| **blocked** (< 1% open area) | 0 | 0.0% |

## Intruder statistics
- Faces with ≥1 intruder: **0/151**
- Clean faces (no intruders): **151/151**
- Total intruder instances: 0

## Clearance statistics
- Max clearance (saddle): med=0.1571 (Voronoi ref=0.0073)
- Min positive clearance (constriction): med=0.000483
- Convex hull corners inside: 0/8

## Decision
⚠️ 1 interfaces failed viability: 0 blocked, 1 fragmented.

# Delaunay tetrahedron partition — feasibility check

**Sphere centres**: 27 (18 boundary + 9 interior)
**Delaunay tetrahedra**: 84
**Shared interfaces**: 151

## Convex hull coverage
- Cube corners inside hull: **0/8**
- With 8 corner dummies: 131 tets, 111 real–real + 46 real–dummy interfaces

## Interface pitfalls
- 4th-sphere penetrated: **151/151**
- No fluid channel (3-disc cover): **0/151**

## Saddle clearance
- min = 0.060522
- **median = 0.154644**
- max = 0.248311
- Voronoi reference median = 0.0073

## Centroid clearance (quick diagnostic)
- min = -0.066498
- median = 0.101539
- max = 0.197571

## Saddle clearance histogram bins
| bin | count |
|---|---|
| [-0.050, -0.020) | 0 |
| [-0.020, 0.000) | 0 |
| [0.000, 0.005) | 0 |
| [0.005, 0.010) | 0 |
| [0.010, 0.020) | 0 |
| [0.020, 0.050) | 0 |
| [0.050, 0.100) | 7 |
| [0.100, 0.150) | 57 |
| [0.150, 0.250) | 87 |

## Decision
- ⚠️ 151 interfaces are **penetrated by a 4th sphere** — the face plane cuts through an uninvolved sphere, creating a hole.
- ⚠️ 8 cube corners are **outside the convex hull** — need dummy corner points or shell closure.
- Saddle clearance median 0.1546 vs Voronoi 0.0073

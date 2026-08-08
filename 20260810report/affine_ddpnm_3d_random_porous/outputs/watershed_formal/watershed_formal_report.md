# Formal-mesh partition comparison: Voronoi vs watershed (hand-off 5.1)

- mesh cells: watershed 6586, voronoi 15249
- mesh parameters: {'bulk_size': 0.13, 'sphere_size': 0.065, 'boundary_size': 0.085, 'sphere_band': 0.15, 'boundary_band': 0.13, 'policy': 'walls_and_spheres'}

## Caveat: interface semantics

Voronoi interfaces are sphere-centre bisector planes; watershed interfaces are clearance-saddle facet sets between basins. The per-throat Hausdorff compares the planar Voronoi approximation with the mesh-resolved watershed saddle surface of the same sphere pair; it is a drawing-method deviation, not a partition error.

The watershed facet set is the reference for the Hausdorff comparison; the Voronoi face is the approximation under test.

## Region statistics

### Watershed (82 basins)

- regions: **82** (watershed count is not prescribed to equal the sphere count)
- fluid volume: 0.94586
- region volume: min 0.00023, median 0.00348, max 0.13213
- interfaces: **369**, areas min 0.00132, median 0.01660, max 0.42430, total 12.2304
- coordination distribution: 2->3, 3->5, 4->10, 5->10, 6->10, 7->6, 8->9, 9->4, 10->2, 11->3, 12->4, 13->1, 14->2, 15->4, 16->1, 18->1, 19->1, 20->2, 22->1, 29->1, 35->1, 36->1

### Voronoi (27 sphere regions)

- regions: **27** (watershed count is not prescribed to equal the sphere count)
- fluid volume: 0.94613
- region volume: min 0.01363, median 0.02982, max 0.07116
- interfaces: **114**, areas min 0.00002, median 0.04904, max 0.22167, total 6.6737
- coordination distribution: 5->2, 6->4, 7->9, 8->2, 9->3, 10->1, 12->2, 13->3, 15->1

## Saddle displacement of the Voronoi plane (rationale cross-check)

- throats: 126
- median displacement: 0.0073 (2.8% of the gap) — matches the partition rationale (0.0073, 2.8%)
- worst pair by absolute displacement: [8, 19] 0.0273 (23.0% of its gap)
- worst pair by gap fraction: [4, 22] 28.5% of its gap — pair [4,22] at 0.0218 / 28.5% is exactly the rationale's reported value (candidate sets differ in composition, not in median)

## Per-throat Hausdorff distance (watershed as reference)

- matched: 106/114 pairs (no_match 8)
- symmetric Hausdorff: median 0.0858, max 0.2420 (pair [[8, 14]])
- Voronoi -> watershed: median 0.0858, max 0.2420
- watershed -> Voronoi: median 0.0470, max 0.0500

The measured surface deviation is bounded below by the mesh-scale staircase noise of the watershed facets (sphere/boundary size fields 0.065/0.085, bulk 0.13); the analytic saddle displacement (median 0.0073) is far below that floor, so at this resolution the drawing deviation is sub-grid.

## Watershed topology invariants (formal mesh)

- every_cell_labeled: PASS
- every_basin_connected: PASS
- interface_facets_have_two_cells: PASS
- interface_facets_labels_differ: PASS
- interface_components_edge_connected: PASS
- interfaces_terminate_at_walls_or_junctions: PASS
- pore_volumes_sum_to_fluid: PASS
- interface_areas_positive: PASS
- normals_oriented_low_to_high: PASS

## Threshold scan (abs / rel -> basins)

| abs | rel | basins |
|-----|-----|--------|
| 0.01 | 0.05 | 223 |
| 0.01 | 0.1 | 213 |
| 0.02 | 0.05 | 85 |
| 0.02 | 0.1 | 84 |
| 0.04 | 0.05 | 10 |
| 0.04 | 0.1 | 10 |

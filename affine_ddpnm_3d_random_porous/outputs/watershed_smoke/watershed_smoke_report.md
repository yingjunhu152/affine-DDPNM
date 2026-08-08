# Watershed partition smoke report (experiment A)

- solid spheres: **27**
- mesh cells: **5479**
- boundary policy: `walls_and_spheres` (spheres + lateral walls; inlet/outlet open)
- raw clearance local maxima: **5479**

## Pores after persistence filtering

- pore basins: **78** (not constrained to equal the sphere count)
- basin sizes: [635, 350, 550, 224, 49, 68, 185, 95, 168, 58, 44, 197, 56, 140, 296, 76, 153, 86, 89, 64, 25, 152, 38, 117, 32, 9, 23, 105, 70, 82, 44, 8, 19, 30, 98, 37, 86, 19, 42, 21, 27, 105, 12, 54, 53, 25, 16, 11, 22, 24, 39, 10, 26, 11, 67, 25, 26, 9, 12, 20, 9, 31, 18, 10, 50, 8, 4, 59, 8, 12, 3, 4, 4, 3, 5, 4, 5, 8]
- coordination numbers: {'1': 1, '2': 6, '3': 3, '4': 10, '5': 12, '6': 7, '7': 5, '8': 8, '9': 5, '10': 5, '11': 2, '12': 2, '13': 3, '14': 0, '15': 0, '16': 0, '17': 1, '18': 1, '19': 2, '20': 2, '21': 0, '22': 0, '23': 1, '24': 0, '25': 0, '26': 0, '27': 0, '28': 0, '29': 0, '30': 0, '31': 0, '32': 1, '33': 0, '34': 0, '35': 0, '36': 0, '37': 0, '38': 0, '39': 0, '40': 0, '41': 0, '42': 0, '43': 0, '44': 0, '45': 0, '46': 0, '47': 1}

## Interfaces

- interface components: **329**
- areas: min 0.0015, median 0.0203, max 0.5533
- normal dispersion: median 35.1218, max 222.3031
- saddle values: min 0.0112, max 0.2224

## Threshold scan (abs / rel -> basins)

| abs | rel | basins | markers |
|-----|-----|--------|---------|
| 0.01 | 0.05 | 205 | 205 |
| 0.01 | 0.1 | 197 | 197 |
| 0.02 | 0.05 | 80 | 80 |
| 0.02 | 0.1 | 80 | 80 |
| 0.04 | 0.05 | 12 | 12 |
| 0.04 | 0.1 | 12 | 12 |

## Boundary-policy sensitivity

- `walls_and_spheres`: 80 basins
- `cube_and_spheres`: 90 basins

## Topology invariants

- every_cell_labeled: PASS
- every_basin_connected: PASS
- interface_facets_have_two_cells: PASS
- interface_facets_labels_differ: PASS
- interface_components_edge_connected: PASS
- interfaces_terminate_at_walls_or_junctions: PASS
- pore_volumes_sum_to_fluid: PASS
- interface_areas_positive: PASS
- normals_oriented_low_to_high: PASS

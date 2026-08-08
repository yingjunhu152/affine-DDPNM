"""Replace SPHERES in geometry.py with the 85.6% dense packing."""
import numpy as np
from weighted_voronoi_prototype import generate_dense_packing

s = generate_dense_packing(seed=20260806, target=100, min_gap=0.008,
    r_small=(0.055, 0.068), r_large=(0.080, 0.098),
    large_fraction=0.40, boundary_count=18)
v = np.sum(4/3*np.pi*s[:,3]**3)
print(f'{len(s)} spheres, porosity={1-v:.1%}')

# Build SPHERES string
lines = ['SPHERES = np.asarray([']
for x,y,z,r in s:
    lines.append(f'    ({x:.12f}, {y:.12f}, {z:.12f}, {r:.12f}),')
lines.append('], dtype=float)')
new_spheres = '\n'.join(lines)

# Replace in geometry.py
with open('geometry.py', encoding='utf-8') as f:
    content = f.read()

start = content.index('SPHERES = np.asarray([')
end = content.index('], dtype=float)', start)
end = content.index('\n', end) + 1
new_content = content[:start] + new_spheres + content[end:]

with open('geometry.py', 'w', encoding='utf-8') as f:
    f.write(new_content)
print('geometry.py updated')

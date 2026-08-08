"""W_{1n} (normal-only linear, 3 modes/face) vs W_{0v} vs W_{0n} vs W_{1v}.
Runs from 20260727 where the fenicsx environment actually works."""
import sys, time, numpy as np
from pathlib import Path

# Set up paths
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ddpnm_3d_uniform_spheres"))

from ddpnm_core.library import build_response_library
from ddpnm_core.assembler import InterfaceAssembler
from ddpnm_core.fem_utils import solve_reference
from ddpnm_core.validation import finite_element_error_analysis
from ddpnm_core.io import topology_arrays
from ddpnm3d.geometry import build_partition
from ddpnm3d.basis_3d import HierarchyBasis
# Monkey-patch: assembler calls active_transform which the basis doesn't define
HierarchyBasis.active_transform = lambda self, pm, pi, lv: None
from ddpnm3d.solver import LocalResponse, build_modes, DdpnmSolution

# ---------------------------------------------------------------------------
# W_{1n} basis: overrides level 0 = {normal} × {P0, P1_s, P1_t} (3 modes)
# ---------------------------------------------------------------------------
class W1nBasis(HierarchyBasis):
    name = "W1n-normal-linear"
    def active_indices(self, primitive_modes, port_index, level):
        indices = []
        for idx, mode in enumerate(primitive_modes):
            if mode.port_index != port_index:
                continue
            if mode.interface_id is None:
                indices.append(idx); continue
            # W_{1n}: normal direction, all scalar shapes
            if mode.component == "normal":
                indices.append(idx)
        return tuple(indices)

# ---------------------------------------------------------------------------
# One helper to convert SchurSystem → DdpnmSolution (for error analysis)
# ---------------------------------------------------------------------------
def to_ddpnm_solution(lib, system, m):
    k2d = {k: d for d, k in enumerate(system.global_keys)}
    loc = []
    for e in lib.entries:
        G = e.primitive_G; s = max(float(np.linalg.norm(G)), 1e-30)
        loc.append(LocalResponse(e.operator.pore_id, e.operator.submesh,
            e.operator.parent_cell_map, e.operator.parent_vertex_map,
            e.operator.ports, build_modes(e.operator.ports), e.operator.W,
            G, e.primitive_responses, e.operator.ndofs,
            e.symmetry_error, float(np.linalg.norm(G@np.ones(G.shape[0]))/s)))
    ip = np.array([system.coefficients[k2d[(i,'normal','P0')]] for i in range(m)])
    fi = np.array([system.moment_residuals[k2d[(i,'normal','P0')]] for i in range(m)])
    return DdpnmSolution(ip, system.schur_matrix, system.rhs, loc, system.local_solutions,
        fi, system.boundary_fluxes, system.min_schur_eigenvalue,
        float(np.max(np.abs(system.moment_residuals))) if len(system.moment_residuals) else 0.0)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
t0 = time.perf_counter()
print('Building geometry (mesh_size=0.28)...', flush=True)
partition = build_partition(mesh_size=0.28, sphere_size=0.070, boundary_size=0.10,
                            interface_size=0.10, sphere_band=0.10,
                            boundary_band=0.08, interface_band=0.08)
m = len(partition.interface_pairs)
ncells = partition.mesh.topology.index_map(partition.mesh.topology.dim).size_local
print(f'  {ncells} cells, {m} interfaces', flush=True)

print('FEM reference...', flush=True)
ref = solve_reference(partition.mesh, iterative_threshold=100000)
print(f'  {ref.ndofs} DOFs', flush=True)

print('Library (factorize once)...', flush=True)
t1 = time.perf_counter()
basis = HierarchyBasis(partition)
lib = build_response_library(partition, basis)
offline = time.perf_counter() - t1
print(f'  {offline:.1f}s', flush=True)

# Standard levels (validated path)
asm = InterfaceAssembler(lib)
print('W_0n (Classic, Level 0)...', flush=True); t1=time.perf_counter()
sys_0n = asm.assemble(np.zeros(m, dtype=np.int8)); t_0n = time.perf_counter()-t1
print('W_0v (P0-vector, Level 1)...', flush=True); t1=time.perf_counter()
sys_0v = asm.assemble(np.ones(m, dtype=np.int8)); t_0v = time.perf_counter()-t1
print('W_1v (Affine, Level 2)...', flush=True); t1=time.perf_counter()
sys_1v = asm.assemble(np.full(m, 2, dtype=np.int8)); t_1v = time.perf_counter()-t1

# W_{1n} (normal-only linear, custom basis)
print('W_1n (normal-linear, 3 modes)...', flush=True); t1=time.perf_counter()
w1n_asm = InterfaceAssembler(lib)
w1n_asm._basis = W1nBasis(partition)  # swap basis
sys_1n = w1n_asm.assemble(np.zeros(m, dtype=np.int8))  # level 0 → W_{1n}
t_1n = time.perf_counter()-t1

# Error analysis
pts, tet = topology_arrays(partition.mesh)
xyz = pts[tet]
v = np.stack([xyz[:,1]-xyz[:,0], xyz[:,2]-xyz[:,0], xyz[:,3]-xyz[:,0]], axis=1)
vols = np.abs(np.linalg.det(v)) / 6.0

sol_0n = to_ddpnm_solution(lib, sys_0n, m); met_0n = finite_element_error_analysis(partition, sol_0n, ref, vols)[0]
sol_0v = to_ddpnm_solution(lib, sys_0v, m); met_0v = finite_element_error_analysis(partition, sol_0v, ref, vols)[0]
sol_1n = to_ddpnm_solution(lib, sys_1n, m); met_1n = finite_element_error_analysis(partition, sol_1n, ref, vols)[0]
sol_1v = to_ddpnm_solution(lib, sys_1v, m); met_1v = finite_element_error_analysis(partition, sol_1v, ref, vols)[0]

ttl = time.perf_counter()-t0
print()
print('='*80)
print(f'{"Method":<38} {"m/f":>4} {"L2(u)":>8} {"br-H1":>8} {"flux":>8} {"online":>8}')
print('-'*80)
print(f'{"W_0n: Classic P0 (1·n)":<38} {1:>4} {met_0n["velocity_relative_l2"]:>7.2%} {met_0n["velocity_relative_broken_h1_seminorm"]:>7.2%} {met_0n["outlet_flux_relative_error"]:>7.2%} {t_0n:>7.1f}s')
print(f'{"W_0v: P0-vector (1·n, 1·t1, 1·t2)":<38} {3:>4} {met_0v["velocity_relative_l2"]:>7.2%} {met_0v["velocity_relative_broken_h1_seminorm"]:>7.2%} {met_0v["outlet_flux_relative_error"]:>7.2%} {t_0v:>7.1f}s')
print(f'{"W_1n: normal-linear (1·n, s·n, t·n)":<38} {3:>4} {met_1n["velocity_relative_l2"]:>7.2%} {met_1n["velocity_relative_broken_h1_seminorm"]:>7.2%} {met_1n["outlet_flux_relative_error"]:>7.2%} {t_1n:>7.1f}s')
print(f'{"W_1v: Affine (all 9)":<38} {9:>4} {met_1v["velocity_relative_l2"]:>7.2%} {met_1v["velocity_relative_broken_h1_seminorm"]:>7.2%} {met_1v["outlet_flux_relative_error"]:>7.2%} {t_1v:>7.1f}s')
print('='*80)
print(f'cells={ncells}, FEM={ref.ndofs} DOFs, offline={offline:.1f}s, total={ttl:.1f}s')

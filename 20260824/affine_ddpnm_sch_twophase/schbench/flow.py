"""Interchangeable FEM, classic DDPNM and affine-DDPNM flow solvers."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np

from .config import FlowMethod, Numerics, Physics
from .geometry import enable_flow_dependencies


@dataclass(frozen=True)
class FlowResult:
    vertex_velocity: np.ndarray
    outlet_flux: float
    global_unknowns: int
    relative_linear_residual: float
    wall_seconds: float
    stabilization: str = "none"


def _pod_affine_solution(library, full, tolerance: float = 1.0e-8):
    """Solve an ill-conditioned complete Affine Schur space by scaled POD."""
    S = 0.5 * (np.asarray(full.schur_matrix) + np.asarray(full.schur_matrix).T)
    diagonal = np.abs(np.diag(S))
    cutoff = tolerance * max(float(diagonal.max()), 1.0e-30)
    kept_indices = np.flatnonzero(diagonal >= cutoff)
    if not len(kept_indices):
        raise RuntimeError("POD discarded every Affine diagonal candidate")
    active = S[np.ix_(kept_indices, kept_indices)]
    scaling = 1.0 / np.sqrt(diagonal[kept_indices])
    scaled = scaling[:, None] * active * scaling[None, :]
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (scaled + scaled.T))
    maximum = max(float(np.max(np.abs(eigenvalues))), 1.0e-30)
    keep = eigenvalues >= tolerance * maximum
    if not np.any(keep):
        raise RuntimeError("POD discarded every Affine Schur direction")
    transform = np.zeros((len(diagonal), int(np.count_nonzero(keep))))
    transform[np.ix_(kept_indices, np.arange(np.count_nonzero(keep)))] = (
        scaling[:, None] * eigenvectors[:, keep]
    )
    reduced = transform.T @ S @ transform
    reduced_rhs = transform.T @ full.rhs
    alpha = np.linalg.solve(reduced, reduced_rhs)
    coefficients = transform @ alpha
    projected_residual = float(
        np.linalg.norm(reduced @ alpha - reduced_rhs)
        / max(np.linalg.norm(reduced_rhs), 1.0e-30)
    )
    full_residual = float(
        np.linalg.norm(S @ coefficients - full.rhs)
        / max(np.linalg.norm(full.rhs), 1.0e-30)
    )

    key_to_dof = {key: index for index, key in enumerate(full.global_keys)}
    local_solutions: list[np.ndarray] = []
    boundary_fluxes = {"inlet": 0.0, "outlet": 0.0}
    for entry in library.entries:
        local_coefficients = np.zeros(len(entry.primitive_modes), dtype=float)
        for index, mode in enumerate(entry.primitive_modes):
            if mode.interface_id is None:
                local_coefficients[index] = float(mode.known_coefficient)
            else:
                key = (int(mode.interface_id), mode.component, mode.polynomial)
                local_coefficients[index] = coefficients[key_to_dof[key]]
        local_solutions.append(entry.primitive_responses @ local_coefficients)
        moments = entry.primitive_G @ local_coefficients
        for index, mode in enumerate(entry.primitive_modes):
            if mode.interface_id is not None:
                continue
            port = entry.operator.ports[mode.port_index]
            if port.kind in boundary_fluxes:
                boundary_fluxes[port.kind] -= float(moments[index])
    note = (
        f"scaled-POD tol={tolerance:g}, retained={np.count_nonzero(keep)}/{len(diagonal)}, "
        f"full-space residual before={full.relative_linear_residual:.3e}, "
        f"discarded-direction residual={full_residual:.3e}"
    )
    return SimpleNamespace(local_solutions=local_solutions), boundary_fluxes, projected_residual, int(np.count_nonzero(keep)), note


def _rescale_library(base, pore_viscosity: np.ndarray):
    """Exact velocity/compliance scaling for porewise-constant viscosity."""
    entries = []
    for entry in base.entries:
        pore = int(entry.operator.pore_id)
        scale = 1.0 / float(pore_viscosity[pore])
        entries.append(replace(
            entry,
            primitive_responses=entry.primitive_responses * scale,
            primitive_G=entry.primitive_G * scale,
        ))
    return replace(base, entries=entries)


def _reconstruct_vertex_velocity(partition, library, system) -> np.ndarray:
    from postprocess.fields import mixed_solution_to_p1

    nvertices = partition.mesh.topology.index_map(0).size_local
    gdim = partition.mesh.geometry.dim
    total = np.zeros((nvertices, gdim), dtype=float)
    counts = np.zeros(nvertices, dtype=np.int32)
    for entry, solution in zip(library.entries, system.local_solutions, strict=True):
        local_u, _local_p = mixed_solution_to_p1(entry.operator.W, solution)
        parent = np.asarray(entry.operator.parent_vertex_map, dtype=np.int32)
        total[parent] += local_u
        counts[parent] += 1
    if np.any(counts == 0):
        raise RuntimeError("DDPNM reconstruction missed global mesh vertices")
    return total / counts[:, None]


class FlowSolver:
    method: FlowMethod
    offline_seconds: float = 0.0

    def solve(self, pore_viscosity: np.ndarray) -> FlowResult:
        raise NotImplementedError


class FemFlowSolver(FlowSolver):
    method = FlowMethod.FEM

    def __init__(self, partition, physics: Physics, numerics: Numerics):
        started = time.perf_counter()
        enable_flow_dependencies()
        from basix.ufl import element
        from dolfinx import fem
        import ufl
        from ddpnm_core.fem_utils import global_boundary_tags, mixed_stokes_space, to_scipy_matrix
        from scipy.sparse.linalg import splu

        self.partition = partition
        self.physics = physics
        self.numerics = numerics
        msh = partition.mesh
        cell = msh.basix_cell()
        self._mu = fem.Function(fem.functionspace(msh, element("DG", cell, 0)))
        self._inlet_pressure = fem.Constant(msh, float(physics.inlet_pressure))
        self._outlet_pressure = fem.Constant(msh, float(physics.outlet_pressure))
        self._W = mixed_stokes_space(msh, velocity_degree=2)
        (u, p) = ufl.TrialFunctions(self._W)
        (v, q) = ufl.TestFunctions(self._W)
        self._tags = global_boundary_tags(msh)
        self._normal = ufl.FacetNormal(msh)
        dx = ufl.dx(domain=msh)
        self._ds = ufl.Measure("ds", domain=msh, subdomain_data=self._tags)
        a = (
            self._mu * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
            - p * ufl.div(v) * dx - q * ufl.div(u) * dx
            - numerics.pressure_stabilization * p * q * dx
        )
        L = (
            -self._inlet_pressure * ufl.dot(self._normal, v) * self._ds(2)
            -self._outlet_pressure * ufl.dot(self._normal, v) * self._ds(3)
        )
        velocity_space, self._velocity_to_mixed = self._W.sub(0).collapse()
        pressure_space, self._pressure_to_mixed = self._W.sub(1).collapse()
        self._velocity_to_mixed = np.asarray(self._velocity_to_mixed, dtype=np.int32)
        self._pressure_to_mixed = np.asarray(self._pressure_to_mixed, dtype=np.int32)
        wall_dofs = fem.locate_dofs_topological(
            (self._W.sub(0), velocity_space), msh.topology.dim - 1, self._tags.find(1)
        )
        self._bcs = [fem.dirichletbc(fem.Function(velocity_space), wall_dofs, self._W.sub(0))]
        self._a_form = fem.form(a)
        self._L_form = fem.form(L)
        pressure_trial = ufl.TrialFunction(pressure_space)
        pressure_test = ufl.TestFunction(pressure_space)
        pressure_mass = to_scipy_matrix(fem.assemble_matrix(fem.form(pressure_trial * pressure_test * dx)))
        self._pressure_factor = splu(pressure_mass.tocsc())
        self.offline_seconds = time.perf_counter() - started

    def solve(self, pore_viscosity: np.ndarray) -> FlowResult:
        from dolfinx import fem
        import ufl
        from ddpnm_core.fem_utils import to_numpy_vector, to_scipy_matrix
        from postprocess.fields import mixed_solution_to_p1
        from scipy.sparse.linalg import LinearOperator, gmres, spilu

        started = time.perf_counter()
        pore_mu = np.asarray(pore_viscosity, dtype=float)
        mu_ref = float(np.exp(np.mean(np.log(pore_mu))))
        labels = np.asarray(self.partition.cell_labels, dtype=np.int32)
        self._mu.x.array[:] = pore_mu[labels] / mu_ref
        self._inlet_pressure.value = self.physics.inlet_pressure / mu_ref
        self._outlet_pressure.value = self.physics.outlet_pressure / mu_ref
        A = to_scipy_matrix(fem.assemble_matrix(self._a_form, bcs=self._bcs))
        b = fem.assemble_vector(self._L_form)
        fem.apply_lifting(b.array, [self._a_form], [self._bcs])
        fem.set_bc(b.array, self._bcs)
        rhs = to_numpy_vector(b)
        velocity_block = A[self._velocity_to_mixed][:, self._velocity_to_mixed].tocsr()
        scalar_stiffness = velocity_block[0::3, 0::3].tocsc()
        velocity_ilu = spilu(scalar_stiffness, drop_tol=1.0e-3, fill_factor=8.0, permc_spec="COLAMD")

        def precondition(vector: np.ndarray) -> np.ndarray:
            result = np.zeros_like(vector)
            velocity_rhs = vector[self._velocity_to_mixed].reshape(-1, 3)
            velocity_result = np.column_stack(
                [velocity_ilu.solve(velocity_rhs[:, component]) for component in range(3)]
            )
            result[self._velocity_to_mixed] = velocity_result.reshape(-1)
            result[self._pressure_to_mixed] = self._pressure_factor.solve(vector[self._pressure_to_mixed])
            return result

        preconditioner = LinearOperator(A.shape, matvec=precondition, dtype=np.float64)
        solution, info = gmres(
            A, rhs, M=preconditioner, rtol=1.0e-6, atol=0.0,
            restart=50, maxiter=10,
        )
        relative_residual = float(np.linalg.norm(A @ solution - rhs) / max(np.linalg.norm(rhs), 1.0e-30))
        if info != 0 or relative_residual > 2.0e-5:
            raise RuntimeError(
                f"FEM block-GMRES failed (info={info}); relative residual={relative_residual:.3e}"
            )
        vertex_u, _vertex_p = mixed_solution_to_p1(self._W, solution)
        wh = fem.Function(self._W)
        wh.x.array[:] = solution
        uh = wh.sub(0).collapse()
        outlet_flux = float(
            fem.assemble_scalar(fem.form(ufl.dot(uh, self._normal) * self._ds(3)))
        )
        return FlowResult(
            vertex_velocity=np.asarray(vertex_u, dtype=float),
            outlet_flux=outlet_flux,
            global_unknowns=int(A.shape[0]),
            relative_linear_residual=relative_residual,
            wall_seconds=time.perf_counter() - started,
            stabilization="none",
        )


class DdpnmFlowSolver(FlowSolver):
    def __init__(self, partition, physics: Physics, numerics: Numerics, method: FlowMethod):
        if method not in (FlowMethod.CLASSIC, FlowMethod.AFFINE):
            raise ValueError("DdpnmFlowSolver requires Classic or Affine")
        enable_flow_dependencies()
        from affine_face_basis import AffineFaceBasis, CompatibleClassicP0Basis
        from ddpnm_core.library import build_response_library

        self.method = method
        self.partition = partition
        self.physics = physics
        self.numerics = numerics
        self.level = 0 if method is FlowMethod.CLASSIC else 2
        basis = CompatibleClassicP0Basis() if method is FlowMethod.CLASSIC else AffineFaceBasis(partition)
        started = time.perf_counter()
        self.base = build_response_library(
            partition,
            basis,
            viscosity=1.0,
            inlet_pressure=physics.inlet_pressure,
            outlet_pressure=physics.outlet_pressure,
            pressure_stabilization=numerics.pressure_stabilization,
            retain_responses=True,
        )
        self.offline_seconds = time.perf_counter() - started

    def solve(self, pore_viscosity: np.ndarray) -> FlowResult:
        from ddpnm_core.assembler import InterfaceAssembler

        started = time.perf_counter()
        library = _rescale_library(self.base, np.asarray(pore_viscosity, dtype=float))
        levels = np.full(len(self.partition.interface_pairs), self.level, dtype=np.int8)
        system = InterfaceAssembler(library).assemble(levels, compute_min_eigenvalue=False)
        if not np.all(np.isfinite(system.coefficients)):
            raise RuntimeError(f"{self.method.value} Schur solve returned non-finite coefficients")
        residual = float(system.relative_linear_residual)
        reconstruction = system
        boundary_fluxes = system.boundary_fluxes
        unknowns = len(system.global_keys)
        stabilization = "none"
        if residual > 1.0e-7:
            if self.method is not FlowMethod.AFFINE:
                raise RuntimeError(
                    f"{self.method.value} Schur residual is {residual:.3e}"
                )
            reconstruction, boundary_fluxes, residual, unknowns, stabilization = (
                _pod_affine_solution(
                    library, system, tolerance=self.numerics.affine_pod_tolerance
                )
            )
        if residual > 1.0e-7:
            raise RuntimeError(
                f"{self.method.value} stabilized Schur residual is {residual:.3e}"
            )
        velocity = _reconstruct_vertex_velocity(self.partition, library, reconstruction)
        return FlowResult(
            vertex_velocity=velocity,
            outlet_flux=float(boundary_fluxes.get("outlet", np.nan)),
            global_unknowns=unknowns,
            relative_linear_residual=residual,
            wall_seconds=time.perf_counter() - started,
            stabilization=stabilization,
        )


def build_flow_solver(partition, physics: Physics, numerics: Numerics, method: FlowMethod) -> FlowSolver:
    if method is FlowMethod.FEM:
        return FemFlowSolver(partition, physics, numerics)
    return DdpnmFlowSolver(partition, physics, numerics, method)

"""Robust global conforming Cahn--Hilliard transport discretization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ufl
from basix.ufl import element, mixed_element
from dolfinx import fem
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

from .config import Numerics, Physics
from .geometry import enable_flow_dependencies


@dataclass(frozen=True)
class TransportDiagnostics:
    newton_iterations: int
    residual_inf: float
    line_search_reductions: int
    mass: float
    free_energy: float
    phi_min: float
    phi_max: float


def _matrix_to_scipy(matrix):
    if hasattr(matrix, "to_scipy"):
        return matrix.to_scipy().tocsr()
    from scipy.sparse import csr_matrix
    return csr_matrix(matrix.to_dense())


class CahnHilliardTransport:
    """P1--P1 mixed convex-splitting CH solve on the parent mesh.

    All three flow methods use this same transport object.  That makes the
    six-arm comparison measure flow-space reduction rather than differences
    between unrelated transport discretizations.
    """

    def __init__(self, msh, physics: Physics, numerics: Numerics):
        enable_flow_dependencies()
        from ddpnm_core.fem_utils import global_boundary_tags
        from ddpnm_core.io import topology_vertex_coordinates

        self.mesh = msh
        self.physics = physics
        self.numerics = numerics
        cell = msh.basix_cell()
        scalar = element("Lagrange", cell, 1)
        vector = element("Lagrange", cell, 1, shape=(msh.geometry.dim,))
        self.Q = fem.functionspace(msh, scalar)
        self.V = fem.functionspace(msh, vector)
        self.Z = fem.functionspace(msh, mixed_element([scalar, scalar]))
        self.phi_old = fem.Function(self.Q, name="phi_old")
        self.velocity = fem.Function(self.V, name="velocity")
        self.state = fem.Function(self.Z, name="phi_mu")
        self._phi_space, self._phi_to_state = self.Z.sub(0).collapse()
        self._mu_space, self._mu_to_state = self.Z.sub(1).collapse()
        self._phi_to_state = np.asarray(self._phi_to_state, dtype=np.int32)
        self._mu_to_state = np.asarray(self._mu_to_state, dtype=np.int32)

        tags = global_boundary_tags(msh)
        inlet_facets = tags.find(2)
        inlet_dofs = fem.locate_dofs_topological(
            (self.Z.sub(0), self._phi_space), msh.topology.dim - 1, inlet_facets
        )
        inlet_value = fem.Function(self._phi_space)
        inlet_value.x.array[:] = physics.phi_inlet
        self.bcs = [fem.dirichletbc(inlet_value, inlet_dofs, self.Z.sub(0))]

        coords = topology_vertex_coordinates(msh)
        qcoords = self.Q.tabulate_dof_coordinates()[:, :msh.geometry.dim]
        distances, self._vertex_to_q = cKDTree(qcoords).query(coords, k=1)
        if float(np.max(distances)) > 1.0e-9:
            raise RuntimeError("P1 scalar dofs do not match topology vertices")
        vcoords = self.V.tabulate_dof_coordinates()[:, :msh.geometry.dim]
        distances, self._vertex_to_v = cKDTree(vcoords).query(coords, k=1)
        if float(np.max(distances)) > 1.0e-9:
            raise RuntimeError("P1 velocity dofs do not match topology vertices")
        self._vertex_to_q = np.asarray(self._vertex_to_q, dtype=np.int32)
        self._vertex_to_v = np.asarray(self._vertex_to_v, dtype=np.int32)

        h = self._mean_edge_length()
        self.epsilon = physics.epsilon_factor * h
        phi, chemical = ufl.split(self.state)
        test_phi, test_chemical = ufl.TestFunctions(self.Z)
        dx = ufl.dx(domain=msh)
        dt = numerics.dt
        residual = (
            (phi - self.phi_old) * test_phi / dt * dx
            + ufl.dot(self.velocity, ufl.grad(phi)) * test_phi * dx
            + physics.mobility * ufl.dot(ufl.grad(chemical), ufl.grad(test_phi)) * dx
            + chemical * test_chemical * dx
            - physics.surface_tension / self.epsilon
            * (phi ** 3 - self.phi_old) * test_chemical * dx
            - physics.surface_tension * self.epsilon
            * ufl.dot(ufl.grad(phi), ufl.grad(test_chemical)) * dx
        )
        trial = ufl.TrialFunction(self.Z)
        jacobian = ufl.derivative(residual, self.state, trial)
        self._residual_form = fem.form(residual)
        self._minus_residual_form = fem.form(-residual)
        self._jacobian_form = fem.form(jacobian)
        phase = fem.Function(self.Q)
        self._phase_for_metrics = phase
        self._mass_form = fem.form(phase * dx)
        self._energy_form = fem.form(
            physics.surface_tension
            * (0.25 * (phase * phase - 1.0) ** 2 / self.epsilon
               + 0.5 * self.epsilon * ufl.dot(ufl.grad(phase), ufl.grad(phase))) * dx
        )

    def _mean_edge_length(self) -> float:
        msh = self.mesh
        msh.topology.create_connectivity(msh.topology.dim, 0)
        c2v = msh.topology.connectivity(msh.topology.dim, 0)
        coords = np.asarray(msh.geometry.x, dtype=float)
        ncells = msh.topology.index_map(msh.topology.dim).size_local
        total = 0.0
        count = 0
        for cell in range(ncells):
            vertices = c2v.links(cell)
            for i in range(len(vertices)):
                for j in range(i + 1, len(vertices)):
                    total += float(np.linalg.norm(coords[vertices[i], :3] - coords[vertices[j], :3]))
                    count += 1
        return total / count

    def initial_vertices(self) -> np.ndarray:
        self.phi_old.interpolate(
            lambda x: np.where(x[0] <= 1.0e-8, self.physics.phi_inlet, self.physics.phi_initial)
        )
        return self.vertex_values(self.phi_old)

    def vertex_values(self, function: fem.Function) -> np.ndarray:
        return np.asarray(function.x.array, dtype=float)[self._vertex_to_q].copy()

    def _set_scalar_vertices(self, function: fem.Function, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=float)
        if values.shape != (len(self._vertex_to_q),):
            raise ValueError("Scalar vertex field has the wrong shape")
        accumulator = np.zeros_like(function.x.array, dtype=float)
        counts = np.zeros_like(function.x.array, dtype=np.int32)
        np.add.at(accumulator, self._vertex_to_q, values)
        np.add.at(counts, self._vertex_to_q, 1)
        function.x.array[:] = accumulator / np.maximum(counts, 1)

    def _set_velocity_vertices(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=float)
        gdim = self.mesh.geometry.dim
        if values.shape != (len(self._vertex_to_v), gdim):
            raise ValueError(f"Velocity vertex field has shape {values.shape}, expected {(len(self._vertex_to_v), gdim)}")
        blocks = self.velocity.x.array.reshape(-1, gdim)
        total = np.zeros_like(blocks)
        counts = np.zeros(len(blocks), dtype=np.int32)
        np.add.at(total, self._vertex_to_v, values)
        np.add.at(counts, self._vertex_to_v, 1)
        blocks[:] = total / np.maximum(counts[:, None], 1)

    def _rhs(self) -> tuple[np.ndarray, float]:
        vector = fem.assemble_vector(self._minus_residual_form)
        fem.apply_lifting(
            vector.array, [self._jacobian_form], [self.bcs],
            x0=[self.state.x.array], alpha=1.0,
        )
        fem.set_bc(vector.array, self.bcs, x0=self.state.x.array, alpha=1.0)
        rhs = np.asarray(vector.array, dtype=float).copy()
        return rhs, float(np.linalg.norm(rhs, ord=np.inf))

    def advance(
        self,
        old_phi_vertices: np.ndarray,
        velocity_vertices: np.ndarray,
        initial_guess_vertices: np.ndarray | None = None,
    ) -> tuple[np.ndarray, TransportDiagnostics]:
        self._set_scalar_vertices(self.phi_old, old_phi_vertices)
        self._set_velocity_vertices(velocity_vertices)
        guess = old_phi_vertices if initial_guess_vertices is None else initial_guess_vertices
        self._set_scalar_vertices(self._phase_for_metrics, guess)
        self.state.x.array[:] = 0.0
        self.state.x.array[self._phi_to_state] = self._phase_for_metrics.x.array
        fem.set_bc(self.state.x.array, self.bcs)

        reductions = 0
        residual_inf = float("inf")
        for iteration in range(self.numerics.newton_max_iterations + 1):
            rhs, residual_inf = self._rhs()
            if not np.isfinite(residual_inf):
                raise RuntimeError("CH Newton residual became non-finite")
            if residual_inf <= self.numerics.newton_tolerance:
                break
            if iteration == self.numerics.newton_max_iterations:
                raise RuntimeError(
                    f"CH Newton failed after {iteration} iterations; residual={residual_inf:.3e}"
                )
            matrix = _matrix_to_scipy(fem.assemble_matrix(self._jacobian_form, bcs=self.bcs))
            delta = splu(matrix.tocsc()).solve(rhs)
            if not np.all(np.isfinite(delta)):
                raise RuntimeError("CH Newton linear solve returned non-finite update")
            previous = self.state.x.array.copy()
            alpha = 1.0
            accepted = False
            while alpha + 1.0e-15 >= self.numerics.line_search_min:
                self.state.x.array[:] = previous + alpha * delta
                fem.set_bc(self.state.x.array, self.bcs)
                _candidate_rhs, candidate_norm = self._rhs()
                if candidate_norm < residual_inf:
                    accepted = True
                    break
                alpha *= 0.5
                reductions += 1
            if not accepted:
                self.state.x.array[:] = previous
                raise RuntimeError(
                    f"CH Newton line search stalled at residual={residual_inf:.3e}"
                )
        phase = fem.Function(self._phi_space)
        phase.x.array[:] = self.state.x.array[self._phi_to_state]
        vertices = self.vertex_values(phase)
        self._phase_for_metrics.x.array[:] = phase.x.array
        mass = float(fem.assemble_scalar(self._mass_form))
        energy = float(fem.assemble_scalar(self._energy_form))
        return vertices, TransportDiagnostics(
            newton_iterations=iteration,
            residual_inf=residual_inf,
            line_search_reductions=reductions,
            mass=mass,
            free_energy=energy,
            phi_min=float(np.min(vertices)),
            phi_max=float(np.max(vertices)),
        )

"""
Wave equation (see membrane.py doc comment).
Square domain with known plane wave solution
imposed on the edge as Dirichlet boundary condition.
Examining error compared to known solution as the
mesh is refined and timestep shortened.
"""

import mesh
import animate as anim
from sim_runner import Simulation

import numpy as np
import numpy.typing as npt
import math
import matplotlib.pyplot as plt
import pydec
from typing import Iterable


class AccuracyTest(Simulation):
    def __init__(self, elem_size: float, timesteps_per_second: int):
        # mesh parameters
        mesh_dim = np.pi
        cmp_mesh = mesh.rect_unstructured(mesh_dim, mesh_dim, elem_size)
        cmp_complex = pydec.SimplicialComplex(cmp_mesh)

        # time parameters
        sim_time = 2.0 * np.pi
        dt = 1.0 / timesteps_per_second
        step_count = math.ceil(sim_time / dt)

        # incoming wave parameters
        inc_wavenumber = 2.0
        inc_wave_dir = np.array([0.0, 1.0])
        self.inc_wave_vector = inc_wavenumber * inc_wave_dir
        self.inc_angular_vel = 2.0

        # gather boundary elements
        bound_verts = []
        bound_edges = []
        for edge in cmp_complex.boundary():
            edge_verts = edge.boundary()
            bound_edges.append(edge)
            bound_verts.extend(edge_verts)
        # remove duplicates by constructing sets
        self.bound_verts = set(bound_verts)
        self.bound_edges = set(bound_edges)

        # time stepping matrices
        self.v_step_mat = dt * cmp_complex[2].star * cmp_complex[1].d
        self.q_step_mat = dt * cmp_complex[1].star_inv * cmp_complex[1].d.T

        super().__init__(complex=cmp_complex, dt=dt, step_count=step_count)

    def init_state(self):
        # time stored for incoming wave evaluation
        self.t = 0.0

        self.v = np.zeros(self.complex[2].num_simplices)
        for vert_idx in range(len(self.v)):
            self.v[vert_idx] = self._eval_inc_wave_pressure(
                0.0, self.complex[2].circumcenter[vert_idx]
            )
        self.q = np.zeros(self.complex[1].num_simplices)
        for edge_idx in range(len(self.q)):
            edge = self.complex[1].simplices[edge_idx]
            self.q[edge_idx] = self._eval_inc_wave_flux(0.5 * self.dt, edge)

    def step(self):
        self.t += self.dt
        self.v += self.v_step_mat * self.q
        # w is computed at a time instance offset by half dt
        t_at_q = self.t + 0.5 * self.dt
        self.q += self.q_step_mat * self.v
        # plane wave at the boundary for q
        for bound_edge in self.bound_edges:
            edge_idx = self.complex[1].simplex_to_index.get(bound_edge)
            self.q[edge_idx] = self._eval_inc_wave_flux(t_at_q, bound_edge)

    def _eval_inc_wave_pressure(self, t, position: npt.NDArray) -> float:
        """Evaluate the value of v for the incoming plane wave at a point."""

        return self.inc_angular_vel * math.sin(
            self.inc_angular_vel * t - np.dot(self.inc_wave_vector, position)
        )

    def _eval_inc_wave_flux(self, t: float, edge_vert_indices: Iterable[int]) -> float:
        """Evaluate the line integral of the area flux of the incoming wave
        over an edge of the mesh, in other words compute a value of `q` from the wave."""

        p = [self.complex.vertices[v] for v in edge_vert_indices]
        kdotp = np.dot(self.inc_wave_vector, p[0])
        l = p[1] - p[0]
        kdotl = np.dot(self.inc_wave_vector, l)
        kdotn = np.dot(self.inc_wave_vector, np.array([l[1], -l[0]]))
        wave_angle = self.inc_angular_vel * t

        if abs(kdotl) < 1e-5:
            return -kdotn * math.sin(wave_angle - kdotp)
        else:
            return (kdotn / kdotl) * (
                math.cos(wave_angle - kdotp) - math.cos(wave_angle - kdotp - kdotl)
            )

    def current_max_pressure_error(self) -> float:
        """Find the largest deviation from the analytical plane wave solution's pressure
        in the current state of the simulation."""

        max_err = 0.0
        for vert, v_val in zip(self.complex[2].circumcenter, self.v):
            exact_pressure = self._eval_inc_wave_pressure(self.t, vert)
            err = abs(exact_pressure - v_val)
            if err > max_err:
                max_err = err
        return max_err

    def current_max_flux_error(self) -> float:
        """Find the largest deviation from the analytical plane wave solution's velocity
        in the current state of the simulation."""

        max_err = 0.0
        for edge_idx in range(len(self.q)):
            q_val = self.q[edge_idx]
            edge = self.complex[1].simplices[edge_idx]
            exact_flux = self._eval_inc_wave_flux(self.t + 0.5 * self.dt, edge)
            err = abs(exact_flux - q_val)
            if err > max_err:
                max_err = err
        return max_err


mesh_sizes = [np.pi / n for n in [5, 8, 10, 20, 40]]
sims = [AccuracyTest(elem_size=n, timesteps_per_second=60) for n in mesh_sizes]
vis = anim.FluxAndPressure(
    sim=sims[2], get_pressure=lambda s: s.v, get_flux=lambda s: s.q, vmin=-2, vmax=2
)
vis.show()
# vis.save_mp4()

v_errors = []
q_errors = []
for sim in sims:
    sim.run_to_end()
    v_errors.append(sim.current_max_pressure_error())
    q_errors.append(sim.current_max_flux_error())

fig = plt.figure()
v_ax = fig.add_subplot(2, 1, 1)
v_ax.set(xlabel="mesh element size", ylabel="max error in pressure")
v_ax.plot(mesh_sizes, v_errors)
w_ax = fig.add_subplot(2, 1, 2)
w_ax.set(xlabel="mesh element size", ylabel="max error in velocity")
w_ax.plot(mesh_sizes, q_errors)
plt.show()
fig.savefig("errors.png")

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


class AccuracyTest(Simulation):
    def __init__(self, mesh_subdivisions: int, timesteps_per_second: int):
        # mesh parameters
        mesh_dim = np.pi
        mesh_scale = mesh_dim / mesh_subdivisions
        cmp_mesh = mesh.rect_unstructured(mesh_dim, mesh_dim, mesh_scale)
        self.cmp_complex = pydec.SimplicialComplex(cmp_mesh)

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
        for edge in self.cmp_complex.boundary():
            edge_verts = edge.boundary()
            bound_edges.append(edge)
            bound_verts.extend(edge_verts)
        # remove duplicates by constructing sets
        self.bound_verts = set(bound_verts)
        self.bound_edges = set(bound_edges)

        # time stepping matrices
        self.v_step_mat = (
            -dt
            * self.cmp_complex[0].star_inv
            * self.cmp_complex[0].d.T
            * self.cmp_complex[1].star
        )
        self.w_step_mat = dt * self.cmp_complex[0].d

        super().__init__(mesh=cmp_mesh, dt=dt, step_count=step_count)

    def init_state(self):
        # time needed for incoming wave evaluation
        self.t = 0.0

        self.v = np.zeros(self.cmp_complex[0].num_simplices)
        for vert_idx in range(len(self.v)):
            # time derivative of the incoming wave
            self.v[vert_idx] = self._eval_inc_wave_pressure(
                0.0, self.mesh.vertices[vert_idx]
            )
        self.w = np.zeros(self.cmp_complex[1].num_simplices)
        for edge_idx in range(len(self.w)):
            edge = self.cmp_complex[1].simplices[edge_idx]
            self.w[edge_idx] = self._eval_inc_wave_velocity(0.5 * self.dt, edge)

    def step(self):
        self.t += self.dt
        self.v += self.v_step_mat * self.w
        # plane wave at the boundary for v
        for bound_vert in self.bound_verts:
            vert_idx = self.cmp_complex[0].simplex_to_index.get(bound_vert)
            self.v[vert_idx] = self._eval_inc_wave_pressure(
                self.t, self.mesh.vertices[vert_idx]
            )
        # w is computed at a time instance offset by half dt
        t_at_w = self.t + 0.5 * self.dt
        self.w += self.w_step_mat * self.v
        # plane wave at the boundary for w
        for bound_edge in self.bound_edges:
            edge_idx = self.cmp_complex[1].simplex_to_index.get(bound_edge)
            self.w[edge_idx] = self._eval_inc_wave_velocity(
                t_at_w, bound_edge.boundary()
            )

    def get_z_data(self):
        # visualizing acoustic pressure
        return self.v

    def _eval_inc_wave_pressure(self, t, position: npt.NDArray) -> float:
        """Evaluate the value of v for the incoming plane wave at a point."""

        return self.inc_angular_vel * math.sin(
            self.inc_angular_vel * t - np.dot(self.inc_wave_vector, position)
        )

    def _eval_inc_wave_velocity(self, t: float, edge_verts: list[npt.NDArray]) -> float:
        """Evaluate the line integral of the particle velocity of the incoming wave
        over an edge of the mesh, in other words compute a value of `w` from the wave."""

        p1 = self.mesh.vertices[edge_verts[0]]
        p2 = self.mesh.vertices[edge_verts[1]]
        kdotp = np.dot(self.inc_wave_vector, p1)
        kdotl = np.dot(self.inc_wave_vector, p2 - p1)
        angle = self.inc_angular_vel * t
        return math.cos(angle - kdotp) - math.cos(angle - kdotp - kdotl)

    def current_max_pressure_error(self) -> float:
        """Find the largest deviation from the analytical plane wave solution's pressure
        in the current state of the simulation."""

        max_err = 0.0
        for vert, v_val in zip(self.mesh.vertices, self.v):
            exact_pressure = self._eval_inc_wave_pressure(self.t, vert)
            err = abs(exact_pressure - v_val)
            if err > max_err:
                max_err = err
        return max_err

    def current_max_velocity_error(self) -> float:
        """Find the largest deviation from the analytical plane wave solution's velocity
        in the current state of the simulation."""

        max_err = 0.0
        for edge_idx in range(len(self.w)):
            w_val = self.w[edge_idx]
            edge = self.cmp_complex[1].simplices[edge_idx]
            exact_vel = self._eval_inc_wave_velocity(self.t + 0.5 * self.dt, edge)
            err = abs(exact_vel - w_val)
            if err > max_err:
                max_err = err
        return max_err


div_counts = [5, 10, 20, 40, 60]
sims = [AccuracyTest(mesh_subdivisions=n, timesteps_per_second=40) for n in div_counts]
vis_first = anim.ZeroForm(sim=sims[0], get_data=lambda s: s.v, zlim=[-8.0, 8.0])
vis_first.show()
vis_last = anim.ZeroForm(sim=sims[-1], get_data=lambda s: s.v, zlim=[-8.0, 8.0])
vis_last.show()

v_errors = []
w_errors = []
for sim, div_count in zip(sims, div_counts):
    sim.run_to_end()
    v_errors.append(sim.current_max_pressure_error())
    w_errors.append(sim.current_max_velocity_error())

fig = plt.figure()
v_ax = fig.add_subplot(2, 1, 1)
v_ax.set(xlabel="mesh division count", ylabel="max error in pressure")
v_ax.plot(div_counts, v_errors)
w_ax = fig.add_subplot(2, 1, 2)
w_ax.set(xlabel="mesh division count", ylabel="max error in velocity")
w_ax.plot(div_counts, w_errors)
plt.show()
fig.savefig("errors.png")

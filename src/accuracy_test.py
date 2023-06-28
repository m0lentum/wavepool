from utils import animate as anim
from utils import mesh
from utils.sim_runner import Simulation

import numpy as np
import numpy.typing as npt
import math
import matplotlib.pyplot as plt
from typing import Iterable


class AccuracyTest(Simulation):
    def __init__(
        self, elem_size: float, timesteps_per_second: int, use_harmonic_terms: bool
    ):
        # mesh parameters
        mesh_dim = np.pi
        cmp_mesh = mesh.rect_unstructured(mesh_dim, mesh_dim, elem_size)
        cmp_complex = cmp_mesh.complex

        # time parameters
        sim_time = 2.0 * np.pi
        dt = 1.0 / timesteps_per_second
        step_count = math.ceil(sim_time / dt)

        super().__init__(complex=cmp_complex, dt=dt, step_count=step_count)

        # incident wave parameters
        self.inc_wavenumber = 2.0
        inc_wave_dir = np.array([0.0, 1.0])
        self.inc_wave_vector = self.inc_wavenumber * inc_wave_dir
        self.wave_speed = 1.0
        self.inc_angular_vel = self.inc_wavenumber * self.wave_speed

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
        if use_harmonic_terms:
            self.build_harmonic_timestep()
        else:
            self.p_step_mat = (
                dt * self.wave_speed**2 * cmp_complex[2].star * cmp_complex[1].d
            )
            self.q_step_mat = dt * cmp_complex[1].star_inv * cmp_complex[1].d.T

    def build_harmonic_timestep(self):
        """Build Hodge operators and time-stepping matrices
        optimized for harmonic waves."""

        wavenumber_sq = self.inc_wavenumber**2
        star_1_inv_diag = self.complex[1].star_inv.diagonal()
        for edge_idx, (edge_len, dual_edge_len) in enumerate(
            zip(self.complex[1].primal_volume, self.complex[1].dual_volume)
        ):
            edge_curv = edge_len**2 * wavenumber_sq
            dual_curv = dual_edge_len**2 * wavenumber_sq
            star_1_inv_diag[edge_idx] /= (1.0 - dual_curv / 16.0) / (
                1.0 - edge_curv / 94.0 - dual_curv / 32.0
            )
        star_2_diag = self.complex[2].star.diagonal()
        for face_idx, face in enumerate(self.complex[2].simplices):
            center = self.complex[2].circumcenter[face_idx]
            verts = [self.complex.vertices[i] for i in face]
            vert_distances_sq = [np.dot(center - v, center - v) for v in verts]
            edge_distances_sq = []
            for i in range(len(verts)):
                edge_vec = verts[i] - verts[i - 1]
                edge_dir = edge_vec / np.linalg.norm(edge_vec)
                edge_normal = np.array([-edge_dir[1], edge_dir[0]])
                edge_dist = abs(np.dot(verts[i] - center, edge_normal))
                edge_distances_sq.append(edge_dist**2)

            face_curv = (wavenumber_sq / (3.0 * len(verts))) * (
                2.0 * sum(edge_distances_sq) + sum(vert_distances_sq)
            )
            star_2_diag[face_idx] /= 1.0 - face_curv / 8.0

        star_1_inv = np.diag(star_1_inv_diag)
        star_2 = np.diag(star_2_diag)

        harmonic_dt = (2.0 / self.inc_angular_vel) * math.sin(
            self.inc_angular_vel * self.dt / 2.0
        )
        self.p_step_mat = (
            harmonic_dt * self.wave_speed**2 * star_2 * self.complex[1].d
        )
        self.q_step_mat = harmonic_dt * star_1_inv * self.complex[1].d.T

    def init_state(self):
        # time stored for incident wave evaluation
        self.t = 0.0

        self.p = np.zeros(self.complex[2].num_simplices)
        for vert_idx in range(len(self.p)):
            self.p[vert_idx] = self._eval_inc_wave_pressure(
                0.0, self.complex[2].circumcenter[vert_idx]
            )
        self.q = np.zeros(self.complex[1].num_simplices)
        for edge_idx in range(len(self.q)):
            edge = self.complex[1].simplices[edge_idx]
            self.q[edge_idx] = self._eval_inc_wave_flux(0.5 * self.dt, edge)

    def step(self):
        self.t += self.dt
        self.p += self.p_step_mat @ self.q
        # q is computed at a time instance offset by half dt
        t_at_q = self.t + 0.5 * self.dt
        self.q += self.q_step_mat @ self.p
        # plane wave at the boundary for q
        for bound_edge in self.bound_edges:
            edge_idx = self.complex[1].simplex_to_index.get(bound_edge)
            self.q[edge_idx] = self._eval_inc_wave_flux(t_at_q, bound_edge)

    def _eval_inc_wave_pressure(self, t, position: npt.NDArray) -> float:
        """Evaluate the value of p for the incident plane wave at a point."""

        return self.inc_angular_vel * math.sin(
            self.inc_angular_vel * t - np.dot(self.inc_wave_vector, position)
        )

    def _eval_inc_wave_flux(self, t: float, edge_vert_indices: Iterable[int]) -> float:
        """Evaluate the line integral of the flux of the incident wave
        over an edge of the mesh, in other words compute a value of `q` from the wave.
        """

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
        for vert, p_val in zip(self.complex[2].circumcenter, self.p):
            exact_pressure = self._eval_inc_wave_pressure(self.t, vert)
            err = abs(exact_pressure - p_val)
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
            # normalize by edge length
            edge_len = self.complex[1].primal_volume[edge_idx]
            err = abs((exact_flux - q_val) / edge_len)
            if err > max_err:
                max_err = err
        return max_err


mesh_sizes = [np.pi / n for n in [5, 8, 10, 20, 40]]
sims_yee = [
    AccuracyTest(elem_size=n, timesteps_per_second=60, use_harmonic_terms=False)
    for n in mesh_sizes
]
sims_harmonic = [
    AccuracyTest(elem_size=n, timesteps_per_second=60, use_harmonic_terms=True)
    for n in mesh_sizes
]

vis = anim.FluxAndPressure(sim=sims_harmonic[1])
vis.show()

p_errors_yee = []
q_errors_yee = []
p_errors_harmonic = []
q_errors_harmonic = []
for sim in sims_harmonic:
    sim.run_to_end()
    p_errors_harmonic.append(sim.current_max_pressure_error())
    q_errors_harmonic.append(sim.current_max_flux_error())
for sim in sims_yee:
    sim.run_to_end()
    p_errors_yee.append(sim.current_max_pressure_error())
    q_errors_yee.append(sim.current_max_flux_error())

fig = plt.figure()
p_ax = fig.add_subplot(2, 1, 1)
p_ax.set(xlabel="mesh element size", ylabel="max error in pressure")
(plot_yee,) = p_ax.plot(mesh_sizes, p_errors_yee, label="Yee's")
(plot_har,) = p_ax.plot(mesh_sizes, p_errors_harmonic, label="Harmonic")
p_ax.legend(handles=[plot_yee, plot_har])

w_ax = fig.add_subplot(2, 1, 2)
w_ax.set(xlabel="mesh element size", ylabel="max error in velocity")
(plot_yee,) = w_ax.plot(mesh_sizes, q_errors_yee, label="Yee's")
(plot_har,) = w_ax.plot(mesh_sizes, q_errors_harmonic, label="Harmonic")
w_ax.legend(handles=[plot_yee, plot_har])
plt.show()
fig.savefig("errors.png")

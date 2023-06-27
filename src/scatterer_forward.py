"""
Wave equation (see membrane.py doc comment).
Circular scatterer object centered at origin,
circular computational domain around it with
absorbing boundary condition at the outer edge
modeling empty space.
Incident plane wave -cos(ωt - 𝜿·x).
"""

from utils import animate as anim
from utils import mesh
from utils import measure_mesh
from utils.sim_runner import Simulation

import numpy as np
import numpy.typing as npt
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Iterable


scatterer_radius = 1.0
outer_edge_radius = 3.0
cmp_mesh = mesh.annulus(scatterer_radius, outer_edge_radius, refine_count=1)
cmp_complex = cmp_mesh.complex
measure_mesh.print_measurements(cmp_complex)
inner_bound_edges: list[int] = cmp_mesh.edge_groups["inner boundary"]
outer_bound_edges: list[int] = cmp_mesh.edge_groups["outer boundary"]


# for each outer boundary edge, find the triangle this edge is part of
# and save some info for computing the absorbing boundary condition
@dataclass
class BoundaryEdgeInfo:
    dual_vert_idx: int
    length: float
    orientation: int


outer_bound_infos: list[BoundaryEdgeInfo] = []
for edge_idx in outer_bound_edges:
    # find the triangle using the incidence matrix
    tri_indices = cmp_complex[1].d[:, edge_idx].nonzero()[0]
    assert len(tri_indices) == 1, "boundary edge is part of one triangle only"
    edge_ends = [cmp_complex.vertices[v] for v in cmp_complex[1].simplices[edge_idx]]
    outer_bound_infos.append(
        BoundaryEdgeInfo(
            dual_vert_idx=tri_indices[0],
            orientation=cmp_complex[1].d[tri_indices[0], edge_idx],
            length=np.linalg.norm(edge_ends[1] - edge_ends[0]),
        )
    )


class CircleScatterer(Simulation):
    def __init__(self):
        # incident wave parameters
        inc_wavenumber = 1.0
        inc_wave_dir = np.array([0.0, 1.0])
        self.inc_wave_vector = inc_wavenumber * inc_wave_dir
        self.inc_angular_vel = 3.0

        # time parameters
        sim_time = 10.0 * np.pi
        dt = 1.0 / 20.0
        step_count = math.ceil(sim_time / dt)

        # time stepping matrices
        self.v_step_mat = dt * cmp_complex[2].star * cmp_complex[1].d
        self.q_step_mat = dt * cmp_complex[1].star_inv * cmp_complex[1].d.T

        super().__init__(complex=cmp_complex, dt=dt, step_count=step_count)

    def init_state(self):
        # time needed for incident wave evaluation
        self.t = 0.0

        self.v = np.zeros(cmp_complex[2].num_simplices)
        for vert_idx in range(len(self.v)):
            self.v[vert_idx] = self._eval_inc_wave_pressure(
                0.0, self.complex[2].circumcenter[vert_idx]
            )
        self.q = np.zeros(cmp_complex[1].num_simplices)
        for edge_idx in range(len(self.q)):
            edge = cmp_complex[1].simplices[edge_idx]
            self.q[edge_idx] = self._eval_inc_wave_flux(0.5 * self.dt, edge)

    def step(self):
        self.t += self.dt
        self.v += self.v_step_mat * self.q
        # q is computed at a time instance offset by half dt
        t_at_w = self.t + 0.5 * self.dt
        self.q += self.q_step_mat * self.v
        # incident wave on the scatterer's surface
        for edge_idx in inner_bound_edges:
            self.q[edge_idx] = self._eval_inc_wave_flux(
                t_at_w,
                cmp_complex[1].simplices[edge_idx],
            )
        # absorbing outer boundary condition
        for edge_idx, edge_info in zip(outer_bound_edges, outer_bound_infos):
            self.q[edge_idx] = (
                -self.v[edge_info.dual_vert_idx]
                * edge_info.length
                * edge_info.orientation
            )

    def _eval_inc_wave_pressure(self, t: float, position: npt.NDArray) -> float:
        """Evaluate the value of v for the incident plane wave at a point."""

        return self.inc_angular_vel * math.sin(
            self.inc_angular_vel * t - np.dot(self.inc_wave_vector, position)
        )

    def _eval_inc_wave_flux(self, t: float, edge_vert_indices: Iterable[int]) -> float:
        """Evaluate the line integral of the area flux of the incident wave
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

    def get_pressure_with_inc_wave(
        self, inc_wave_scaling: float = 0.5
    ) -> npt.NDArray[np.float64]:
        """Sum pressure of the scattered wave with that of the incident wave.
        Used for visualization."""

        inc_wave: npt.NDArray[np.float64] = np.zeros(cmp_complex[2].num_simplices)
        for vert_idx in range(len(inc_wave)):
            inc_wave[vert_idx] = self._eval_inc_wave_pressure(
                self.t, cmp_complex[2].circumcenter[vert_idx]
            )
        return self.v + inc_wave_scaling * inc_wave

    def get_flux_with_inc_wave(
        self, inc_wave_scaling: float = 0.5
    ) -> npt.NDArray[np.float64]:
        """Sum pressure of the scattered wave with that of the incident wave.
        Used for visualization."""

        inc_wave: npt.NDArray[np.float64] = np.zeros(cmp_complex[1].num_simplices)
        for edge_idx in range(len(inc_wave)):
            if edge_idx in inner_bound_edges:
                continue
            inc_wave[edge_idx] = self._eval_inc_wave_flux(
                self.t + 0.5 * self.dt, cmp_complex[1].simplices[edge_idx]
            )
        return self.q + inc_wave_scaling * inc_wave


sim = CircleScatterer()
vis = anim.FluxAndPressure(
    sim=sim,
    arrow_scale=20,
    vmin=-3,
    vmax=3,
    get_pressure=lambda s: s.get_pressure_with_inc_wave(),
    get_flux=lambda s: s.get_flux_with_inc_wave(),
)
vis.show()
# vis.save()

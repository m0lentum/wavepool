"""
Wave equation (see membrane.py doc comment).
Circular scatterer object centered at origin,
circular computational domain around it with
absorbing boundary condition at the outer edge
modeling empty space.
Incoming plane wave -cos(ωt - 𝜿·x).
"""

import mesh
import animate as anim
from sim_runner import Simulation

import numpy as np
import numpy.typing as npt
import math
import matplotlib.pyplot as plt
import pydec
from dataclasses import dataclass
from typing import Iterable


scatterer_radius = 1.0
outer_edge_radius = 3.0
cmp_mesh = mesh.annulus(scatterer_radius, outer_edge_radius, refine_count=1)
cmp_complex = pydec.SimplicialComplex(cmp_mesh)


# gather sets of inner and outer boundary edges
# for applying different boundary conditions to each
inner_bound_edges: list[pydec.Simplex] = []
outer_bound_edges: list[pydec.Simplex] = []
# distinguish between boundaries by checking squared distance from origin
# (TODO: this won't work for more complicated mesh shapes.
# can we label these somehow with gmsh instead?
# another option would be to use the orientation of the edge and triangle it's part of)
middle_dist_sq = ((outer_edge_radius + scatterer_radius) / 2.0) ** 2
for edge in cmp_complex.boundary():
    edge_verts = edge.boundary()
    test_vert = cmp_mesh.vertices[cmp_complex[0].simplex_to_index.get(edge_verts[0])]
    if np.dot(test_vert, test_vert) < middle_dist_sq and edge not in inner_bound_edges:
        inner_bound_edges.append(edge)
    elif edge not in outer_bound_edges:
        outer_bound_edges.append(edge)


# for each outer boundary edge, find the triangle this edge is part of
# and save some info for computing the absorbing boundary condition
@dataclass
class BoundaryEdgeInfo:
    dual_vert_idx: int
    length: float
    orientation: int


outer_bound_infos: list[BoundaryEdgeInfo] = []
for edge in outer_bound_edges:
    edge_idx = cmp_complex[1].simplex_to_index.get(edge)
    # find the triangle using the incidence matrix
    tri_indices = cmp_complex[1].d[:, edge_idx].nonzero()[0]
    assert len(tri_indices) == 1, "boundary edge is part of one triangle only"
    edge_ends = [
        cmp_complex.vertices[cmp_complex[0].simplex_to_index.get(v)]
        for v in edge.boundary()
    ]
    outer_bound_infos.append(
        BoundaryEdgeInfo(
            dual_vert_idx=tri_indices[0],
            orientation=cmp_complex[1].d[tri_indices[0], edge_idx],
            length=np.linalg.norm(edge_ends[1] - edge_ends[0]),
        )
    )


class CircleScatterer(Simulation):
    def __init__(self):
        # incoming wave parameters
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
        # time needed for incoming wave evaluation
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
        # incoming wave on the scatterer's surface
        for bound_edge in inner_bound_edges:
            edge_idx = cmp_complex[1].simplex_to_index.get(bound_edge)
            self.q[edge_idx] = self._eval_inc_wave_flux(
                t_at_w,
                [
                    self.complex[0].simplex_to_index.get(v)
                    for v in bound_edge.boundary()
                ],
            )
        # absorbing outer boundary condition
        for bound_edge, edge_info in zip(outer_bound_edges, outer_bound_infos):
            edge_idx = cmp_complex[1].simplex_to_index.get(bound_edge)
            self.q[edge_idx] = (
                -self.v[edge_info.dual_vert_idx]
                * edge_info.length
                * edge_info.orientation
            )

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


sim = CircleScatterer()
vis = anim.FluxAndPressure(sim=sim, arrow_scale=20)
vis.show()
# vis.save()

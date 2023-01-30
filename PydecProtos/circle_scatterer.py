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

scatterer_radius = 1.0
outer_edge_radius = 3.0

cmp_mesh = mesh.annulus(scatterer_radius, outer_edge_radius, refine_count=1)
cmp_complex = pydec.SimplicialComplex(cmp_mesh)
plt.triplot(cmp_mesh.vertices[:, 0], cmp_mesh.vertices[:, 1], cmp_mesh.indices)
plt.show()


# gather sets of inner and outer boundary edges & vertices
# for applying different boundary conditions to each
inner_bound_verts = []
inner_bound_edges = []
outer_bound_verts = []
outer_bound_edges = []
# distinguish between boundaries by checking squared distance from origin
# (TODO: can we label these somehow with gmsh instead?
# this won't work for more complicated mesh shapes)
middle_dist_sq = ((outer_edge_radius + scatterer_radius) / 2.0) ** 2
for edge in cmp_complex.boundary():
    edge_verts = edge.boundary()
    test_vert = cmp_mesh.vertices[cmp_complex[0].simplex_to_index.get(edge_verts[0])]
    if np.dot(test_vert, test_vert) < middle_dist_sq:
        inner_bound_edges.append(edge)
        inner_bound_verts.extend(edge_verts)
    else:
        outer_bound_edges.append(edge)
        outer_bound_verts.extend(edge_verts)
# remove duplicates by constructing sets
inner_bound_verts = set(inner_bound_verts)
inner_bound_edges = set(inner_bound_edges)
outer_bound_verts = set(outer_bound_verts)
outer_bound_edges = set(outer_bound_edges)


class CircleScatterer(Simulation):
    def __init__(self):
        # incoming wave parameters
        inc_wavenumber = 1.0
        inc_wave_dir = np.array([0.0, 1.0])
        self.inc_wave_vector = inc_wavenumber * inc_wave_dir
        self.inc_angular_vel = 3.0

        # time parameters
        sim_time = 2.0 * np.pi
        dt = 1.0 / 20.0
        step_count = math.ceil(sim_time / dt)

        # time stepping matrices
        self.v_step_mat = (
            -dt * cmp_complex[0].star_inv * cmp_complex[0].d.T * cmp_complex[1].star
        )
        self.w_step_mat = dt * cmp_complex[0].d

        super().__init__(mesh=cmp_mesh, dt=dt, step_count=step_count)

    def init_state(self):
        # time needed for incoming wave evaluation
        self.t = 0.0

        self.v = np.zeros(cmp_complex[0].num_simplices)
        for vert_idx in range(len(self.v)):
            # time derivative of the incoming wave
            self.v[vert_idx] = self.inc_angular_vel * math.sin(
                -np.dot(self.inc_wave_vector, self.mesh.vertices[vert_idx, :])
            )
        self.w = np.zeros(cmp_complex[1].num_simplices)
        for edge_idx in range(len(self.w)):
            edge = cmp_complex[1].simplices[edge_idx]
            self.w[edge_idx] = self._eval_inc_wave_w(0.5 * self.dt, edge)

    def step(self):
        self.t += self.dt
        self.v += self.v_step_mat * self.w
        # inner edge boundary condition (i.e. incoming wave) for v
        for bound_vert in inner_bound_verts:
            vert_idx = cmp_complex[0].simplex_to_index.get(bound_vert)
            self.v[vert_idx] = self.inc_angular_vel * math.sin(
                self.inc_angular_vel * self.t
                - np.dot(self.inc_wave_vector, cmp_mesh.vertices[vert_idx])
            )
        # w is computed at a time instance offset by half dt
        t_at_w = self.t + 0.5 * self.dt
        self.w += self.w_step_mat * self.v
        # inner edge boundary condition for w
        for bound_edge in inner_bound_edges:
            edge_idx = cmp_complex[1].simplex_to_index.get(bound_edge)
            self.w[edge_idx] = self._eval_inc_wave_w(t_at_w, bound_edge.boundary())

        # TODO: outer absorbing boundary condition

    def get_z_data(self):
        # visualizing acoustic pressure
        return self.v

    def _eval_inc_wave_w(self, t: float, edge_verts: list[npt.NDArray]) -> float:
        """Evaluate the line integral of the particle velocity of the incoming wave
        over an edge of the mesh, in other words compute a value of `w` from the wave."""

        p1 = self.mesh.vertices[edge_verts[0]]
        p2 = self.mesh.vertices[edge_verts[1]]
        kdotp = np.dot(self.inc_wave_vector, p1)
        kdotl = np.dot(self.inc_wave_vector, p2 - p1)
        angle = self.inc_angular_vel * t
        return math.cos(angle - kdotp) - math.cos(angle - kdotp - kdotl)


sim = CircleScatterer()
vis = anim.ZeroForm(sim=sim, get_data=lambda s: s.v, zlim=[-8.0, 8.0])
vis.show()
# vis.save_gif()

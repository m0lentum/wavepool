"""
Wave equation (see membrane.py doc comment).
Square domain with known plane wave solution
imposed on the edge as Dirichlet boundary condition.
Examining error compared to known solution.
"""

import mesh
from sim_runner import Simulation

import numpy as np
import numpy.typing as npt
import math
import matplotlib.pyplot as plt
import pydec

# mesh parameters
mesh_dim = np.pi
verts_per_side = 20
mesh_scale = mesh_dim / verts_per_side
cmp_mesh = mesh.rect_unstructured(mesh_dim, mesh_dim, mesh_scale)
cmp_complex = pydec.SimplicialComplex(cmp_mesh)

# time parameters
sim_time = 2.0 * np.pi
dt = 1.0 / 20.0
step_count = math.ceil(sim_time / dt)

# incoming wave parameters
inc_wavenumber = 2.0
inc_wave_dir = np.array([0.0, 1.0])
inc_wave_vector = inc_wavenumber * inc_wave_dir
inc_angular_vel = 2.0

# gather boundary elements
bound_verts = []
bound_edges = []
for edge in cmp_complex.boundary():
    edge_verts = edge.boundary()
    bound_edges.append(edge)
    bound_verts.extend(edge_verts)
# remove duplicates by constructing sets
bound_verts = set(bound_verts)
bound_edges = set(bound_edges)


class AccuracyTest(Simulation):
    def __init__(self):
        # time stepping matrices
        self.v_step_mat = (
            -dt * cmp_complex[0].star_inv * cmp_complex[0].d.T * cmp_complex[1].star
        )
        self.w_step_mat = dt * cmp_complex[0].d

        super().__init__(mesh=cmp_mesh, dt=dt, step_count=step_count, zlim=[-8.0, 8.0])

    def init_state(self):
        # time needed for incoming wave evaluation
        self.t = 0.0

        self.v = np.zeros(cmp_complex[0].num_simplices)
        for vert_idx in range(len(self.v)):
            # time derivative of the incoming wave
            self.v[vert_idx] = inc_angular_vel * math.sin(
                -np.dot(inc_wave_vector, self.mesh.vertices[vert_idx, :])
            )
        self.w = np.zeros(cmp_complex[1].num_simplices)
        for edge_idx in range(len(self.w)):
            edge = cmp_complex[1].simplices[edge_idx]
            self.w[edge_idx] = self._eval_inc_wave_w(0.5 * self.dt, edge)

    def step(self):
        self.t += self.dt
        self.v += self.v_step_mat * self.w
        # plane wave at the boundary for v
        for bound_vert in bound_verts:
            vert_idx = cmp_complex[0].simplex_to_index.get(bound_vert)
            self.v[vert_idx] = inc_angular_vel * math.sin(
                inc_angular_vel * self.t
                - np.dot(inc_wave_vector, cmp_mesh.vertices[vert_idx])
            )
        # w is computed at a time instance offset by half dt
        t_at_w = self.t + 0.5 * self.dt
        self.w += self.w_step_mat * self.v
        # plane wave at the boundary for w
        for bound_edge in bound_edges:
            edge_idx = cmp_complex[1].simplex_to_index.get(bound_edge)
            self.w[edge_idx] = self._eval_inc_wave_w(t_at_w, bound_edge.boundary())

    def get_z_data(self):
        # visualizing acoustic pressure
        return self.v

    def _eval_inc_wave_w(self, t: float, edge_verts: list[npt.NDArray]) -> float:
        """Evaluate the line integral of the particle velocity of the incoming wave
        over an edge of the mesh, in other words compute a value of `w` from the wave."""

        p1 = self.mesh.vertices[edge_verts[0]]
        p2 = self.mesh.vertices[edge_verts[1]]
        kdotp = np.dot(inc_wave_vector, p1)
        kdotl = np.dot(inc_wave_vector, p2 - p1)
        angle = inc_angular_vel * t
        return math.cos(angle - kdotp) - math.cos(angle - kdotp - kdotl)


sim = AccuracyTest()
sim.show()
# sim.save_mp4()

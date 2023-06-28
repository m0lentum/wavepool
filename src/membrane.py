from utils import mesh
from utils import animate as anim
from utils.sim_runner import Simulation

import numpy as np

mesh_dim = np.pi
verts_per_side = 20
mesh_scale = mesh_dim / verts_per_side
cmp_mesh = mesh.rect_unstructured(mesh_dim, mesh_dim, mesh_scale)
cmp_complex = cmp_mesh.complex


class StandingWave(Simulation):
    def __init__(self):
        # time stepping

        steps_per_unit_time = 20
        dt = 1.0 / float(steps_per_unit_time)
        sim_time_units = 6
        step_count = steps_per_unit_time * sim_time_units

        # multiplier matrix for w in the time-stepping formula for v
        v_step_mat = (
            -dt * cmp_complex[0].star_inv * cmp_complex[0].d.T * cmp_complex[1].star
        )
        # enforce boundary condition by setting rows of this matrix to 0 for boundary vertices
        for bound_edge in cmp_complex.boundary():
            for bound_vert in bound_edge.boundary():
                vert_idx = cmp_complex[0].simplex_to_index.get(bound_vert)
                v_step_mat[vert_idx, :] = 0.0
        # multiplier matrix for v in the time-stepping formula for w
        w_step_mat = dt * cmp_complex[0].d

        self.v_step_mat = v_step_mat
        self.w_step_mat = w_step_mat
        super().__init__(complex=cmp_complex, dt=dt, step_count=step_count)

    def init_state(self):
        x_wave_count = 2
        y_wave_count = 3
        self.v = np.sin(x_wave_count * self.complex.vertices[:, 0]) * np.sin(
            y_wave_count * self.complex.vertices[:, 1]
        )
        # w is vector-valued, but represented in DEC as a scalar per mesh edge,
        # therefore a single scalar per edge here
        self.w = np.zeros(cmp_complex[1].num_simplices)

    def step(self):
        self.v += self.v_step_mat * self.w
        self.w += self.w_step_mat * self.v


# another simulation with the same equations but different initial conditions
class MovingWave(StandingWave):
    def init_state(self):
        x = self.complex.vertices[:, 0]
        y = self.complex.vertices[:, 1]
        self.v = np.multiply(0.2 * np.multiply(x, x), np.sin(3 * x)) * np.sin(y)
        self.w = np.zeros(cmp_complex[1].num_simplices)


sim1 = StandingWave()
sim1_vis = anim.PressureOnVertices(sim=sim1, get_data=lambda s: s.v, zlim=[-1.5, 1.5])
sim1_vis.show()

sim2 = MovingWave()
sim2_vis = anim.PressureOnVertices(sim=sim2, get_data=lambda s: s.v, zlim=[-1.5, 1.5])
sim2_vis.show()
# sim2_vis.save()

"""
Vibrating rectangular membrane simulated with leapfrog time discretization
and Dirichlet boundary condition u = 0.

Wave equation ∂²u/∂t² - c²∇²u = f
represented as a first-order system of differential form equations
with v = ∂u/∂t represented by a 0-form on the primal mesh
and w = ∇u represented by a 1-form on the primal mesh

(TODO: add a .md note to explain in more detail)
"""

import mesh

import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse
import pydec

# mesh setup

mesh_dim = np.pi
verts_per_side = 10
mesh_scale = mesh_dim / verts_per_side
cmp_mesh = mesh.rect_unstructured(mesh_dim, mesh_dim, mesh_scale)
cmp_complex = cmp_mesh.as_pydec_complex()

# TODO: initial condition should maybe be specified on u
# and then computed from that for v and w? look at papers, how do they do this?

# initial condition v = sin(x)sin(2y), w = 0
v = np.sin(cmp_mesh.vertices[:, 0]) * np.sin(2 * cmp_mesh.vertices[:, 1])
w = np.zeros(cmp_complex[1].num_simplices)

# time stepping

steps_per_unit_time = 10
dt = 1.0 / steps_per_unit_time
sim_time_units = 5
step_count = steps_per_unit_time * sim_time_units

# TODO: add explicit wavenumber, currently it's implicitly 1
# TODO: can we incorporate the boundary conditions to these matrices?

# multiplier matrix for w^n in the time-stepping formula for v
v_step_mat = dt * cmp_complex[0].star_inv * cmp_complex[1].d.T * cmp_complex[1].star
# multiplier matrix for v^{n+1/2} in the time-stepping formula for w
w_step_mat = dt * cmp_complex[0].d

for step_idx in range(step_count):
    # TODO: v and w here are v^n and w^{n+1/2}
    # but the step matrices should be applied to
    # v^{n+1/2} and w^n, how do we get those
    v += v_step_mat * w
    w += w_step_mat * v

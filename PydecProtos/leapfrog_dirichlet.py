"""
Vibrating rectangular membrane simulated with leapfrog time discretization
and Dirichlet boundary condition u = 0.

Wave equation ∂²u/∂t² - c²∇²u = f
represented as a first-order system of differential form equations
with v = ∂u/∂t represented by a 0-form on the primal mesh
and w = ∇u represented by a 1-form on the primal mesh.
The boundary condition u = 0 implies v = 0, but w can vary on the boundary.

(TODO: add a .md note to explain in more detail)
"""

import mesh

import functools
import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np

# mesh setup

mesh_dim = np.pi
verts_per_side = 20
mesh_scale = mesh_dim / verts_per_side
cmp_mesh = mesh.rect_unstructured(mesh_dim, mesh_dim, mesh_scale)
cmp_complex = cmp_mesh.as_pydec_complex()

# TODO: initial condition should maybe be specified on u
# and then computed from that for v and w? look at papers, how do they do this?

# initial condition v = sin(nx)sin(my), w = 0
x_wave_count = 2
y_wave_count = 3
v = np.sin(x_wave_count * cmp_mesh.vertices[:, 0]) * np.sin(
    y_wave_count * cmp_mesh.vertices[:, 1]
)
w = np.zeros(cmp_complex[1].num_simplices)

# time stepping

steps_per_unit_time = 20
dt = 1.0 / float(steps_per_unit_time)
sim_time_units = 6
step_count = steps_per_unit_time * sim_time_units

# TODO: add explicit wavenumber, currently it's implicitly 1

# multiplier matrix for w in the time-stepping formula for v
v_step_mat = -dt * cmp_complex[0].star_inv * cmp_complex[0].d.T * cmp_complex[1].star
# enforce boundary condition by setting rows of this matrix to 0 for boundary vertices
edge_vert_indices = []  # store edge vertex indices for debugging
for bound_edge in cmp_complex.boundary():
    for bound_vert in bound_edge.boundary():
        vert_idx = cmp_complex[0].simplex_to_index.get(bound_vert)
        v_step_mat[vert_idx, :] = 0.0
        edge_vert_indices.append(vert_idx)
edge_vert_indices = set(edge_vert_indices)
# multiplier matrix for v in the time-stepping formula for w
w_step_mat = dt * cmp_complex[0].d

# setup animated drawing

fig = plt.figure(layout="tight")
ax = fig.add_subplot(1, 1, 1, projection="3d")


# setup drawing and time stepping


def draw_mesh(z_data):
    ax.clear()
    ax.set_zlim([-1, 1])
    return ax.plot_trisurf(
        cmp_mesh.vertices[:, 0],
        cmp_mesh.vertices[:, 1],
        z_data,
        cmap="viridis",
        edgecolor="none",
    )


def step_and_draw(step_idx, v, w):
    # TODO: add source term to v
    v += v_step_mat * w
    w += w_step_mat * v

    return draw_mesh(v)


anim = matplotlib.animation.FuncAnimation(
    fig=fig,
    func=functools.partial(step_and_draw, v=v, w=w),
    frames=step_count,
    interval=int(1000 * dt),
)

# TODO: CLI argument for this
save_gif = False
if save_gif:
    print("Saving gif. This takes a while")
    anim.save("result.gif", writer="imagemagick", fps=steps_per_unit_time)

plt.show()

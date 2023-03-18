"""
Same problem as scatterer_forward.py,
but using the exact controllability method
to find a time-harmonic solution.
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


#
# mesh generation
#

# this part is copied verbatim from scatterer_forward.py.
# TODO: perhaps make a module that could generate test cases like this
# with different shapes for the scatterer object?

scatterer_radius = 1.0
outer_edge_radius = 3.0
cmp_mesh = mesh.annulus(scatterer_radius, outer_edge_radius, refine_count=1)
cmp_complex = pydec.SimplicialComplex(cmp_mesh)


# gather sets of inner and outer boundary edges
# for applying different boundary conditions to each
inner_bound_edges: list[pydec.Simplex] = []
outer_bound_edges: list[pydec.Simplex] = []
# distinguish between boundaries by checking squared distance from origin
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


#
# simulation parameters and helpers
#

# incoming wave parameters
inc_wavenumber = 1.0
inc_wave_dir = np.array([0.0, 1.0])
inc_wave_vector = inc_wavenumber * inc_wave_dir
# angular velocity of the wave in radians per second
inc_angular_vel = 2.0

# time parameters

# since we're looking for a time-periodic solution,
# it's important the simulated time range
# coincides with the period the incoming wave
time_period = (2.0 * np.pi) / inc_angular_vel
dt = np.pi / 60.0
step_count = math.ceil(time_period / dt)

# time stepping matrices
p_step_mat = dt * cmp_complex[2].star * cmp_complex[1].d
q_step_mat = dt * cmp_complex[1].star_inv * cmp_complex[1].d.T


# utilities for computing the incoming wave
def eval_inc_wave_pressure(t, position: npt.NDArray) -> float:
    """Evaluate the value of v for the incoming plane wave at a point."""

    return inc_angular_vel * math.sin(
        inc_angular_vel * t - np.dot(inc_wave_vector, position)
    )


def eval_inc_wave_flux(t: float, edge_vert_indices: Iterable[int]) -> float:
    """Evaluate the line integral of the area flux of the incoming wave
    over an edge of the mesh, in other words compute a value of `q` from the wave."""

    p = [cmp_complex.vertices[v] for v in edge_vert_indices]
    kdotp = np.dot(inc_wave_vector, p[0])
    l = p[1] - p[0]
    kdotl = np.dot(inc_wave_vector, l)
    kdotn = np.dot(inc_wave_vector, np.array([l[1], -l[0]]))
    wave_angle = inc_angular_vel * t

    if abs(kdotl) < 1e-5:
        return -kdotn * math.sin(wave_angle - kdotp)
    else:
        return (kdotn / kdotl) * (
            math.cos(wave_angle - kdotp) - math.cos(wave_angle - kdotp - kdotl)
        )


@dataclass
class State:
    pressure: npt.NDArray[np.float64]
    flux: npt.NDArray[np.float64]

    def copy(self):
        return State(pressure=self.pressure.copy(), flux=self.flux.copy())

    def __add__(self, other):
        return State(
            pressure=self.pressure + other.pressure, flux=self.flux + other.flux
        )

    def __sub__(self, other):
        return State(
            pressure=self.pressure - other.pressure, flux=self.flux - other.flux
        )


#
# simulation solver
#


@dataclass
class Solver:
    state: State
    t: float

    def step_forward(self):
        """Solve one timestep in the forward equation."""

        self.t += dt
        self.state.pressure += p_step_mat * self.state.flux
        # q is computed at a time instance offset by half dt
        t_at_w = self.t + 0.5 * dt
        self.state.flux += q_step_mat * self.state.pressure
        # incoming wave on the scatterer's surface
        # (TODO: Mur transition)
        for bound_edge in inner_bound_edges:
            edge_idx = cmp_complex[1].simplex_to_index.get(bound_edge)
            self.state.flux[edge_idx] = eval_inc_wave_flux(
                t_at_w,
                [cmp_complex[0].simplex_to_index.get(v) for v in bound_edge.boundary()],
            )
        # absorbing outer boundary condition
        for bound_edge, edge_info in zip(outer_bound_edges, outer_bound_infos):
            edge_idx = cmp_complex[1].simplex_to_index.get(bound_edge)
            self.state.flux[edge_idx] = (
                -self.state.pressure[edge_info.dual_vert_idx]
                * edge_info.length
                * edge_info.orientation
            )

    def step_backward(self):
        """Solve one timestep in the backward equation."""

        self.t -= dt
        self.state.flux += p_step_mat.T * self.state.pressure
        # inner Dirichlet boundary without source term
        for bound_edge in inner_bound_edges:
            edge_idx = cmp_complex[1].simplex_to_index.get(bound_edge)
            self.state.flux[edge_idx] = 0.0
        # absorbing outer boundary
        # with flipped sign due to going backward in time
        for bound_edge, edge_info in zip(outer_bound_edges, outer_bound_infos):
            edge_idx = cmp_complex[1].simplex_to_index.get(bound_edge)
            self.state.flux[edge_idx] = (
                self.state.pressure[edge_info.dual_vert_idx]
                * edge_info.length
                * edge_info.orientation
            )

        self.state.pressure += q_step_mat.T * self.state.flux


zero_state = State(
    pressure=np.zeros(cmp_complex[2].num_simplices),
    flux=np.zeros(cmp_complex[1].num_simplices),
)

# TODO: run Mur transition here


def compute_gradient(initial_state: State) -> State:
    """Compute the gradient of the cost function
    with respect to the given initial values."""

    # solve the forward equation
    sim_fwd = Solver(state=initial_state.copy(), t=0.0)
    for _ in range(step_count):
        sim_fwd.step_forward()
    final_state = sim_fwd.state

    # compute starting value for the backward equation
    fwd_diff = final_state - initial_state
    bwd_init_q = -fwd_diff.flux
    bwd_init_state = State(
        flux=bwd_init_q,
        pressure=(q_step_mat.T * bwd_init_q) - fwd_diff.pressure,
    )

    # solve the backward equation
    sim_bwd = Solver(state=bwd_init_state, t=time_period)
    for _ in range(step_count - 1):
        sim_bwd.step_backward()
    final_bwd_state = sim_bwd.state

    return final_bwd_state - fwd_diff


grad = compute_gradient(zero_state)
print(grad)

# TODO: conjugate gradient optimization

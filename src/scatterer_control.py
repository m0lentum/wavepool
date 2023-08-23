from utils import mesh
from utils import measure_mesh

import argparse
import numpy as np
import numpy.typing as npt
import math
import matplotlib.animation as plt_anim
import matplotlib.pyplot as plt
import pydec
import scipy.sparse as sps
from dataclasses import dataclass
from typing import Callable, Iterable

# figure size with a good font size for thesis embedding
plt.rcParams["figure.figsize"] = [5, 5]

#
# command line parameters
#

arg_parser = argparse.ArgumentParser(prog="scatterer_control")
arg_parser.add_argument(
    "--shape",
    choices=["square", "star", "diamonds", "diamonds_bad"],
    default="square",
    help="shape of the scatterer object",
)
arg_parser.add_argument(
    "--star-points",
    dest="star_points",
    type=int,
    default=8,
    help="number of points in the star, only effective if --shape star is also given",
)
arg_parser.add_argument(
    "--mesh-scaling",
    dest="mesh_scaling",
    type=float,
    default=1.0,
    help="number to divide default mesh element size by",
)
arg_parser.add_argument(
    "--max-iters",
    dest="max_iters",
    type=int,
    default=50,
    help="maximum iteration count for the conjugate gradient algorithm",
)
arg_parser.add_argument(
    "--inc-angle",
    dest="inc_angle",
    type=float,
    default=90.0,
    help="angle of the incident wave's propagating direction in degrees",
)
arg_parser.add_argument(
    "--wavenumber",
    dest="wavenumber",
    type=float,
    default=1.0,
    help="wavenumber (inverse of wavelength) of the incident wave",
)
arg_parser.add_argument(
    "--no-inc-wave",
    dest="no_inc_wave",
    action="store_true",
    help="hide the incident wave in the animated visualization",
)
arg_parser.add_argument(
    "--save-visuals",
    dest="save_visuals",
    action="store_true",
    help="save still images used in the thesis as .pdf and animated visualization as .gif",
)
args = arg_parser.parse_args()

#
# mesh generation
#

if args.shape == "square":
    cmp_mesh = mesh.square_with_hole(
        outer_extent=np.pi * 2.0,
        inner_extent=np.pi / 3.0,
        elem_size=np.pi / (3.0 * args.mesh_scaling),
    )
elif args.shape == "star":
    cmp_mesh = mesh.star(
        point_count=args.star_points,
        inner_r=np.pi / 3.0,
        outer_r=np.pi,
        domain_r=np.pi * 2.0,
        elem_size=np.pi / (3.0 * args.mesh_scaling),
    )
elif args.shape == "diamonds" or args.shape == "diamonds_bad":
    cmp_mesh = mesh.diamond_lattice(
        domain_radius=np.pi * 2.0,
        horizontal_divs=4,
        vertical_divs=2,
        # diamonds_bad has gap size smaller than element size,
        # making narrow triangles that cause instability
        gap_size=np.pi / 3.0 if args.shape != "diamonds_bad" else np.pi / 6.0,
        elem_size=np.pi / (3.0 * args.mesh_scaling),
    )
else:
    # unreachable because argparse will throw an error,
    # exit to satisfy pyright's type checking
    exit()
cmp_complex = cmp_mesh.complex
inner_bound_edges: list[int] = cmp_mesh.edge_groups["inner boundary"]
outer_bound_edges: list[int] = cmp_mesh.edge_groups["outer boundary"]

measure_mesh.print_measurements(cmp_complex)
print("")


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
            length=float(np.linalg.norm(edge_ends[1] - edge_ends[0])),
        )
    )

#
# simulation parameters and helpers
#

# incident wave parameters
inc_wavenumber = args.wavenumber
wave_speed = 1.0
inc_angle = args.inc_angle * 2.0 * np.pi / 360.0
inc_wave_dir = np.array([math.cos(inc_angle), math.sin(inc_angle)])
inc_wave_vector = inc_wavenumber * inc_wave_dir
# angular velocity of the wave in radians per second
inc_angular_vel = inc_wavenumber * wave_speed

# time parameters

# since we're looking for a time-periodic solution,
# it's important the simulated time range
# coincides with the period the incident wave
wave_period = (2.0 * np.pi) / inc_angular_vel
dt = 0.1 * min(cmp_complex[1].primal_volume)
steps_per_period = math.ceil(wave_period / dt)

# time stepping matrices

p_step_mat_yee = dt * wave_speed**2 * cmp_complex[2].star * cmp_complex[1].d
q_step_mat_yee = dt * cmp_complex[1].star_inv * cmp_complex[1].d.T

# harmonic timestepping

print("Computing harmonic timestep operators...")

wavenumber_sq = inc_wavenumber**2
star_1_inv_diag = cmp_complex[1].star_inv.diagonal()
for edge_idx, (edge_len, dual_edge_len) in enumerate(
    zip(cmp_complex[1].primal_volume, cmp_complex[1].dual_volume)
):
    edge_curv = edge_len**2 * wavenumber_sq
    dual_curv = dual_edge_len**2 * wavenumber_sq
    star_1_inv_diag[edge_idx] /= (1.0 - dual_curv / 16.0) / (
        1.0 - edge_curv / 94.0 - dual_curv / 32.0
    )
star_2_diag = cmp_complex[2].star.diagonal()
for face_idx, face in enumerate(cmp_complex[2].simplices):
    center = cmp_complex[2].circumcenter[face_idx]
    verts = [cmp_complex.vertices[i] for i in face]
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

star_1_inv = sps.diags(star_1_inv_diag)
star_2 = sps.diags(star_2_diag)

harmonic_dt = (2.0 / inc_angular_vel) * math.sin(inc_angular_vel * dt / 2.0)
p_step_mat_har = harmonic_dt * wave_speed**2 * star_2 * cmp_complex[1].d
q_step_mat_har = harmonic_dt * star_1_inv * cmp_complex[1].d.T

# using global state to set which timestep matrices are active.
# TODO: this is super ugly, this (and the rest of the global variables)
# should be encapsulated and parameterized better
p_step_mat = p_step_mat_har
q_step_mat = q_step_mat_har


# utilities for computing the incident wave
def eval_inc_wave_pressure(t, position: npt.NDArray) -> float:
    """Evaluate the value of v for the incident plane wave at a point."""

    return inc_angular_vel * math.sin(
        inc_angular_vel * t - np.dot(inc_wave_vector, position)
    )


def eval_inc_wave_flux(t: float, edge_vert_indices: Iterable[int]) -> float:
    """Evaluate the line integral of the area flux of the incident wave
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
    """State vector with named parts for convenience
    and methods for visualization."""

    pressure: npt.NDArray[np.float64] = np.zeros(cmp_complex[2].num_simplices)
    flux: npt.NDArray[np.float64] = np.zeros(cmp_complex[1].num_simplices)

    def copy(self):
        return State(pressure=self.pressure.copy(), flux=self.flux.copy())

    def scaled(self, s: float):
        return State(pressure=s * self.pressure, flux=s * self.flux)

    def dot(self, other) -> float:
        return np.dot(self.pressure, other.pressure) + np.dot(self.flux, other.flux)

    def energy(self) -> float:
        """Control energy of the exact controllability problem.
        `self` is assumed to be the difference
        between a wave period's beginning state and end state."""
        return 0.5 * self.dot(self)

    def norm(self) -> float:
        """Euclidean norm of the entire state vector."""
        return math.sqrt(self.dot(self))

    def __add__(self, other):
        return State(
            pressure=self.pressure + other.pressure, flux=self.flux + other.flux
        )

    def __sub__(self, other):
        return State(
            pressure=self.pressure - other.pressure, flux=self.flux - other.flux
        )

    def __neg__(self):
        return State(pressure=-self.pressure, flux=-self.flux)

    def draw(self, save: bool = False):
        """Draw a still image of the state.
        Remember to also call `plt.show()` after."""
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        self._draw(ax)
        if save:
            fig.savefig("state.pdf")

    def _draw(
        self,
        ax: plt.Axes,
        vlims: list[int] = [-1, 1],
        draw_velocity: bool = True,
    ):
        ax.tripcolor(
            cmp_complex.vertices[:, 0],
            cmp_complex.vertices[:, 1],
            triangles=cmp_complex.simplices,
            facecolors=self.pressure,
            edgecolors="k",
            # scale vlims and velocity arrows by wavenumber
            # because the range of the variables depends on it
            vmin=inc_wavenumber * vlims[0],
            vmax=inc_wavenumber * vlims[1],
        )
        if draw_velocity:
            barys, arrows = pydec.simplex_quivers(cmp_complex, self.flux)
            arrows = np.vstack((arrows[:, 1], -arrows[:, 0])).T
            ax.quiver(
                barys[:, 0],
                barys[:, 1],
                arrows[:, 0],
                arrows[:, 1],
                units="dots",
                width=1,
                scale=inc_wavenumber / 15.0,
            )

    def save_anim(
        self,
        filename: str = "solution.gif",
        size: list[int] = [6, 6],
        with_incident_wave: bool = True,
    ):
        """Save a nice-looking visualization of the solution
        with the incident wave added and velocity arrows removed.
        Good for social media posting!"""

        sim_fwd = ForwardSolve(state=self.copy())
        fig = plt.figure(figsize=size)
        ax = fig.add_subplot(1, 1, 1)

        def step(_):
            ax.clear()
            sim_fwd.step()
            if with_incident_wave:
                inc_wave = eval_inc_wave_everywhere(sim_fwd.t)
                vis_wave = inc_wave.scaled(0.5) + sim_fwd.state
            else:
                vis_wave = sim_fwd.state
            vis_wave._draw(ax, vlims=[-2, 2], draw_velocity=False)

        anim = plt_anim.FuncAnimation(
            fig=fig,
            init_func=sim_fwd.state._draw(ax),
            func=step,
            frames=steps_per_period,
            interval=int(1000 * dt),
        )

        print(f"Saving {filename}. This takes a while")
        writer = plt_anim.FFMpegWriter(fps=int(1.0 / dt))
        anim.save(filename, writer)


def eval_inc_wave_everywhere(t: float) -> State:
    """Evaluate the incident plane wave on every feature of the mesh.
    Used to add the incident wave to the final visualization."""

    state = State()
    for vert_idx in range(len(state.pressure)):
        state.pressure[vert_idx] = eval_inc_wave_pressure(
            t, cmp_complex[2].circumcenter[vert_idx]
        )
    for edge_idx in range(len(state.flux)):
        if edge_idx in inner_bound_edges:
            continue
        state.flux[edge_idx] = eval_inc_wave_flux(
            t + 0.5 * dt, cmp_complex[1].simplices[edge_idx]
        )
    return state


#
# simulation solver
#


@dataclass
class ForwardSolve:
    state: State
    t: float = 0.0

    def step(self, source_term_scaling: Callable[[float], float] = lambda _: 1.0):
        """Solve one timestep in the forward equation."""

        self.t += dt
        self.state.pressure += p_step_mat @ self.state.flux
        # q is computed at a time instance offset by half dt
        t_at_w = self.t + 0.5 * dt
        self.state.flux += q_step_mat @ self.state.pressure
        # incident wave on the scatterer's surface
        for edge_idx in inner_bound_edges:
            self.state.flux[edge_idx] = source_term_scaling(
                t_at_w
            ) * eval_inc_wave_flux(t_at_w, cmp_complex[1].simplices[edge_idx])

        # absorbing outer boundary condition
        for edge_idx, edge_info in zip(outer_bound_edges, outer_bound_infos):
            self.state.flux[edge_idx] = (
                -self.state.pressure[edge_info.dual_vert_idx]
                * edge_info.length
                * edge_info.orientation
            )


@dataclass
class BackwardSolve:
    state: State

    def step(self):
        """Solve one timestep in the backward equation."""

        self.state.flux += p_step_mat.T @ self.state.pressure
        # inner Dirichlet boundary without source term
        for edge_idx in inner_bound_edges:
            self.state.flux[edge_idx] = 0.0
        # absorbing outer boundary
        # with flipped sign due to going backward in time
        for edge_idx, edge_info in zip(outer_bound_edges, outer_bound_infos):
            self.state.flux[edge_idx] = (
                self.state.pressure[edge_info.dual_vert_idx]
                * edge_info.length
                * edge_info.orientation
            )

        self.state.pressure += q_step_mat.T @ self.state.flux


@dataclass
class GradientResult:
    """Gradient of the cost function and useful related measurements."""

    gradient: State
    # difference between the initial and final states of the forward simulation
    forward_diff: State

    def energy(self) -> float:
        return self.forward_diff.energy()


def compute_cost_gradient(
    initial_state: State, use_source_terms: bool = True
) -> GradientResult:
    """Compute the gradient of the cost function
    with respect to the given initial values
    using the adjoint state method.

    If use_source_terms == True, corresponds to computing Ax - b
    in the linear system `grad J = Ax - b = 0`.
    Else, corresponds to computing Ax in the same system."""

    # solve the forward equation
    sim_fwd = ForwardSolve(state=initial_state.copy())
    source_scaling = (lambda _: 1.0) if use_source_terms else (lambda _: 0.0)
    for _ in range(steps_per_period):
        sim_fwd.step(source_term_scaling=source_scaling)
    final_state = sim_fwd.state

    # compute starting value for the backward equation
    fwd_diff = final_state - initial_state
    bwd_init_q = -fwd_diff.flux
    bwd_init_state = State(
        flux=bwd_init_q,
        pressure=(q_step_mat.T @ bwd_init_q) - fwd_diff.pressure,
    )

    # solve the backward equation
    sim_bwd = BackwardSolve(state=bwd_init_state.copy())
    for _ in range(steps_per_period - 1):
        sim_bwd.step()
    final_bwd_state = State(
        pressure=-sim_bwd.state.pressure,
        flux=-sim_bwd.state.flux - p_step_mat.T @ sim_bwd.state.pressure,
    )

    return GradientResult(
        gradient=final_bwd_state - fwd_diff,
        forward_diff=fwd_diff,
    )


def compute_control_energy(state: State) -> float:
    """Compute the control energy of a given initial state."""

    period_sim = ForwardSolve(state.copy())
    for _ in range(steps_per_period):
        period_sim.step()
    return (period_sim.state - state).energy()


#
# running the simulation
#


@dataclass
class SolveResults:
    """Result of a simulation, plus some measurements taken in the process."""

    # resulting time-harmonic (if correct) state
    state: State
    # number of iterations it took to reach the controllability solution
    step_count: int
    # energy of the exact controllability solution over time
    control_energies: list[float]
    # energy of the forward solution over time
    forward_energies: list[float]
    # residual norm
    resid_norms: list[float]
    # A-inner product between search directions.
    # this is supposed to be zero at all times,
    # but may not be if the mesh isn't good enough
    a_inner_prods: list[float]


def solve():
    zero_state = State(
        pressure=np.zeros(cmp_complex[2].num_simplices),
        flux=np.zeros(cmp_complex[1].num_simplices),
    )

    # ease in the source terms to obtain smooth initial values for optimization

    transition_time = 5 * wave_period
    transition_step_count = math.ceil(transition_time / dt)
    transition_sim = ForwardSolve(state=zero_state.copy())

    def easing(t: float) -> float:
        sin_val = math.sin((t / transition_time) * (np.pi / 2.0))
        return (2.0 - sin_val) * sin_val

    for _ in range(transition_step_count):
        transition_sim.step(source_term_scaling=easing)

    initial_state = transition_sim.state

    # begin conjugate gradient optimization

    state = initial_state.copy()
    stop_condition_sq = (1e-2) ** 2
    residual = -compute_cost_gradient(state, use_source_terms=True).gradient
    initial_resid_norm_sq = residual.dot(residual)
    resid_norm_sq = initial_resid_norm_sq
    search_dir = residual.copy()

    results = SolveResults(
        state=state,
        step_count=0,
        control_energies=[compute_control_energy(state)],
        forward_energies=[],
        resid_norms=[math.sqrt(initial_resid_norm_sq)],
        a_inner_prods=[0.0],
    )

    print("Computing exact controllability solution...")

    for _ in range(args.max_iters):
        resid_update = compute_cost_gradient(
            search_dir, use_source_terms=False
        ).gradient
        solution_update_param = resid_norm_sq / resid_update.dot(search_dir)
        results.state += search_dir.scaled(solution_update_param)
        residual -= resid_update.scaled(solution_update_param)

        next_resid_norm_sq = residual.dot(residual)
        resid_norm_proportion = next_resid_norm_sq / resid_norm_sq
        resid_norm_sq = next_resid_norm_sq
        search_dir = residual + search_dir.scaled(resid_norm_proportion)

        # measurements
        results.control_energies.append(compute_control_energy(results.state))
        results.resid_norms.append(math.sqrt(resid_norm_sq))
        results.a_inner_prods.append(resid_update.dot(search_dir))
        results.step_count += 1

        if (resid_norm_sq / initial_resid_norm_sq) < stop_condition_sq:
            print("Converged within step limit!")
            break

    print("Computing reference forward solution...")

    # compute energy of forward simulation over `max_iterations` time periods
    # to compare results to
    forward_state = initial_state.copy()
    for _ in range(results.step_count + 1):
        period_sim = ForwardSolve(forward_state.copy())
        for _ in range(steps_per_period):
            period_sim.step()
        results.forward_energies.append((period_sim.state - forward_state).energy())
        forward_state = period_sim.state

    return results


results_harmonic = solve()

print("Switching to Yee timestep operators...")
p_step_mat = p_step_mat_yee
q_step_mat = q_step_mat_yee
results_yee = solve()


#
# print and plot results
#

print("")
har_norm = results_harmonic.state.norm()
yee_har_diff = (results_yee.state - results_harmonic.state).norm()
print(f"Magnitude of harmonic solution: {har_norm}")
print(f"Difference between Yee and harmonic solution: {yee_har_diff}")
print(f"Relative difference: {yee_har_diff / har_norm:.2%}")

p_step_mat = p_step_mat_har
q_step_mat = q_step_mat_har
yee_energy_har = compute_control_energy(results_yee.state)
print(f"Yee solution's final energy: {results_yee.control_energies[-1]}")
print(f"Harmonic solution's final energy: {results_harmonic.control_energies[-1]}")
print(f"Yee solution's energy computed with harmonic timestepping: {yee_energy_har}")


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set(xlabel="iteration count", ylabel="control energy")
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
x_axis_har = range(results_harmonic.step_count + 1)
x_axis_yee = range(results_yee.step_count + 1)
(plot_ctl_har,) = ax.plot(
    x_axis_har, results_harmonic.control_energies, label="control (harmonic)"
)
(plot_fwd_har,) = ax.plot(
    x_axis_har, results_harmonic.forward_energies, label="forward (harmonic)"
)
(plot_ctl_yee,) = ax.plot(
    x_axis_yee, results_yee.control_energies, label="control (Yee)"
)
(plot_fwd_yee,) = ax.plot(
    x_axis_yee, results_yee.forward_energies, label="forward (Yee)"
)
ax.legend(handles=[plot_ctl_har, plot_fwd_har, plot_ctl_yee, plot_fwd_yee])
if args.save_visuals:
    fig.savefig("convergence.pdf")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set(xlabel="energy difference between harmonic and Yee")
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
# padding possibly required to match dimension of the vectors,
# since the number of steps to convergence may differ
max_step_count = max(results_harmonic.step_count, results_yee.step_count)
if results_harmonic.step_count < max_step_count:
    results_harmonic.control_energies += [0.0] * (
        max_step_count - results_harmonic.step_count
    )
if results_yee.step_count < max_step_count:
    results_yee.control_energies += [0.0] * (max_step_count - results_yee.step_count)
ax.plot(
    range(max_step_count + 1),
    np.array(results_yee.control_energies)
    - np.array(results_harmonic.control_energies),
)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set(xlabel="iteration count", ylabel="residual norm")
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax.plot(x_axis_har, results_harmonic.resid_norms)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set(xlabel="iteration count", ylabel="A-inner product of search directions")
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax.plot(x_axis_har, results_harmonic.a_inner_prods)
plt.show()

results_harmonic.state.draw(save=args.save_visuals)
plt.show()
if args.save_visuals:
    results_harmonic.state.save_anim(with_incident_wave=not args.no_inc_wave)

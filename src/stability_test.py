import argparse
import numpy as np
import numpy.typing as npt
import math
import matplotlib.pyplot as plt
import pydec
from dataclasses import dataclass
from typing import Iterable

from utils import mesh

# command line parameters
arg_parser = argparse.ArgumentParser(prog="stability_test")
arg_parser.add_argument(
    "--save-visuals",
    dest="save_visuals",
    action="store_true",
    help="save still images used in the thesis as .pdf",
)
args = arg_parser.parse_args()

# incident wave parameters
inc_wavenumber = 2.0
inc_wave_dir = np.array([0.0, 1.0])
inc_wave_vector = inc_wavenumber * inc_wave_dir
wave_speed = 1.0
inc_angular_vel = inc_wavenumber * wave_speed


@dataclass
class TestSetup:
    cmp_complex: pydec.SimplicialComplex
    bound_edges: set[pydec.Simplex]
    p_step_mat: npt.NDArray[np.float64]
    q_step_mat: npt.NDArray[np.float64]


print("Generating meshes...")
meshes = mesh.series_of_unstructured_square_refinements(
    edge_length=np.pi, elem_size=np.pi / 6.0, refinement_count=5
)
print("Preparing test setups...")
test_setups: list[TestSetup] = []
for mesh in meshes:
    cmp_complex = mesh.complex
    bound_edges = set(cmp_complex.boundary())
    # time stepping matrices
    p_step_mat = wave_speed**2 * cmp_complex[2].star * cmp_complex[1].d
    q_step_mat = cmp_complex[1].star_inv * cmp_complex[1].d.T

    test_setups.append(
        TestSetup(
            cmp_complex=cmp_complex,
            bound_edges=bound_edges,
            p_step_mat=p_step_mat,
            q_step_mat=q_step_mat,
        )
    )


@dataclass
class TestRun:
    setup: TestSetup
    dt: float

    pressure: npt.NDArray[np.float64]
    flux: npt.NDArray[np.float64]
    t: float = 0.0

    def __init__(self, setup: TestSetup, dt: float):
        self.setup = setup
        self.dt = dt
        self.pressure = np.zeros(setup.cmp_complex[2].num_simplices)
        for vert_idx in range(len(self.pressure)):
            self.pressure[vert_idx] = _eval_inc_wave_pressure(
                0.0, setup.cmp_complex[2].circumcenter[vert_idx]
            )
        self.flux = np.zeros(setup.cmp_complex[1].num_simplices)
        for edge_idx in range(len(self.flux)):
            edge = setup.cmp_complex[1].simplices[edge_idx]
            self.flux[edge_idx] = _eval_inc_wave_flux(
                0.5 * self.dt, edge, setup.cmp_complex
            )

    def step(self):
        self.t += self.dt
        self.pressure += self.dt * setup.p_step_mat @ self.flux
        # q is computed at a time instance offset by half dt
        t_at_q = self.t + 0.5 * self.dt
        self.flux += self.dt * setup.q_step_mat @ self.pressure
        # plane wave at the boundary for q
        for bound_edge in setup.bound_edges:
            edge_idx = self.setup.cmp_complex[1].simplex_to_index.get(bound_edge)
            self.flux[edge_idx] = _eval_inc_wave_flux(
                t_at_q, bound_edge, setup.cmp_complex
            )


# TODO: refactor the incident wave evaluation into a reusable module since it's used everywhere


def _eval_inc_wave_pressure(t: float, position: npt.NDArray) -> float:
    """Evaluate the value of p for the incident plane wave at a point."""

    return inc_angular_vel * math.sin(
        inc_angular_vel * t - np.dot(inc_wave_vector, position)
    )


def _eval_inc_wave_flux(
    t: float, edge_vert_indices: Iterable[int], cmp_complex: pydec.SimplicialComplex
) -> float:
    """Evaluate the line integral of the flux of the incident wave
    over an edge of the mesh, in other words compute a value of `q` from the wave.
    """

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


def is_stable(setup: TestSetup, dt: float) -> bool:
    """Check if a given dt value results in stable timestepping."""

    run = TestRun(setup, dt)
    # simulation is unstable if the variables' absolute values rise over time;
    # step a few times and compare
    initial_max_p = max(run.pressure)
    for _ in range(10):
        run.step()
    current_max_p = max(run.pressure)
    return (current_max_p - initial_max_p) < 0.01


print("Finding stable dt values...")

stable_dts: list[float] = []
for setup in test_setups:
    # binary search for the highest stable dt value
    lower_bound = 1e-3
    higher_bound = 1.0
    assert is_stable(setup, lower_bound), "lower bound at the start should be stable"
    assert not is_stable(
        setup, higher_bound
    ), "higher bound at the start should be unstable"

    SEARCH_ITERS = 10
    for _ in range(SEARCH_ITERS):
        middle = (higher_bound + lower_bound) / 2
        if is_stable(setup, middle):
            lower_bound = middle
        else:
            higher_bound = middle

    stable_dts.append(lower_bound)

lowest_edge_lengths: list[float] = [
    min(setup.cmp_complex[1].primal_volume) for setup in test_setups
]

# find a line just below the lowest slope given by these pairs
lowest_slope = min([y / x for x, y in zip(lowest_edge_lengths, stable_dts)])
print(f"Lowest dt/h: {lowest_slope}")
safe_slope = lowest_slope - 0.05

# plot the meshes

fig = plt.figure(figsize=[5, 5])
ax = plt.subplot(1, 1, 1)
for i in reversed(range(len(test_setups))):
    setup = test_setups[i]
    ax.triplot(
        setup.cmp_complex.vertices[:, 0],
        setup.cmp_complex.vertices[:, 1],
        triangles=setup.cmp_complex.simplices,
        color=(0, 0.5, 1.0, (1.0 - i * 0.2) ** 2) if i > 0 else (0, 0, 0, 1),
    )
if args.save_visuals:
    fig.savefig("stability_test_meshes.pdf")
plt.show()

# plot the results

fig = plt.figure(figsize=[5, 4])
ax = plt.subplot(1, 1, 1)
ax.set(xlabel="minimum edge length", ylabel="maximum stable dt")
ax.scatter(lowest_edge_lengths, stable_dts, s=80)
end_x = lowest_edge_lengths[0] + 0.05
(plt_slope,) = ax.plot(
    [0, end_x],
    [0, safe_slope * end_x],
    "--c",
    label=f"dt = {safe_slope:1.2f}h",
)
ax.legend(handles=[plt_slope])
if args.save_visuals:
    fig.savefig("stability_test_coefficients.pdf")
plt.show()

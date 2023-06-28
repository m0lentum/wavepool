"""Abstractions for visualizing values stored on different elements of a mesh."""

import matplotlib.animation as plt_anim
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
import numpy as np
import numpy.typing as npt
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

import pydec

from .sim_runner import Simulation


@dataclass(init=False)
class Animator(ABC):
    sim: Simulation

    fig: plt.Figure
    anim: plt_anim.FuncAnimation

    def __init__(
        self,
        sim: Simulation,
    ):
        self.sim = sim

        self.fig = plt.figure(layout="tight")
        self.anim = plt_anim.FuncAnimation(
            fig=self.fig,
            init_func=lambda: self.draw(),
            func=lambda _: self._step_and_draw(),
            frames=sim.step_count,
            interval=int(1000 * sim.dt),
        )

    @abstractmethod
    def draw(self):
        pass

    def _step_and_draw(self):
        self.sim.step()
        return self.draw()

    def show(self):
        """Run the simulation from the beginning and show it in an interactive window."""

        self.sim.init_state()
        plt.show()

    def save(self, filename: str = "result.mp4"):
        """Run the simulation from the beginning and save it to a file."""

        print(f"Saving {filename}. This takes a while")
        assert self.anim is not None
        self.sim.init_state()
        writer = plt_anim.FFMpegWriter(fps=int(1.0 / self.sim.dt))
        self.anim.save(filename, writer)


@dataclass(init=False)
class PressureOnVertices(Animator):
    """Visualize a 0-form on the vertices of a mesh."""

    ax: plt3d.Axes3D
    zlim: list[float]

    def __init__(
        self,
        sim: Simulation,
        get_data: Callable[[Any], npt.NDArray[np.float64]],
        zlim: list[float],
    ):
        super().__init__(sim)

        self.get_data = get_data
        self.ax = self.fig.add_subplot(1, 1, 1, projection="3d")
        self.zlim = zlim

    def draw(self):
        self.ax.clear()
        self.ax.set_zlim3d(self.zlim)
        return self.ax.plot_trisurf(
            self.sim.complex.vertices[:, 0],
            self.sim.complex.vertices[:, 1],
            self.get_data(self.sim),
            triangles=self.sim.complex.simplices,
            cmap="viridis",
            edgecolor="none",
        )


@dataclass(init=False)
class FluxAndPressure(Animator):
    """Visualize a primal 1-form flux and dual 0-form pressure."""

    ax: plt.Axes
    vmin: float
    vmax: float

    def __init__(
        self,
        sim: Simulation,
        get_pressure: Callable[[Any], npt.NDArray[np.float64]] = lambda s: s.p,
        get_flux: Callable[[Any], npt.NDArray[np.float64]] = lambda s: s.q,
        vmin: float = -2,
        vmax: float = 2,
        arrow_scale: float = 30,
    ):
        super().__init__(sim)
        self.ax = self.fig.add_subplot(1, 1, 1)

        self.get_pressure = get_pressure
        self.get_flux = get_flux
        self.vmin = vmin
        self.vmax = vmax
        self.arrow_scale = arrow_scale

    def draw(self):
        self.ax.clear()
        self.ax.tripcolor(
            self.sim.complex.vertices[:, 0],
            self.sim.complex.vertices[:, 1],
            triangles=self.sim.complex.simplices,
            facecolors=self.get_pressure(self.sim),
            edgecolors="k",
            vmin=self.vmin,
            vmax=self.vmax,
        )
        # self._draw_tris_as_arrows()
        barys, arrows = pydec.simplex_quivers(self.sim.complex, self.get_flux(self.sim))
        # rotate flux back to velocity direction
        arrows = np.vstack((arrows[:, 1], -arrows[:, 0])).T
        self.ax.quiver(
            barys[:, 0],
            barys[:, 1],
            arrows[:, 0],
            arrows[:, 1],
            units="dots",
            width=1,
            scale=1.0 / self.arrow_scale,
        )

    def _draw_tris_as_arrows(self):
        """Draw the triangle mesh with arrows denoting the orientation of an edge.
        Useful to debug orientation-related problems."""

        for edge in self.sim.complex[1].simplices:
            edge_start = self.sim.complex.vertices[edge[0]]
            edge_dir = 0.8 * (
                self.sim.complex.vertices[edge[1]] - self.sim.complex.vertices[edge[0]]
            )
            self.ax.arrow(
                edge_start[0],
                edge_start[1],
                edge_dir[0],
                edge_dir[1],
                head_width=0.01,
                head_length=0.04,
            )

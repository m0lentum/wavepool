"""Abstractions for visualizing values stored on different elements of a mesh."""

from typing import Any, Callable, Optional
import matplotlib.animation as plt_anim
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
import numpy as np
import numpy.typing as npt
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass

from sim_runner import Simulation


@dataclass(init=False)
class Animator(ABC):
    sim: Simulation
    get_data: Callable[[Any], npt.NDArray[np.float64]]

    fig: plt.Figure
    anim: plt_anim.FuncAnimation

    def __init__(
        self, sim: Simulation, get_data: Callable[[Any], npt.NDArray[np.float64]]
    ):
        self.sim = sim
        self.get_data = get_data

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

    def save_gif(self, filename: str = "result.gif"):
        """Run the simulation from the beginning and save it to a .gif file."""

        print(f"Saving {filename}. This takes a while")
        assert self.anim is not None
        self.sim.init_state()
        writer = plt_anim.ImageMagickWriter(fps=int(1.0 / self.sim.dt))
        self.anim.save(filename, writer)

    def save_mp4(self, filename: str = "result.mp4"):
        """Run the simulation from the beginning and save it to a .mp4 file."""

        print(f"Saving {filename}. This takes a while")
        assert self.anim is not None
        self.sim.init_state()
        writer = plt_anim.FFMpegWriter(fps=int(1.0 / self.sim.dt))
        self.anim.save(filename, writer)


@dataclass(init=False)
class ZeroForm(Animator):
    """Visualize a 0-form on the vertices of a mesh."""

    ax: plt3d.Axes3D
    zlim: list[float]

    def __init__(
        self,
        sim: Simulation,
        get_data: Callable[[Any], npt.NDArray[np.float64]],
        zlim: list[float],
    ):
        super().__init__(sim, get_data)

        self.ax = self.fig.add_subplot(1, 1, 1, projection="3d")
        self.zlim = zlim

    def draw(self):
        self.ax.clear()
        self.ax.set_zlim3d(self.zlim)
        return self.ax.plot_trisurf(
            self.sim.mesh.vertices[:, 0],
            self.sim.mesh.vertices[:, 1],
            self.get_data(self.sim),
            triangles=self.sim.mesh.indices,
            cmap="viridis",
            edgecolor="none",
        )


@dataclass
class TwoForm(Animator):
    """Visualize a 2-form on the faces of a mesh."""

    ax: plt.Axes
    vmin: float
    vmax: float

    def __init__(
        self,
        sim: Simulation,
        get_data: Callable[[Any], npt.NDArray[np.float64]],
        vmin: float,
        vmax: float,
    ):
        super().__init__(sim, get_data)

        self.ax = self.fig.add_subplot(1, 1, 1)
        self.vmin = vmin
        self.vmax = vmax

    def draw(self):
        self.ax.clear()
        return self.ax.tripcolor(
            self.sim.mesh.vertices[:, 0],
            self.sim.mesh.vertices[:, 1],
            triangles=self.sim.mesh.indices,
            facecolors=self.get_data(self.sim),
            edgecolors="k",
            vmin=self.vmin,
            vmax=self.vmax,
        )

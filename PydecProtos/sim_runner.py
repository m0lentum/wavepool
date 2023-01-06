"""Abstractions for quickly making time-dependent DEC simulations animated using matplotlib."""

import matplotlib.animation as plt_anim
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
import numpy as np
import numpy.typing as npt
import pydec
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass


@dataclass(init=False)
class Simulation(ABC):
    # parameters
    mesh: pydec.SimplicialMesh
    dt: float
    step_count: int
    zlim: list[float]
    # internals
    fig: plt.Figure
    ax: plt3d.Axes3D
    anim: plt_anim.FuncAnimation

    def __init__(
        self, mesh: pydec.SimplicialMesh, dt: float, step_count: int, zlim: list[float]
    ):
        self.mesh = mesh
        self.dt = dt
        self.step_count = step_count
        self.zlim = zlim

        self.fig = plt.figure(layout="tight")
        self.ax = self.fig.add_subplot(1, 1, 1, projection="3d")
        self.anim = plt_anim.FuncAnimation(
            fig=self.fig,
            init_func=lambda: self._draw(),
            func=lambda _: self._step_and_draw(),
            frames=step_count,
            interval=int(1000 * dt),
        )

    @abstractmethod
    def init_state(self):
        """Set up the initial conditions for the simulation."""
        pass

    @abstractmethod
    def step(self):
        """Run one timestep forward in time."""
        pass

    @abstractmethod
    def get_z_data(self) -> npt.NDArray[np.float64]:
        """Get the vector containing the z axis data to draw."""
        pass

    def _draw(self):
        self.ax.clear()
        self.ax.set_zlim3d(self.zlim)
        return self.ax.plot_trisurf(
            self.mesh.vertices[:, 0],
            self.mesh.vertices[:, 1],
            self.get_z_data(),
            cmap="viridis",
            edgecolor="none",
        )

    def _step_and_draw(self):
        self.step()
        return self._draw()

    def show(self):
        """Run the simulation from the beginning and show it in an interactive window."""
        self.init_state()
        plt.show()

    def save_gif(self, filename: str = "result.gif"):
        """Run the simulation from the beginning and save it to a .gif file."""
        print(f"Saving {filename}. This takes a while")
        self.init_state()
        writer = plt_anim.ImageMagickWriter(fps=int(1.0 / self.dt))
        self.anim.save(filename, writer)

    def save_mp4(self, filename: str = "result.mp4"):
        print(f"Saving {filename}. This takes a while")
        self.init_state()
        writer = plt_anim.FFMpegWriter(fps=int(1.0 / self.dt))
        self.anim.save(filename, writer)

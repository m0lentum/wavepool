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
from typing import Optional


@dataclass
class _Animation:
    """Matplotlib animation for visualizing simulations."""

    fig: plt.Figure
    ax: plt3d.Axes3D
    anim: plt_anim.FuncAnimation


@dataclass(init=False)
class Simulation(ABC):
    # parameters
    mesh: pydec.SimplicialMesh
    dt: float
    step_count: int
    zlim: list[float]

    # animation is created lazily when drawing is wanted -
    # we may also just want to run a simulation without animating it
    # in case we're measuring it or doing specialized drawing
    anim: Optional[_Animation] = None

    def __init__(
        self,
        mesh: pydec.SimplicialMesh,
        dt: float,
        step_count: int,
        zlim: list[float],
    ):
        self.mesh = mesh
        self.dt = dt
        self.step_count = step_count
        self.zlim = zlim

    def _init_animation(self):
        """Create a matplotlib animation lazily only when drawing functions are called."""

        fig = plt.figure(layout="tight")
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        anim = plt_anim.FuncAnimation(
            fig=fig,
            init_func=lambda: self._draw(),
            func=lambda _: self._step_and_draw(),
            frames=self.step_count,
            interval=int(1000 * self.dt),
        )
        self.anim = _Animation(fig, ax, anim)

    @abstractmethod
    def init_state(self):
        """Set up the initial conditions for the simulation state.
        For constants you should use `__init__` (or global scope) instead."""
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
        if self.anim is None:
            raise Exception("Animation wasn't created before drawing")

        self.anim.ax.clear()
        self.anim.ax.set_zlim3d(self.zlim)
        return self.anim.ax.plot_trisurf(
            self.mesh.vertices[:, 0],
            self.mesh.vertices[:, 1],
            self.get_z_data(),
            triangles=self.mesh.indices,
            cmap="viridis",
            edgecolor="none",
        )

    def _step_and_draw(self):
        self.step()
        return self._draw()

    def show(self):
        """Run the simulation from the beginning and show it in an interactive window."""

        self._init_animation()
        self.init_state()
        plt.show()

    def run_to_end(self):
        """Run the simulation until the specified end time without animating it."""

        self.init_state()
        for _ in range(self.step_count):
            self.step()

    def save_gif(self, filename: str = "result.gif"):
        """Run the simulation from the beginning and save it to a .gif file."""

        print(f"Saving {filename}. This takes a while")
        self._init_animation()
        assert self.anim is not None
        self.init_state()
        writer = plt_anim.ImageMagickWriter(fps=int(1.0 / self.dt))
        self.anim.anim.save(filename, writer)

    def save_mp4(self, filename: str = "result.mp4"):
        """Run the simulation from the beginning and save it to a .mp4 file."""

        print(f"Saving {filename}. This takes a while")
        self._init_animation()
        assert self.anim is not None
        self.init_state()
        writer = plt_anim.FFMpegWriter(fps=int(1.0 / self.dt))
        self.anim.anim.save(filename, writer)

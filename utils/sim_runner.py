"""Abstractions for quickly making time-dependent DEC simulations animated using matplotlib."""

import pydec
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass


@dataclass
class Simulation(ABC):
    complex: pydec.SimplicialComplex
    dt: float
    step_count: int

    @abstractmethod
    def init_state(self):
        """Set up the initial conditions for the simulation state.
        For constants you should use `__init__` (or global scope) instead."""
        pass

    @abstractmethod
    def step(self):
        """Run one timestep forward in time."""
        pass

    def run_to_end(self):
        """Run the simulation until the specified end time."""

        self.init_state()
        for _ in range(self.step_count):
            self.step()

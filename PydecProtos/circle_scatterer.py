"""
Wave equation (see membrane.py doc comment).
Circular scatterer object centered at origin,
circular computational domain around it with
absorbing boundary condition at the outer edge
modeling empty space.
Incoming plane wave -cos(ωt - 𝜿·x).
"""

import mesh
from sim_runner import Simulation

import numpy as np
import matplotlib.pyplot as plt
import pydec

scatterer_radius = 1.0
outer_edge_radius = 2.0

cmp_mesh = mesh.annulus(scatterer_radius, outer_edge_radius, refine_count=1)
plt.triplot(cmp_mesh.vertices[:, 0], cmp_mesh.vertices[:, 1], cmp_mesh.indices)
plt.show()

inc_wavenumber = 3.0
inc_wave_dir = np.array([0.0, 1.0])
inc_wave_vector = inc_wavenumber * inc_wave_dir

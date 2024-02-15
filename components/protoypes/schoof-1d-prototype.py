import numpy as np
from scipy.optimize import least_squares
import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

# Constants
c1 = 1.3455e-9 # m^3 J^-1
c2 = 7.11e-24 # Pa^-3 s^-1
c3 = 4.05e-2 # m^9/4 Pa^-1/2 s^-1
a = 5/4
n = 3
rho_i = 910 # kg m^-3
rho_w = 1000 # kg m^-3
g = 9.81 # m s^-2
v0 = 3.12e-8 # m^2 s^-1
vp = 5.1e-6 # m^2 Pa^-1
S0 = 5.74 # m^2
psi0 = 1630 # Pa m^-1
L0 = 5000 # m
h0 = 5 # m
m = 2e-3 # m^2 s^-1
n_cells = 500

# Grid
grid = jnp.linspace(0, 5000, n_cells)
bed = -4.563492 - 0.07106481 * grid + 0.00004484127 * grid**2 - 5.324074e-9 * grid**3
surface = h0 * jnp.sqrt(grid)
thickness = surface - bed
base_gradient = jnp.asarray(-np.gradient(rho_i * g * thickness, grid) - rho_w * g * np.gradient(bed, grid))
discharge = (L0 - grid) * m

# Solver
def solve(psi, Q, S):
    return c3 * S**a * psi - Q

lsq = least_squares(
    lambda psi: solve(psi, discharge, 1e-2 * jnp.ones(n_cells)), 
    jnp.zeros(n_cells), 
    method = 'trf',
    jac_sparsity = jax.jacrev(solve)(jnp.zeros(n_cells), discharge, 1e-2 * jnp.ones(n_cells)),
    verbose = 2
)

def solve_N(N, psi):
    return base_gradient + jnp.gradient(N) - psi

lsq_N = least_squares(
    lambda N: solve_N(N, lsq.x), 
    jnp.zeros(n_cells), 
    method = 'trf',
    jac_sparsity = jax.jacrev(solve_N)(jnp.zeros(n_cells), lsq.x)
)

plt.plot(grid, lsq_N.x * 1e-6)
plt.show()

import numpy as np
import jax
import jax.numpy as jnp
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

# Solver
def run_one_step(dt, S, psi, N):
    dSdt = c1 * m * psi + v0 * (1 - S / S0) - c2 * S * N**n
    S += dSdt * dt
    Q = c3 * S**a * jnp.abs(psi)**(-1/2) * psi
    Q.at[-1].set(0.0)

    dNdt = (m - np.gradient(Q, grid)) / vp
    N += dNdt * dt
    N = jnp.where(N < 0, 0, N)
    N = jnp.where(N > rho_i * g * thickness, rho_i * g * thickness, N)
    N.at[0].set(0.0)

    psi = base_gradient + np.gradient(N, grid)

    N_fit = jnp.polyfit(grid, N, 3)
    N = N_fit[0] * grid**3 + N_fit[1] * grid**2 + N_fit[2] * grid + N_fit[3]

    psi_fit = jnp.polyfit(grid, psi, 3)
    psi = psi_fit[0] * grid**3 + psi_fit[1] * grid**2 + psi_fit[2] * grid + psi_fit[3]

    return S, psi, N

# Initial conditions
S = S0 * jnp.ones(n_cells)
N = jnp.zeros(n_cells)
psi = base_gradient + np.gradient(N, grid)

# Plot initial conditions
# plt.plot(grid, bed, label = 'Bed elevation')
# plt.plot(grid, surface, label = 'Surface elevation')
# plt.xlabel('Distance from terminus (m)')
# plt.ylabel('Elevation (m)')
# plt.legend()
# plt.show()

# Time stepping
dt = 100.0
n_steps = 1000

plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, 10)))

for t in range(n_steps):
    S, psi, N = run_one_step(dt, S, psi, N)

    if t % 100 == 99:
        print(f'Step {t} of {n_steps}')

        overburden = rho_i * g * thickness
        plt.plot(grid, psi)

plt.xlabel('Distance from terminus (m)')
plt.ylabel('Hydraulic gradient (Pa m$^{-1}$)')
plt.show()

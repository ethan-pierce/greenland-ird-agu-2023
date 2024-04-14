"""Vertical (1D) model of sediment entrainment at the GISP-2 site.

GISP2 is located at 214965, -1886724 
in the NSIDC Sea Ice Polar Stereographic North CRS (EPSG:3413).
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import equinox as eqx
from scipy.special import erf

from landlab import RasterModelGrid
from components import ModelState, FrozenFringe
from utils import StaticGrid, plot_triangle_mesh, freeze_grid

temperature = np.genfromtxt(
    './examples/cores/gisp2/gisp2_temp_accum_alley2000.txt', 
    skip_header = 75,
    max_rows = 1707 - 75
)
temperature[:, 0] *= 1e3
temperature[:, 1] += 273

accumulation = np.genfromtxt(
    './examples/cores/gisp2/gisp2_temp_accum_alley2000.txt', 
    skip_header = 1717,
    max_rows = 3414 - 1717
)
accumulation[:, 0] *= 1e3
accumulation[:, 1] *= 1 / 31556926

accumulation_interp = np.column_stack(
    (temperature[:, 0],
    np.interp(temperature[:, 0], accumulation[:, 0], accumulation[:, 1]))
)

z = np.arange(0, 3050, 0.1)

H = 3050
heat_capacity = 2093
conductivity = 2.1
diffusivity = 1.09e-6
latent_heat = 3.335e5
geothermal_heat_flux = 0.05
accumulation_today = 0.2 / 31556926
surface_temperature = -30.54
melt_temperature = 273
premelting_length_scale = 4e-10
premelting_exponent = 2.7
water_density = 1000
ice_density = 917
viscosity = 1.8e-3
particle_radius = 1e-4
effective_pressure = 250e3

def T_at(z, Ts, b):
    advection_coeff = np.sqrt(2 * diffusivity * H / b)
    heat_flux = -geothermal_heat_flux / conductivity
    T = (
        Ts 
        + np.sqrt(np.pi) / 2
        * advection_coeff
        * heat_flux
        * (
            erf(
                z / advection_coeff
            )
            -
            erf(
                np.max(z) / advection_coeff
            )
        )
    )
    return np.where(T > melt_temperature, melt_temperature - 1, T)

def film_thickness(T):
    return (
        premelting_length_scale * (melt_temperature / (melt_temperature - T))**(1 / premelting_exponent)
    )

def regelation_velocity(T):
    return (
        (water_density**2 * film_thickness(T)**3 * effective_pressure)
        / (12 * ice_density**2 * viscosity * particle_radius**2)
    )

dt = np.diff(temperature[:, 0], prepend = temperature[0, 0])
accs = []
basal_temperature = []
regelation_rate = []
cumulative_regelation = [0]

for t in reversed(range(temperature.shape[0])):
    acc = accumulation_interp[t, 1]
    accs.append(acc)

    Tb = T_at(z, temperature[t, 1], acc)[0]
    basal_temperature.append(Tb)

    V = regelation_velocity(Tb)
    regelation_rate.append(V)

    cumulative_regelation.append(cumulative_regelation[-1] + V * 31556926 * dt[t])

# plt.plot(temperature[:, 0][::-1], accs)
# plt.xlabel('years BP')
# plt.show()

fig, ax = plt.subplots(figsize = (12, 4))
plt.plot(temperature[:, 0][::-1], basal_temperature)
plt.xlabel('years BP')
plt.ylabel('Basal temperature (K)')
plt.savefig('./examples/cores/gisp2/figures/basal-temperature.png', dpi = 300)
# plt.show()

fig, ax = plt.subplots(figsize = (12, 4))
ax.plot(temperature[:, 0][::-1], regelation_rate)
ax.set_yscale('log')
plt.xlabel('years BP')
plt.ylabel('(log) regelation rate (m / s)')
plt.savefig('./examples/cores/gisp2/figures/regelation-rate.png', dpi = 300)
# plt.show()

fig, ax = plt.subplots(figsize = (12, 4))
plt.plot(temperature[:, 0][::-1], cumulative_regelation[1:])
plt.xlabel('years BP')
plt.ylabel('Cumulative regelation (m)')
plt.savefig('./examples/cores/gisp2/figures/cumulative-regelation.png', dpi = 300)
# plt.show()


# ##############################
# # Step 1: Load the grid data #
# ##############################

# with open('./examples/cores/gisp2/kangilliup-sermia.grid', 'rb') as g:
#     tmg = pickle.load(g)
# print(tmg.at_node.keys())
# print(tmg.at_link.keys())

# # Identify the node closest to the GISP-2 core location
# nodes = np.array([tmg.node_x, tmg.node_y]).T
# pt = np.array([214965, -1886724])
# idx = np.sum((nodes - pt)**2, axis = 1, keepdims = True).argmin(axis = 0)

# # Initialize the 1D model
# grid = RasterModelGrid((3, 3))

# H = tmg.at_node['ice_thickness'][idx][0]
# print('Ice thickness:', H, 'm')
# grid.add_field('ice_thickness', np.full(grid.number_of_nodes, H), at = 'node')

# S = tmg.at_node['surface_elevation'][idx][0]
# print('Surface elevation:', S, 'm')
# grid.add_field('surface_elevation', np.full(grid.number_of_nodes, S), at = 'node')

# dS = tmg.calc_slope_at_node('surface_elevation')[idx][0]
# print('Surface slope:', dS)
# grid.add_field('surface_slope', np.full(grid.number_of_nodes, dS), at = 'node')

# A = 1.2e-24
# gamma = (2/5) * A * (917 * 9.81)**3
# secpera = 31556926
# Ud = gamma * H**4 * dS**3 * secpera
# Us = np.sqrt(tmg.at_node['surface_velocity_x'][idx][0]**2 + tmg.at_node['surface_velocity_y'][idx][0]**2)
# Ub = Us - Ud
# print('Surface velocity:', Us, 'm/yr')
# print('Deformation velocity:', Ud, 'm/yr')
# print('Basal velocity:', Ub, 'm/yr')
# grid.add_field('sliding_velocity', np.full(grid.number_of_links, Ub), at = 'link')

# grid.add_field('geothermal_heat_flux', np.full(grid.number_of_nodes, 0.05), at = 'node')

# grid.add_field('water_pressure', np.full(grid.number_of_nodes, H * 917 * 9.81 * 0.95), at = 'node')
# print('Effective pressure:', H * 917 * 9.81 * 0.05 * 1e-6, 'MPa')

# # Initialize the model state
# state = ModelState(
#     freeze_grid(grid),
#     ice_thickness = grid.at_node['ice_thickness'],
#     surface_elevation = grid.at_node['surface_elevation'],
#     sliding_velocity = grid.at_link['sliding_velocity'],
#     geothermal_heat_flux = grid.at_node['geothermal_heat_flux'],
#     water_pressure = grid.at_node['water_pressure']
# )
# state = eqx.tree_at(lambda t: t.surface_slope, state, grid.at_node['surface_slope'])
# state = eqx.tree_at(lambda t: t.till_thickness, state, np.full(grid.number_of_nodes, 20.0))
# state = eqx.tree_at(lambda t: t.fringe_thickness, state, np.full(grid.number_of_nodes, 0.10))

# # Initialize the FrozenFringe component
# model = FrozenFringe(state = state)
# model = eqx.tree_at(lambda t: t.critical_depth, model, H - 100)

# print(model.dispersed_growth_rate[4] * 31556926)
# quit()

# # Get the frozen fringe to steady state
# dts = [1e-7]
# fringes = []
# dispersed = []

# dt = 1e-7
# for i in range(20000):
#     model = model.update(dt)
#     model = eqx.tree_at(lambda t: t.critical_depth, model, H - 100)

#     dts.append(dts[-1] + dt)
#     fringes.append(model.state.fringe_thickness[4])
#     dispersed.append(model.state.dispersed_thickness[4])

# plt.plot(dts[1:], fringes)
# plt.show()

# print(dispersed[-1] + model.dispersed_growth_rate[4] * 100000 * 31556926)

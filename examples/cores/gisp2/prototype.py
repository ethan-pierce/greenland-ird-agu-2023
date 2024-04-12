"""Vertical (1D) model of sediment entrainment at the GISP-2 site.

GISP2 is located at 214965, -1886724 
in the NSIDC Sea Ice Polar Stereographic North CRS (EPSG:3413).
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import equinox as eqx

from landlab import RasterModelGrid
from components import ModelState, FrozenFringe
from utils import StaticGrid, plot_triangle_mesh, freeze_grid

##############################
# Step 1: Load the grid data #
##############################

with open('./examples/cores/gisp2/kangilliup-sermia.grid', 'rb') as g:
    tmg = pickle.load(g)
print(tmg.at_node.keys())
print(tmg.at_link.keys())

# Identify the node closest to the GISP-2 core location
nodes = np.array([tmg.node_x, tmg.node_y]).T
pt = np.array([214965, -1886724])
idx = np.sum((nodes - pt)**2, axis = 1, keepdims = True).argmin(axis = 0)

# Initialize the 1D model
grid = RasterModelGrid((3, 3))

H = tmg.at_node['ice_thickness'][idx][0]
print('Ice thickness:', H, 'm')
grid.add_field('ice_thickness', np.full(grid.number_of_nodes, H), at = 'node')

S = tmg.at_node['surface_elevation'][idx][0]
print('Surface elevation:', S, 'm')
grid.add_field('surface_elevation', np.full(grid.number_of_nodes, S), at = 'node')

dS = tmg.calc_slope_at_node('surface_elevation')[idx][0]
print('Surface slope:', dS)
grid.add_field('surface_slope', np.full(grid.number_of_nodes, dS), at = 'node')

A = 1.2e-24
gamma = (2/5) * A * (917 * 9.81)**3
secpera = 31556926
Ud = gamma * H**4 * dS**3 * secpera
Us = np.sqrt(tmg.at_node['surface_velocity_x'][idx][0]**2 + tmg.at_node['surface_velocity_y'][idx][0]**2)
Ub = Us - Ud
print('Surface velocity:', Us, 'm/yr')
print('Deformation velocity:', Ud, 'm/yr')
print('Basal velocity:', Ub, 'm/yr')
grid.add_field('sliding_velocity', np.full(grid.number_of_links, Ub), at = 'link')

grid.add_field('geothermal_heat_flux', np.full(grid.number_of_nodes, 0.05), at = 'node')

grid.add_field('water_pressure', np.full(grid.number_of_nodes, H * 917 * 9.81 * 0.8), at = 'node')
print('Effective pressure:', H * 917 * 9.81 * 0.2 * 1e-6, 'MPa')

# Initialize the model state
state = ModelState(
    freeze_grid(grid),
    ice_thickness = grid.at_node['ice_thickness'],
    surface_elevation = grid.at_node['surface_elevation'],
    sliding_velocity = grid.at_link['sliding_velocity'],
    geothermal_heat_flux = grid.at_node['geothermal_heat_flux'],
    water_pressure = grid.at_node['water_pressure']
)
state = eqx.tree_at(lambda t: t.surface_slope, state, grid.at_node['surface_slope'])
state = eqx.tree_at(lambda t: t.till_thickness, state, np.full(grid.number_of_nodes, 20.0))
state = eqx.tree_at(lambda t: t.fringe_thickness, state, np.full(grid.number_of_nodes, 1e-3))

# Initialize the FrozenFringe component
model = FrozenFringe(state = state)
model = eqx.tree_at(lambda t: t.critical_depth, model, H - 100)

# Get the frozen fringe to steady state
dts = [1e-7]
fringes = []
dispersed = []

dt = 1e-7
for i in range(20000):
    model = model.update(dt)
    model = eqx.tree_at(lambda t: t.critical_depth, model, H - 100)

    dts.append(dts[-1] + dt)
    fringes.append(model.state.fringe_thickness[4])
    dispersed.append(model.state.dispersed_thickness[4])

plt.plot(dts[1:], fringes)
plt.show()

print(dispersed[-1] + model.dispersed_growth_rate[4] * 100000 * 31556926)

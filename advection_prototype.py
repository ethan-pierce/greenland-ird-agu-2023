import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import RBFInterpolator

from components import ModelState
from utils import StaticGrid
from utils.plotting import plot_links, plot_triangle_mesh

with open('examples/ird/meshes/rolige-brae.grid', 'rb') as g:
    tmg = pickle.load(g)

print(tmg.at_node.keys())

grid = StaticGrid.from_grid(tmg)

S = tmg.at_node['surface_elevation'][:]
H = tmg.at_node['ice_thickness'][:]
coords = np.stack((tmg.node_x, tmg.node_y)).T

# smoothS = RBFInterpolator(coords, S, smoothing = 10)(coords)
gradS = grid.calc_grad_at_link(S)

gamma = (2/5) * 6e-24 * (917 * 9.81)**3 * 31556926
Ud = gamma * grid.map_mean_of_link_nodes_to_link(H)**4 * gradS**3
Us = grid.map_vectors_to_links(tmg.at_node['surface_velocity_x'][:], tmg.at_node['surface_velocity_y'][:])

Ub = np.where(
    np.abs(Ud) > np.abs(Us),
    Us,
    np.sign(Us) * (np.abs(Us) - np.abs(Ud))
)

tmg.add_field('sliding_velocity', Ub, at = 'link')
tmg.add_field('water_pressure', H * 917 * 9.81 * 0.8, at = 'node')

state = ModelState.from_grid(tmg)

Q = tmg.add_zeros('field', at = 'node')
Q[(tmg.node_x > 5.9e5) & (tmg.node_x < 5.93e5) & (tmg.node_y > -2.046e6) & (tmg.node_y < -2.043e6)] = 10
Q0 = Q.copy()

for i in range(1000):
    dQ = grid.calc_flux_div_at_node(Ub * tmg.map_value_at_max_node_to_link(S, Q))
    Q += dQ * 2.0

plot_triangle_mesh(tmg, Q- Q0, subplots_args = {'figsize': (18, 6)})
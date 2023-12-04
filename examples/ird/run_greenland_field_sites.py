"""Frozen fringe models for Greenland tidewater glaciers.

Runs a series of numerical experiments over three fjord systems in Greenland.
Sites were chosen based on available field data from the 2019-2022 seasons,
and include representative examples from both east and west coasts. Here,
we model frozen fringe development, advection, and eventual export to the fjord.
"""

import os
import pickle
import numpy as np
import xarray as xr
import rioxarray as rxr
import geopandas as gpd
from scipy.interpolate import bisplrep, bisplev
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import equinox as eqx

from landlab import RasterModelGrid
from components import ModelState, GlacialEroder, FrozenFringe
from components.model_state import initialize_state_from_grid
from utils import StaticGrid, TVDAdvection, UpwindAdvection
from utils.static_grid import freeze_grid
from utils.plotting import plot_links, plot_triangle_mesh

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gen_mesh', default = False, action = 'store_true')
args = parser.parse_args()

# Bounds for each terminus, in the form [xmin, xmax, ymin, ymax]
bounds = {
    'rolige-brae': [6.08e5, 6.3e5, -2.035e6, -2.03e6],
    'sermeq-avannarleq': [-2.063e5, -1.94e5, -2.175e6, -2.17e6],
    'charcot-gletscher': [5.438e5, 5.453e5, -1.8834e6, -1.8814e6],
    'sydbrae': [6.917e5, 6.966e5, -2.05300e6, -2.0503e6],
    'kangiata-nunaata-sermia': [-2.322e5, -2.2387e5, -2.8211e6, -2.81829e6],
    'eielson-gletsjer': [[5.9432e5, 5.9806e5, -1.9938e6, -1.99107e6], [6.0470e5, 6.0975e5, -1.9721e6, -1.9684e6]],
    'narsap-sermia': [-2.4817e5, -2.4318e5, -2.78049e6, -2.77618e6],
    'kangilernata-sermia': [-2.0785e5, -2.0248e5, -2.19282e6, -2.1822e6],
    'dode-brae': [5.84982e5, 5.87e5, -2.057326e6, -2.0553e6],
    'daugaard-jensen-gletsjer': [5.5327e5, 5.6091e5, -1.89852e6, -1.8912e6],
    'vestfjord-gletsjer': [5.849e5, 5.8915e5, -2.064e6, -2.0616e6],
    'sermeq-kullajeq': [-1.99773e5, -1.98032e5, -2.18107e6, -2.17603e6],
    'bredegletsjer': [7.2777e5, 7.3204e5, -2.03134e6, -2.02869e6],
    'magga-dan-gletsjer': [6.65261e5, 6.68950e5, -2.09014e6, -2.08383e6],
    'graah-gletscher': [5.48122e5, 5.50237e5, -1.877166e6, -1.874439e6],
    'akullersuup-sermia': [-2.29522e5, -2.26196e5, -2.816803e6, -2.813243e6],
    'eqip-sermia': [-2.04326e5, -2.01153e5, -2.204225e6, -2.200172e6],
    'kista-dan-gletsjer': [6.60337e5, 6.63701e5, -2.09062e6, -2.08841e6]
}

#####################
# Step 1: Load data #
#####################

if args.gen_mesh:

    input_dir = './examples/ird/meshes/'
    models = {}
    grids = {}

    for f in os.listdir(input_dir):
        glacier = f.split('/')[-1].replace('.grid', '')

        with open(input_dir + f, 'rb') as g:
            tmg = pickle.load(g)
            
            from landlab.graph.sort.sort import reorient_link_dirs
            reorient_link_dirs(tmg)

        grid = freeze_grid(tmg)

        H = tmg.at_node['ice_thickness'][:]
        S = tmg.at_node['smoothed_surface'][:]
        
        gradS = grid.calc_grad_at_link(S)
        A = 1.2e-24
        gamma = (2/5) * A * (917 * 9.81)**3
        secpera = 31556926
        Ud = gamma * grid.map_mean_of_link_nodes_to_link(H)**4 * gradS**3 * secpera
        Us = grid.map_vectors_to_links(tmg.at_node['surface_velocity_x'][:], tmg.at_node['surface_velocity_y'][:])
        Ub = np.where(
            np.abs(Ud) > np.abs(Us),
            0.0,
            np.sign(Us) * (np.abs(Us) - np.abs(Ud))
        )

        tmg.add_field('sliding_velocity', Ub, at = 'link')
        tmg.add_field('water_pressure', H * 917 * 9.81 * 0.8, at = 'node')

        state = initialize_state_from_grid(tmg)
        models[glacier] = state
        grids[glacier] = tmg

        print('Loaded data for ' + glacier.replace('-', ' ').title())

    with open('./examples/ird/initial_conditions.pickle', 'wb') as f:
        pickle.dump(models, f)

    with open('./examples/ird/landlab_grids.pickle', 'wb') as g:
        pickle.dump(grids, g)

#################################
# Step 2: Define update routine #
#################################

@jax.jit
def update(state, dt: float):
    eroder = GlacialEroder(state)
    state = eroder.update(dt).state

    fringe = FrozenFringe(state)
    state = fringe.update(dt).state

    advect = UpwindAdvection(
        state.grid, 
        state.fringe_thickness, 
        state.surface_elevation, 
        state.sliding_velocity
    )
    updated_fringe = advect.update(dt)

    state = eqx.tree_at(
        lambda tree: tree.fringe_thickness,
        state,
        updated_fringe
    )

    return state

def constrain_terminus(state, xmin, xmax, ymin, ymax):
    node_is_terminus = jnp.where(
        state.node_is_terminus
        & (state.grid.node_x > xmin)
        & (state.grid.node_x < xmax)
        & (state.grid.node_y > ymin)
        & (state.grid.node_y < ymax),
        1,
        0
    )
    return node_is_terminus

######################
# Step 3: Run models #
######################

with open('./examples/ird/initial_conditions.pickle', 'rb') as f:
    models = pickle.load(f)

with open('./examples/ird/landlab_grids.pickle', 'rb') as g:
    grids = pickle.load(g)

for glacier, state in models.items():
    print(glacier)

    if glacier == 'eielson-gletsjer':
        terminus_a = constrain_terminus(state, bounds[glacier][0][0], bounds[glacier][0][1], bounds[glacier][0][2], bounds[glacier][0][3])
        terminus_b = constrain_terminus(state, bounds[glacier][1][0], bounds[glacier][1][1], bounds[glacier][1][2], bounds[glacier][1][3])
        terminus = terminus_a | terminus_b
    else:
        terminus = constrain_terminus(state, bounds[glacier][0], bounds[glacier][1], bounds[glacier][2], bounds[glacier][3])

    corners = state.grid.corners_at_face[state.grid.length_of_face == 0]
    if len(corners) > 0:
        zero_links = jnp.argwhere(state.grid.length_of_face[state.grid.face_at_link] == 0)
        print(zero_links)

    # cross_terminus_links = terminus[state.grid.node_at_link_head] * terminus[state.grid.node_at_link_tail]
    # print(jnp.sum(jnp.where(cross_terminus_links, state.grid.length_of_link, 0.0)))

    # plot_triangle_mesh(grids[glacier], state.grid.map_mean_of_links_to_node(state.sliding_velocity), subplots_args = {'figsize': (18, 12)})

quit()

for i in range(3000):
    state = update(state, dt = 0.1)
    if i % 100 == 0:
        print(i)

plot_triangle_mesh(grids[glacier], state.fringe_thickness, subplots_args = {'figsize': (18, 6)})

########################
# Step 4: Save results #
########################


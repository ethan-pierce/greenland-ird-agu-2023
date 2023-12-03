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

######################
# Step 3: Run models #
######################

with open('./examples/ird/initial_conditions.pickle', 'rb') as f:
    models = pickle.load(f)

with open('./examples/ird/landlab_grids.pickle', 'rb') as g:
    grids = pickle.load(g)

glacier = 'rolige-brae'
state = models[glacier]

for i in range(1000):
    state = update(state, dt = 0.01)

plot_triangle_mesh(grids[glacier], state.fringe_thickness, subplots_args = {'figsize': (18, 6)})

########################
# Step 4: Save results #
########################
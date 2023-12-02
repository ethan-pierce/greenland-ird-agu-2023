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
import matplotlib.pyplot as plt

from landlab import RasterModelGrid
from components import ModelState, GlacialEroder, FrozenFringe
from utils import StaticGrid, TVDAdvection
from utils.plotting import plot_links, plot_triangle_mesh

#####################
# Step 1: Load data #
#####################

input_dir = './examples/ird/meshes/'
models = {}

for f in os.listdir(input_dir):
    glacier = f.split('/')[-1].replace('.grid', '')

    with open(input_dir + f, 'rb') as g:
        tmg = pickle.load(g)
        
        from landlab.graph.sort.sort import reorient_link_dirs
        reorient_link_dirs(tmg)

    grid = StaticGrid.from_grid(tmg)

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

    state = ModelState.from_grid(tmg)
    models[glacier] = state

    print('Loaded data for ' + glacier.replace('-', ' ').title())

#################################
# Step 2: Define update routine #
#################################

######################
# Step 3: Run models #
######################

########################
# Step 4: Save results #
########################
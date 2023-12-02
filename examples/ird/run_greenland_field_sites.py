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
from scipy.ndimage import gaussian_filter

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

    H = tmg.at_node['ice_thickness'][:]
    S = tmg.at_node['surface_elevation'][:]

    tmg.add_field('water_pressure', H * 917 * 9.81 * 0.8, at = 'node')
    
    

    state = ModelState.from_grid(tmg)
    models[glacier] = state



#################################
# Step 2: Define update routine #
#################################

######################
# Step 3: Run models #
######################

########################
# Step 4: Save results #
########################
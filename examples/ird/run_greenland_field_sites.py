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

input_dirs = '/home/egp/repos/greenland-ird/igm/model-runs/'
shapefiles_dir = '/home/egp/repos/greenland-ird/data/basin-outlines/all/'
models = {}

for folder in os.listdir(input_dirs):
    models[folder] = {'grid': None, 'state': None}
    ds = xr.open_dataset(input_dirs + folder + '/input.nc')
    ds = ds.rio.write_crs('epsg:3413')

    # TODO To upscale or not to upscale?
    # ds = ds.rio.reproject(ds.rio.crs, shape = (ds.rio.height * 2, ds.rio.width * 2))

    series = gpd.read_file(shapefiles_dir + folder + '.geojson')
    translated = series.geometry.translate(
        xoff = -series.geometry.centroid.x[0],
        yoff = -series.geometry.centroid.y[0]
    )

    ds = ds.rio.clip(translated)

    rmg = RasterModelGrid(
        ds.variables['thk'].shape, 
        xy_spacing = (np.diff(ds.variables['x'])[0], np.diff(ds.variables['y'])[0])
    )
    H = rmg.add_field('ice_thickness', ds.variables['thkobs'][:], at = 'node')
    S = rmg.add_field('surface_elevation', ds.variables['usurfobs'][:], at = 'node')
    rmg.add_field('geothermal_heat_flux', np.full(rmg.number_of_nodes, 0.05 * 31556926), at = 'node')
    rmg.add_field('water_pressure', rmg.at_node['ice_thickness'][:] * 917 * 9.81 * 0.8, at = 'node')
    Ux = rmg.add_field('surface_velocity_x', ds.variables['uvelsurfobs'][:], at = 'node')
    Uy = rmg.add_field('surface_velocity_y', ds.variables['vvelsurfobs'][:], at = 'node')

    # TODO How to set up test sliding velocity?
    Ub = rmg.add_field('sliding_velocity_vector', (Ux, Uy), at = 'node')

    grid = StaticGrid.from_grid(rmg)
    state = ModelState.from_grid(rmg)

    models[folder]['grid'] = grid
    models[folder]['state'] = state

    print('Finished loading data for ' + folder.replace('-', ' ').title())

with open('examples/ird/starting_point.pickle', 'wb') as f:
    pickle.dump(models, f)

#################################
# Step 2: Define update routine #
#################################

######################
# Step 3: Run models #
######################

########################
# Step 4: Save results #
########################
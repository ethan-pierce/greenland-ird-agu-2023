"""Frozen fringe models for Greenland tidewater glaciers.

Runs a series of numerical experiments over three fjord systems in Greenland.
Sites were chosen based on available field data from the 2019-2022 seasons,
and include representative examples from both east and west coasts. Here,
we model frozen fringe development, advection, and eventual export to the fjord.
"""

import os
import xarray as xr
import rioxarray as rxr
import geopandas as gpd
import shapely

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from components import ModelState, GlacialEroder, FrozenFringe
from utils import StaticGrid, TVDAdvection
from utils.plotting import plot_links, plot_triangle_mesh

#####################
# Step 1: Load data #
#####################

input_dirs = '/home/egp/repos/greenland-ird/igm/model-runs/'
shapefiles_dir = '/home/egp/repos/greenland-ird/data/basin-outlines/all/'
data = {}

for folder in os.listdir(input_dirs):
    ds = xr.open_dataset(input_dirs + folder + '/input.nc')
    ds = ds.rio.write_crs('epsg:3413')

    series = gpd.read_file(shapefiles_dir + folder + '.geojson')
    translated = series.geometry.translate(
        xoff = -series.geometry.centroid.x[0],
        yoff = -series.geometry.centroid.y[0]
    )

    ds = ds.rio.clip(translated)

    data[folder] = ds

    # fig, ax = plt.subplots(figsize = (18, 6))
    # im = ax.imshow(ds.variables['thkobs'][:])
    # plt.colorbar(im)
    # plt.title(folder.replace('-', ' ').title())
    # plt.show()

#################################
# Step 2: Define update routine #
#################################

######################
# Step 3: Run models #
######################

########################
# Step 4: Save results #
########################
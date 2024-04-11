"""Generate a mesh and load data for the GISP2 ice core."""

import numpy as np
import geopandas as gpd
from utils import GridLoader
from utils.plotting import plot_triangle_mesh

def main():
    geojson = 'glacierbento/examples/cores/gisp2/kangilliup-sermia.geojson'
    bedmachine = "/home/egp/repos/greenland-ird/data/ignore/BedMachineGreenland-v5.nc"
    velocity = "/home/egp/repos/greenland-ird/data/ignore/GRE_G0120_0000.nc"
    geotherm = "/home/egp/repos/greenland-ird/data/ignore/geothermal_heat_flow_map_10km.nc"
    shapefiles = "/home/egp/repos/greenland-ird/data/basin-outlines/"

    glacier = 'kangilliup-sermia'

    area = gpd.read_file(shapefiles + path).geometry.area
    tol = np.round(float(area) * 2e-8, 2)
    buffer = 250
    q = 24

    loader = GridLoader(
        geojson, 
        centered = False, 
        generate_grid = True, 
        max_area = int(area / 10000), 
        quality = q,
        buffer = buffer,
        tolerance = tol
    )

    print('Mesh nodes: ', loader.grid.number_of_nodes)
    print('Mesh links: ', loader.grid.number_of_links)

    resolution = calc_resolution(loader.polygon, n_cells = 40000)

    loader.add_field(
        bedmachine,
        "thickness",
        "ice_thickness",
        resolution,
        crs = "epsg:3413",
        no_data = -9999.0,
        neighbors = 9,
        smoothing = 0.0
    )
    h = loader.grid.at_node['ice_thickness']
    h[h < 0.5] = 0.0
    print('Added ice thickness to grid nodes.')

    loader.add_field(
        bedmachine,
        "bed",
        "bedrock_elevation",
        resolution,
        crs = "epsg:3413",
        neighbors = 9,
        no_data = -9999.0,
    )
    print("Added bedrock elevation to grid nodes.")

    loader.add_field(
        bedmachine,
        "surface",
        "smoothed_surface",
        resolution,
        crs = "epsg:3413",
        neighbors = 9,
        no_data = -9999.0,
        sigma = 7
    )

    loader.grid.add_field(
        'surface_elevation', 
        loader.grid.at_node['ice_thickness'][:] + loader.grid.at_node['bedrock_elevation'][:],
        at = 'node'
    )

    loader.add_field(
        geotherm, 
        "GHF", 
        "geothermal_heat_flux", 
        resolution,
        crs = "epsg:3413", 
        neighbors = 100, 
        no_data = np.nan, 
        scalar = 1e-3 * 31556926
    )
    print("Added geothermal heat flux to grid nodes.")

    loader.add_field(
        velocity, 
        "vx", 
        "surface_velocity_x", 
        resolution,
        crs = "epsg:3413", 
        neighbors = 100, 
        no_data = -1
    )
    loader.add_field(
        velocity, 
        "vy", 
        "surface_velocity_y", 
        resolution,
        crs = "epsg:3413", 
        neighbors = 100, 
        no_data = -1
    )
    loader.grid.add_field(
        'surface_velocity_vector', 
        (loader.grid.at_node['surface_velocity_x'][:], loader.grid.at_node['surface_velocity_y'][:]),
        at = 'node'
    )
    print("Added surface velocity to grid nodes.")

    loader.grid.save('/home/egp/repos/glacierbento/examples/ird/meshes/' + glacier + '.grid', clobber = True)

    # QC
    plot_triangle_mesh(loader.grid, loader.grid.at_node['smoothed_surface'][:], subplots_args = {'figsize': (18, 6)})
    plot_triangle_mesh(loader.grid, loader.grid.at_node['surface_velocity_x'][:], subplots_args = {'figsize': (18, 6)})

    print('Finished loading data for ' + glacier.replace('-', ' ').capitalize())

"""Frozen fringe models for the GISP2 ice core site.

GISP2 is located at 214965, -1886724 
in the NSIDC Sea Ice Polar Stereographic North CRS (EPSG:3413)
"""

import os
import pickle
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
import geopandas as gpd
from scipy.interpolate import bisplrep, bisplev
import matplotlib.pyplot as plt
import matplotlib.colors
import cmcrameri.cm as cmc

plt.rcParams.update({'image.cmap': 'cmc.batlow'})

import jax
import jax.numpy as jnp
import equinox as eqx

from landlab import RasterModelGrid
from components import ModelState, GlacialEroder, FrozenFringe
from components.model_state import initialize_state_from_grid
from utils import StaticGrid, TVDAdvector
from utils.static_grid import freeze_grid
from utils.plotting import plot_links, plot_triangle_mesh

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gen_mesh', default = False, action = 'store_true')
args = parser.parse_args()

# Bounds for each terminus, in the form [xmin, xmax, ymin, ymax]
bounds = {
    'kangilliup-sermia': [-2.324e4, -2.304e5, -1.984e6, -1.9795e6]
}

discharge_gate = {
    'kangilliup-sermia': 143
}
discharge_data = pd.read_csv('/home/egp/repos/local/ice-discharge/dataverse_files/gate_D.csv', header = 0)
discharge = {key: discharge_data[str(val)].iloc[2666:].mean() for key, val in discharge_gate.items()}

results = {
    'glacier': [], 'region': [], 'N_scalar': [], 'ice_flux': [], 
    'boundary_flux': [], 'proximal_flux': [], 'sediment_flux': [], 
    'max_velocity': [], 'max_pressure': [], 'drainage_area': [],
    'mean_fringe': [], 'med_fringe': []
}

#####################
# Step 1: Load data #
#####################

glacier = 'kangilliup-sermia'

with open('./examples/cores/gisp2/kangilliup-sermia.grid', 'rb') as g:
    tmg = pickle.load(g)
    
    from landlab.graph.sort.sort import reorient_link_dirs
    reorient_link_dirs(tmg)

H = tmg.at_node['ice_thickness'][:]
tmg.add_field('water_pressure', H * 917 * 9.81, at = 'node')

grid = freeze_grid(tmg)

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
    np.sign(Ud) * (np.abs(Us) - np.abs(Ud))
)

tmg.add_field('sliding_velocity', Ub, at = 'link')

state = initialize_state_from_grid(tmg)

print('Loaded data for ' + glacier.replace('-', ' ').title())

with open('./examples/cores/gisp2/landlab_grid.pickle', 'wb') as g:
    pickle.dump(tmg, g)

#################################
# Step 2: Define update routine #
#################################

def update(state, dt: float):
    eroder = GlacialEroder(state)
    state = eroder.update(dt).state

    fringe = FrozenFringe(state)
    state = fringe.update(dt).state

    advect_fringe = TVDAdvector(
        state.grid,
        state.sliding_velocity,
        state.fringe_thickness
    )
    state = eqx.tree_at(
        lambda t: t.fringe_thickness,
        state,
        advect_fringe.update(dt).tracer
    )

    advect_disp = TVDAdvector(
        state.grid,
        state.sliding_velocity,
        state.dispersed_thickness
    )
    state = eqx.tree_at(
        lambda t: t.dispersed_thickness,
        state,
        advect_disp.update(dt).tracer
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

with open('./examples/cores/gisp2/landlab_grid.pickle', 'rb') as g:
    grid = pickle.load(g)

glacier = 'kangilliup-sermia'

for Nc in [0.95, 0.9, 0.8, 0.7, 0.6]:
    print('Water pressure = ', Nc, ' * overburden pressure.')

    title = glacier.replace('-', ' ').title()        
    print('Simulating... ' + title)

    Pw = grid.at_node['water_pressure'][:] * Nc
    static = freeze_grid(grid)

    state = ModelState(
        static,
        grid.at_node['ice_thickness'],
        grid.at_node['surface_elevation'],
        grid.at_link['sliding_velocity'],
        grid.at_node['geothermal_heat_flux'],
        Pw
    )

    terminus = constrain_terminus(state, bounds[glacier][0], bounds[glacier][1], bounds[glacier][2], bounds[glacier][3])

    adj_terminus = jnp.unique(state.grid.adjacent_nodes_at_node[terminus == 1])
    adj_terminus = adj_terminus.at[adj_terminus != -1].get()
    cross_terminus_links = terminus[state.grid.node_at_link_head] * terminus[state.grid.node_at_link_tail]
    len_terminus = jnp.sum(jnp.where(cross_terminus_links, state.grid.length_of_link, 0.0))
    height_terminus = jnp.mean(state.ice_thickness[terminus])
    terminus_velocity = jnp.max(jnp.abs(state.sliding_velocity))
    ice_flux = height_terminus * len_terminus * terminus_velocity * 917 * 1e-12
    ice_fluxes = (
        jnp.sum(
            state.ice_thickness[terminus == 1]
            * jnp.mean(state.grid.length_of_link[cross_terminus_links])
            * jnp.max(jnp.abs(state.sliding_velocity))
            * state.ice_density
            * 1e-12
        ),
        jnp.mean(state.grid.length_of_link[cross_terminus_links]) * jnp.max(jnp.abs(state.sliding_velocity))
    )

    xish = ((jnp.abs(jnp.max(state.grid.node_x) - jnp.min(state.grid.node_x)))) / 1e4
    yish = ((jnp.abs(jnp.max(state.grid.node_y) - jnp.min(state.grid.node_y)))) / 1e4
    figsize = (xish, yish)

    plt.rcParams.update({'font.size': np.sqrt(xish**2 + yish**2)})
    plt.rcParams.update({'axes.linewidth': np.sqrt(xish**2 + yish**2) / 10})

    if Nc == 0.95:
        fig = plot_triangle_mesh(grid, state.ice_thickness, subplots_args = {'figsize': figsize}, show = False, cmap = cmc.batlow)
        plt.title(title + ' ice thickness (m)')
        plt.tick_params(axis = 'x', rotation = 25)
        plt.tick_params(axis = 'y', rotation = 25)
        plt.xlabel('Grid x-coordinate')
        plt.ylabel('Grid y-coordinate')
        plt.tight_layout()
        plt.savefig('./examples/cores/gisp2/results/png/' + glacier + '.png', dpi = 300)
        plt.close('all')
        np.savetxt('./examples/cores/gisp2/results/txt/' + glacier + '_' + str(int(Nc * 100)) + '.txt', np.asarray(state.ice_thickness))

        fig = plot_triangle_mesh(grid, state.grid.map_mean_of_links_to_node(jnp.abs(state.sliding_velocity)), subplots_args = {'figsize': figsize}, show = False, cmap = cmc.batlow)
        plt.title(title + ' sliding velocity (m a$^{-1}$)')
        plt.tick_params(axis = 'x', rotation = 25)
        plt.tick_params(axis = 'y', rotation = 25)
        plt.xlabel('Grid x-coordinate')
        plt.ylabel('Grid y-coordinate')
        plt.tight_layout()
        plt.savefig('./examples/cores/gisp2/results/png/' + glacier + '.png', dpi = 300)
        plt.close('all')
        np.savetxt('./examples/cores/gisp2/results/txt/' + glacier + '_' + str(int(Nc * 100)) + '.txt', state.grid.map_mean_of_links_to_node(jnp.abs(state.sliding_velocity)))

    eroder = GlacialEroder(state)
    fig = plot_triangle_mesh(grid, eroder.calc_abrasion_rate() + eroder.calc_quarrying_rate(), subplots_args = {'figsize': figsize}, show = False, cmap = cmc.batlow, norm = matplotlib.colors.LogNorm())
    plt.title(title + ' erosion rate (m a$^{-1}$)')
    plt.tick_params(axis = 'x', rotation = 25)
    plt.tick_params(axis = 'y', rotation = 25)
    plt.xlabel('Grid x-coordinate')
    plt.ylabel('Grid y-coordinate')
    plt.tight_layout()
    plt.savefig('./examples/cores/gisp2/results/png/' + glacier + '_' + str(int(Nc * 100)) + '.png', dpi = 300)
    plt.close('all')

    fig = plot_triangle_mesh(grid, state.melt_rate, subplots_args = {'figsize': figsize}, show = False, cmap = cmc.batlow)
    plt.title(title + ' melt rate (m a$^{-1}$)')
    plt.tick_params(axis = 'x', rotation = 25)
    plt.tick_params(axis = 'y', rotation = 25)
    plt.xlabel('Grid x-coordinate')
    plt.ylabel('Grid y-coordinate')
    plt.tight_layout()
    plt.savefig('./examples/cores/gisp2/results/png/' + glacier + '_' + str(int(Nc * 100)) + '.png', dpi = 300)
    plt.close('all')

    C = 0.1

    dt = C * jnp.nanmin(jnp.where(
        state.sliding_velocity != 0,
        state.grid.length_of_link / jnp.abs(state.sliding_velocity),
        np.nan
    ))
    print('dt = ', dt)

    time_elapsed = 0.0
    n_years = 500.0

    for i in range(int(n_years / dt)):
        state = update(state, dt)

        time_elapsed += dt
        if time_elapsed > n_years:
            print('Total time elapsed: ', time_elapsed)
            break

    print('Finished simulation for ' + glacier.replace('-', ' ').title())

    patch_vals = grid.map_mean_of_patch_nodes_to_patch(state.fringe_thickness)
    vmax = jnp.percentile(patch_vals, 99.5)

    fig = plot_triangle_mesh(
        grid, 
        state.fringe_thickness, 
        subplots_args = {'figsize': figsize}, 
        show = False, 
        cmap = cmc.batlow,
        set_clim = {'vmin': 0, 'vmax': vmax}
    )
    plt.title(title + ' fringe thickness (m)')
    plt.tick_params(axis = 'x', rotation = 25)
    plt.tick_params(axis = 'y', rotation = 25)
    plt.xlabel('Grid x-coordinate')
    plt.ylabel('Grid y-coordinate')
    plt.tight_layout()
    plt.savefig('./examples/cores/gisp2/results/png/' + glacier + '_' + str(int(Nc * 100)) + '.png', dpi = 300)
    plt.close('all')
    np.savetxt('./examples/cores/gisp2/results/txt/' + glacier + '_' + str(int(Nc * 100)) + '.txt', np.asarray(state.fringe_thickness))

    fig = plot_triangle_mesh(grid, state.till_thickness, subplots_args = {'figsize': figsize}, show = False, cmap = cmc.batlow)
    plt.title(title + ' till thickness (m)')
    plt.tick_params(axis = 'x', rotation = 25)
    plt.tick_params(axis = 'y', rotation = 25)
    plt.xlabel('Grid x-coordinate')
    plt.ylabel('Grid y-coordinate')
    plt.tight_layout()
    plt.savefig('./examples/cores/gisp2/results/png/' + glacier + '_' + str(int(Nc * 100)) + '.png', dpi = 300)
    plt.close('all')

    results['glacier'].append(glacier)
    results['region'].append('CW')
    results['N_scalar'].append(Nc)
    results['ice_flux'].append(discharge[glacier])
    results['boundary_flux'].append(float(jnp.sum(state.fringe_thickness[terminus == 1] - state.min_fringe_thickness) * len_terminus * 2700 * 0.6))      
    results['proximal_flux'].append(float(jnp.sum(state.fringe_thickness[adj_terminus] - state.min_fringe_thickness) * len_terminus * 2700 * 0.6)) 
    results['sediment_flux'].append(float(results['boundary_flux'][-1] + results['proximal_flux'][-1]))
    results['max_velocity'].append(float(jnp.max(state.sliding_velocity)))
    results['max_pressure'].append(float(jnp.max(state.effective_pressure)))
    results['drainage_area'].append(float(jnp.sum(state.grid.cell_area_at_node) * 1e-6))
    results['mean_fringe'].append(float(jnp.mean(state.fringe_thickness)))
    results['med_fringe'].append(float(jnp.median(state.fringe_thickness)))

    for key, val in results.items():
        print(key, val[-1])

df = pd.DataFrame.from_dict(results)
df.to_csv('./examples/cores/gisp2/results/results.csv')

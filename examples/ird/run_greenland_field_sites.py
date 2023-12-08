"""Frozen fringe models for Greenland tidewater glaciers.

Runs a series of numerical experiments over three fjord systems in Greenland.
Sites were chosen based on available field data from the 2019-2022 seasons,
and include representative examples from both east and west coasts. Here,
we model frozen fringe development, advection, and eventual export to the fjord.
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
    'sermeq-avannarleq': [-2.063e5, -1.94e5, -2.175e6, -2.17102e6],
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

regions = {
    'rolige-brae': 'CE',
    'sermeq-avannarleq': 'CW',
    'charcot-gletscher': 'CE',
    'sydbrae': 'CE',
    'kangiata-nunaata-sermia': 'SW',
    'eielson-gletsjer': 'CE',
    'narsap-sermia': 'SW',
    'kangilernata-sermia': 'CW',
    'dode-brae': 'CE',
    'daugaard-jensen-gletsjer': 'CE',
    'vestfjord-gletsjer': 'CE',
    'sermeq-kullajeq': 'CW',
    'bredegletsjer': 'CE',
    'magga-dan-gletsjer': 'CE',
    'graah-gletscher': 'CE',
    'akullersuup-sermia': 'SW',
    'eqip-sermia': 'CW',
    'kista-dan-gletsjer': 'CE'
}

discharge_gate = {
    'rolige-brae': 150,
    'sermeq-avannarleq': 167,
    'charcot-gletscher': 139,
    'sydbrae': 152,
    'kangiata-nunaata-sermia': 275,
    'eielson-gletsjer': 144,
    'narsap-sermia': 262,
    'kangilernata-sermia': 177,
    'dode-brae': 160, # proxy
    'daugaard-jensen-gletsjer': 140,
    'vestfjord-gletsjer': 154,
    'sermeq-kullajeq': 169,
    'bredegletsjer': 151,
    'magga-dan-gletsjer': 156,
    'graah-gletscher': 138,
    'akullersuup-sermia': 270,
    'eqip-sermia': 180,
    'kista-dan-gletsjer': 158
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
            np.sign(Ud) * (np.abs(Us) - np.abs(Ud))
        )

        tmg.add_field('sliding_velocity', Ub, at = 'link')
        tmg.add_field('water_pressure', H * 917 * 9.81, at = 'node')

        state = initialize_state_from_grid(tmg)
        grids[glacier] = tmg

        print('Loaded data for ' + glacier.replace('-', ' ').title())

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

with open('./examples/ird/landlab_grids.pickle', 'rb') as g:
    grids = pickle.load(g)

for Nc in [0.95, 0.9, 0.8, 0.7, 0.6]:
    print('Water pressure = ', Nc, ' * overburden pressure.')

    for glacier, grid in grids.items():
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

        if glacier == 'eielson-gletsjer':
            terminus_a = constrain_terminus(state, bounds[glacier][0][0], bounds[glacier][0][1], bounds[glacier][0][2], bounds[glacier][0][3])
            terminus_b = constrain_terminus(state, bounds[glacier][1][0], bounds[glacier][1][1], bounds[glacier][1][2], bounds[glacier][1][3])
            terminus = terminus_a | terminus_b
        else:
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

        if glacier == 'magga-dan-gletsjer':
            figsize = (yish, xish * 1.5)

        plt.rcParams.update({'font.size': np.sqrt(xish**2 + yish**2)})
        plt.rcParams.update({'axes.linewidth': np.sqrt(xish**2 + yish**2) / 10})

        if Nc == 0.95:
            fig = plot_triangle_mesh(grids[glacier], state.ice_thickness, subplots_args = {'figsize': figsize}, show = False, cmap = cmc.batlow)
            plt.title(title + ' ice thickness (m)')
            plt.tick_params(axis = 'x', rotation = 25)
            plt.tick_params(axis = 'y', rotation = 25)
            plt.xlabel('Grid x-coordinate')
            plt.ylabel('Grid y-coordinate')
            plt.tight_layout()
            plt.savefig('./examples/ird/icethk/' + glacier + '.png', dpi = 300)
            plt.close('all')
            np.savetxt('./examples/ird/icethk/' + glacier + '_' + str(int(Nc * 100)) + '.txt', np.asarray(state.ice_thickness))

            fig = plot_triangle_mesh(grids[glacier], state.grid.map_mean_of_links_to_node(jnp.abs(state.sliding_velocity)), subplots_args = {'figsize': figsize}, show = False, cmap = cmc.batlow)
            plt.title(title + ' sliding velocity (m a$^{-1}$)')
            plt.tick_params(axis = 'x', rotation = 25)
            plt.tick_params(axis = 'y', rotation = 25)
            plt.xlabel('Grid x-coordinate')
            plt.ylabel('Grid y-coordinate')
            plt.tight_layout()
            plt.savefig('./examples/ird/sliding/' + glacier + '.png', dpi = 300)
            plt.close('all')
            np.savetxt('./examples/ird/sliding/' + glacier + '_' + str(int(Nc * 100)) + '.txt', state.grid.map_mean_of_links_to_node(jnp.abs(state.sliding_velocity)))

        eroder = GlacialEroder(state)
        fig = plot_triangle_mesh(grids[glacier], eroder.calc_abrasion_rate() + eroder.calc_quarrying_rate(), subplots_args = {'figsize': figsize}, show = False, cmap = cmc.batlow, norm = matplotlib.colors.LogNorm())
        plt.title(title + ' erosion rate (m a$^{-1}$)')
        plt.tick_params(axis = 'x', rotation = 25)
        plt.tick_params(axis = 'y', rotation = 25)
        plt.xlabel('Grid x-coordinate')
        plt.ylabel('Grid y-coordinate')
        plt.tight_layout()
        plt.savefig('./examples/ird/erosion/' + glacier + '_' + str(int(Nc * 100)) + '.png', dpi = 300)
        plt.close('all')

        fig = plot_triangle_mesh(grids[glacier], state.melt_rate, subplots_args = {'figsize': figsize}, show = False, cmap = cmc.batlow)
        plt.title(title + ' melt rate (m a$^{-1}$)')
        plt.tick_params(axis = 'x', rotation = 25)
        plt.tick_params(axis = 'y', rotation = 25)
        plt.xlabel('Grid x-coordinate')
        plt.ylabel('Grid y-coordinate')
        plt.tight_layout()
        plt.savefig('./examples/ird/melt/' + glacier + '_' + str(int(Nc * 100)) + '.png', dpi = 300)
        plt.close('all')

        if glacier in ['narsap-sermia', 'vestfjord-gletsjer', 'daugaard-jensen-gletsjer']:
            C = 0.05
        else:
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

        patch_vals = grids[glacier].map_mean_of_patch_nodes_to_patch(state.fringe_thickness)
        vmax = jnp.percentile(patch_vals, 99.5)

        fig = plot_triangle_mesh(
            grids[glacier], 
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
        plt.savefig('./examples/ird/fringe/' + glacier + '_' + str(int(Nc * 100)) + '.png', dpi = 300)
        plt.close('all')
        np.savetxt('./examples/ird/output/' + glacier + '_' + str(int(Nc * 100)) + '.txt', np.asarray(state.fringe_thickness))

        fig = plot_triangle_mesh(grids[glacier], state.till_thickness, subplots_args = {'figsize': figsize}, show = False, cmap = cmc.batlow)
        plt.title(title + ' till thickness (m)')
        plt.tick_params(axis = 'x', rotation = 25)
        plt.tick_params(axis = 'y', rotation = 25)
        plt.xlabel('Grid x-coordinate')
        plt.ylabel('Grid y-coordinate')
        plt.tight_layout()
        plt.savefig('./examples/ird/till/' + glacier + '_' + str(int(Nc * 100)) + '.png', dpi = 300)
        plt.close('all')

        results['glacier'].append(glacier)
        results['region'].append(regions[glacier])
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
df.to_csv('./examples/ird/results.csv')

"""Test the ConduitHydrology implementation."""

import numpy as np
from numpy.testing import *
import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx
import pytest

from landlab import TriangleModelGrid
from utils import StaticGrid, freeze_grid
from utils.plotting import plot_links, plot_triangle_mesh
from components import SubglacialDrainageSystem, ModelState

def make_grid():
    """Create a simple unstructured grid."""
    g = TriangleModelGrid(
        (
            [1, 1, 20e3, 20e3],
            [1, 60e3, 60e3, 1]
        ),
        triangle_opts = 'pqDevjza500000q26',
        sort = False
    )
    static = freeze_grid(g)

    g.add_field(
        'surface_elevation', 
        np.sqrt(g.node_x + 10) * (1500 / np.sqrt(np.max(g.node_x))),
        at = 'node'
    )

    rng = np.random.default_rng(135)
    g.add_field(
        'bedrock_elevation',
        rng.random(g.number_of_nodes) + (np.max(g.node_x) - g.node_x) * 1e-3,
        at = 'node'
    )

    g.add_field(
        'ice_thickness',
        g.at_node['surface_elevation'] - g.at_node['bedrock_elevation'],
        at = 'node'
    )
    
    g.add_field(
        'sliding_velocity',
        np.full(g.number_of_links, 1e-6) * 31556926,
        at = 'link'
    )

    g.add_field(
        'water_pressure',
        g.at_node['ice_thickness'] * 9.81 * 917 * 0.8,
        at = 'node'
    )
    
    g.add_field(
        'geothermal_heat_flux',
        np.full(g.number_of_nodes, 0.05 * 31556926),
        at = 'node'
    )

    g.add_field(
        'surface_melt_rate',
        np.maximum(
            1.62e-6 - (g.at_node['surface_elevation'] * 1e-3) * 1.16e-6,
            0.0
        ),
        at = 'node'
    )

    return g

@pytest.fixture
def grid():
    return make_grid()

@pytest.fixture
def state(grid):
    """Instantiate the ModelState."""
    return ModelState(
        freeze_grid(grid),
        grid.at_node['ice_thickness'],
        grid.at_node['surface_elevation'],
        grid.at_link['sliding_velocity'],
        grid.at_node['geothermal_heat_flux'],
        grid.at_node['water_pressure']
    )

@pytest.fixture
def model(state, grid):
    """Create an instance of the SubglacialDrainageSystem model."""
    return SubglacialDrainageSystem(
        state, 
        grid.at_node['surface_melt_rate'],
        state.overburden_pressure * 0.2,
        np.full(grid.number_of_nodes, 0.05),
        np.full(grid.number_of_links, 0.0),
    )

def test_channel_source(grid, model):
    m = model.exchange_term(grid.node_y**2, model.sheet_thickness)

    plot_triangle_mesh(grid, grid.node_y**2, subplots_args={'figsize': (18, 4)})

    plot_links(grid, m, subplots_args={'figsize': (18, 4)})

# def test_tmp(grid, model):

#     model = model.update(60.0)

#     import time
#     start = time.time()
#     model = model.update(60.0)
#     print('Iteration time:', time.time() - start)

#     for i in range(5):
#         model = model.update(60.0 * 60.0)

#         print('Iteration', i)

#     plot_triangle_mesh(
#         grid,
#         model.base_potential - model.potential,
#         subplots_args = {'figsize': (18, 4)},
#         title = 'Effective pressure (Pa)'
#     )

#     Qc = jnp.abs(
#         -model.channel_conductivity
#         * model.channel_size**model.flow_exp
#         * model.grid.calc_grad_at_link(model.potential)
#     )
#     plot_links(
#         grid,
#         Qc,
#         subplots_args = {'figsize': (18, 4)},
#         title = 'Channelized discharge (m$^3$ s$^{-1}$)'
#     )

#     gradient = model.grid.map_mean_of_links_to_node(
#         model.grid.calc_grad_at_link(model.potential)
#     )
#     Qs = jnp.abs(
#         -model.sheet_conductivity
#         * model.sheet_thickness**model.flow_exp
#         * jnp.power(jnp.abs(gradient), -1/2)
#         * gradient
#     )
#     plot_triangle_mesh(
#         grid,
#         Qs,
#         subplots_args = {'figsize': (18, 4)},
#         title = 'Distributed discharge (m$^3$ s$^{-1}$)'
#     )

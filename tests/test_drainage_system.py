"""Test the SubglacialDrainageSystem implementation."""

import numpy as np
from numpy.testing import *
import jax
import jax.numpy as jnp
import pytest

from landlab import TriangleModelGrid
from utils import StaticGrid, freeze_grid
from components import SubglacialDrainageSystem, ModelState

import matplotlib.pyplot as plt
from utils.plotting import plot_triangle_mesh, plot_links

def make_grid():
    """Create a simple unstructured grid."""
    g = TriangleModelGrid(
        (
            [-1.01, -1, 1, 1],
            [1, -1, -1, 1]
        ),
        triangle_opts = 'pqDevjza0.5q26',
        sort = False
    )
    static = freeze_grid(g)

    g.add_field(
        'surface_elevation', 
        g.node_x + 100,
        at = 'node'
    )

    rng = np.random.default_rng(135)
    g.add_field(
        'bedrock_elevation',
        rng.random(g.number_of_nodes) * 60,
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
        g.at_node['ice_thickness'] * 9.81 * 917 * 0.2,
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
        grid.at_node['bedrock_elevation'] * 1000 * 9.81,
        np.full(grid.number_of_nodes, 0.05),
        np.full(grid.number_of_links, 1e-3)
    )

def test_initialize(grid, model):
    """Test the initialization of the model."""
    assert model.grid == freeze_grid(grid)
    assert model.surface_melt_rate.shape == (grid.number_of_nodes,)
    assert model.potential.shape == (grid.number_of_nodes,)
    assert model.sheet_thickness.shape == (grid.number_of_nodes,)
    assert model.channel_size.shape == (grid.number_of_links,)
    assert model.links_between_nodes.shape == (grid.number_of_nodes, grid.number_of_nodes)


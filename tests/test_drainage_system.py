"""Test the ConduitHydrology implementation."""

import numpy as np
from numpy.testing import *
import jax
import jax.numpy as jnp
import equinox as eqx
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
        triangle_opts = 'pqDevjza200000q26',
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
        grid, 
        grid.at_node['surface_melt_rate'],
        state.overburden_pressure * 0.2,
        np.full(grid.number_of_nodes, 1e-7)
    )

def test_set_potential(grid, model):
    """Test the base potential calculation."""
    assert model.base_potential.shape == (grid.number_of_nodes,)

def test_route_flow(grid, model):
    """Test the flow routing."""
    assert model.discharge.shape == (grid.number_of_links,)
    assert np.all(model.discharge >= 0)
    assert model.flow_direction.shape == (grid.number_of_links,)
    assert np.all(np.isin(model.flow_direction, [-1, 1]))

def test_tmp(grid, model):

    print(model.grid.get_unit_normal_at_links())
    

# def test_solve(grid, model):
#     """Test the potential solver."""
#     for i in range(10):
#         model = model.update(60*60*24)

#         plot_triangle_mesh(
#             grid, 
#             model.potential, 
#             subplots_args={'figsize': (18, 4)},
#             title = 'Hydraulic potential (Pa)'
#         )
#         plot_triangle_mesh(
#             grid, 
#             model.sheet_thickness, 
#             subplots_args={'figsize': (18, 4)},
#             title = 'Mean thickness of sheet flow (m)'
#         )
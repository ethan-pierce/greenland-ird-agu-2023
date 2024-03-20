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
            [-1000, 1000, 1000, -1000],
            [1000, 1000, -1000, -1000]
        ),
        triangle_opts = 'pqDevjza1000q26',
        sort = False
    )
    static = freeze_grid(g)

    g.add_field(
        'surface_elevation', 
        (np.max(g.node_x) - g.node_x)**(1/3) * 50 + 5.,
        at = 'node'
    )

    rng = np.random.default_rng(135)
    g.add_field(
        'bedrock_elevation',
        (np.max(g.node_x) - g.node_x) * 0.05 + (rng.random(g.number_of_nodes)),
        at = 'node'
    )

    g.add_field(
        'ice_thickness',
        g.at_node['surface_elevation'] - g.at_node['bedrock_elevation'],
        at = 'node'
    )
    
    g.add_field(
        'sliding_velocity',
        g.map_mean_of_link_nodes_to_link((np.max(g.node_x) + g.node_x) * 0.5),
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
        np.full(g.number_of_nodes, 3.45e-8),
        at = 'node'
    ) # in m/s, corresponds to 0.25 cm/day

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
        np.full(state.grid.number_of_nodes, 0.05),
        grid.at_node['surface_melt_rate']
    )

def test_set_potential(grid, model):
    """Test the base potential calculation."""
    assert model.base_potential.shape == (grid.number_of_nodes,)
    assert np.all(model.state.overburden_pressure <= model.base_potential)

def test_route_flow(grid, model):
    """Test the flow routing."""
    assert model.discharge.shape == (grid.number_of_links,)
    assert np.all(model.discharge >= 0)
    assert model.flow_direction.shape == (grid.number_of_links,)
    assert np.all(np.isin(model.flow_direction, [-1, 1]))

# Runs slowly
# def test_solve_for_potential(grid, model):
#     """Test the potential solver."""
#     potential = model._solve_for_potential(jnp.full(grid.number_of_nodes, 0.05))

#     assert potential.shape == (grid.number_of_nodes,)

def test_sheet_growth_rate(grid, model):
    """Test the sheet growth rate."""
    dhdt = model._calc_sheet_growth_rate(
        jnp.full(grid.number_of_nodes, 0.05),
        model.state.overburden_pressure
    )

    assert dhdt.shape == (grid.number_of_nodes,)

def test_update_sheet_flow(grid, model):
    """Test the sheet flow update."""
    for i in range(2):
        model = model._update_sheet_flow(60.0)

    assert model.sheet_thickness.shape == (grid.number_of_nodes,)

    plot_triangle_mesh(grid, model.sheet_thickness)
    plot_triangle_mesh(grid, model.potential)
    plot_triangle_mesh(grid, model.base_potential - model.potential)
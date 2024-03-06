"""Test the ConduitHydrology implementation."""

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_almost_equal
import jax
import jax.numpy as jnp
import pytest

from landlab import TriangleModelGrid
from utils import StaticGrid, freeze_grid
from utils.plotting import plot_triangle_mesh, plot_links, timer
from components import ConduitHydrology, ModelState

@pytest.fixture
def grid():
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
        (np.max(g.node_x) - g.node_x)**(1/3) * 50,
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
        g.map_mean_of_link_nodes_to_link(np.max(g.node_x) + g.node_x),
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

    return g

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
def conduits(state, grid):
    """Create an instance of the ConduitHydrology model."""
    return ConduitHydrology(
        state,
        grid,
        np.full(state.grid.number_of_nodes, 1e-3),
    )

def test_flow_routing(grid, conduits):
    """Test the flow director."""
    discharge = conduits.discharge
    assert discharge.shape == (grid.number_of_nodes,)

def test_hydraulic_gradient(grid, conduits):
    """Test the hydraulic gradient."""
    gradient = conduits.calc_hydraulic_gradient(jnp.full(grid.number_of_nodes, 1.0))
    assert gradient.shape == (grid.number_of_nodes,)

def test_solve_for_potential(grid, conduits):
    """Test the hydraulic potential field."""
    potential = conduits.solve_for_potential(jnp.full(grid.number_of_nodes, 1.0))
    assert potential.shape == (grid.number_of_nodes,)

def test_effective_pressure(grid, conduits):
    """Test the effective pressure."""
    effective_pressure = conduits.calc_effective_pressure(jnp.full(grid.number_of_nodes, 1.0))
    assert effective_pressure.shape == (grid.number_of_nodes,)

def test_conduits_roc(grid, conduits):
    """Test the rate of closure of the conduits."""
    roc = conduits.calc_conduits_roc(jnp.full(grid.number_of_nodes, 1.0))
    assert roc.shape == (grid.number_of_nodes,)

def test_update_conduits(grid, conduits):
    """Test the update of the conduits."""
    conduits = conduits.update_conduits(dt = 1.0)
    assert conduits.conduit_size.shape == (grid.number_of_nodes,)


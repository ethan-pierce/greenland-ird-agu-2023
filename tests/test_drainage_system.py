"""Test the ConduitHydrology implementation."""

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_almost_equal
import jax
import jax.numpy as jnp
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
        np.full(state.grid.number_of_nodes, 1e-3),
    )

def test_flow_routing(grid, model):
    """Test the flow director."""
    discharge = model.discharge
    assert discharge.shape == (grid.number_of_nodes,)

def test_hydraulic_gradient(grid, model):
    """Test the hydraulic gradient."""
    gradient = model._calc_hydraulic_gradient(jnp.full(grid.number_of_nodes, 0.1))
    assert gradient.shape == (grid.number_of_nodes,)
    
def test_solve_for_potential(grid, model):
    """Test the hydraulic potential field."""
    gradient = model._calc_hydraulic_gradient(jnp.full(grid.number_of_nodes, 0.1))
    potential = model._solve_for_potential(gradient)
    assert potential.shape == (grid.number_of_nodes,)

def test_effective_pressure(grid, model):
    """Test the effective pressure."""
    gradient = model._calc_hydraulic_gradient(jnp.full(grid.number_of_nodes, 0.1))
    potential = model._solve_for_potential(gradient)
    effective_pressure = model._calc_effective_pressure(potential)
    assert effective_pressure.shape == (grid.number_of_nodes,)

def test_conduits_roc(grid, model):
    """Test the rate of closure of the conduits."""
    roc = model._calc_conduits_roc(jnp.full(grid.number_of_nodes, 0.1))
    assert roc.shape == (grid.number_of_nodes,)

def test_update_conduits(grid, model):
    """Test the update of the conduits."""
    model = model.update_conduits(dt = 1.0)
    assert model.conduit_size.shape == (grid.number_of_nodes,)

if __name__ == '__main__':
    """Run a test case for an archetypal glacier margin."""
    grid = make_grid()
    state = ModelState(
        freeze_grid(grid),
        grid.at_node['ice_thickness'],
        grid.at_node['surface_elevation'],
        grid.at_link['sliding_velocity'],
        grid.at_node['geothermal_heat_flux'],
        grid.at_node['water_pressure']
    )
    model = SubglacialDrainageSystem(
        state,
        grid,
        np.full(state.grid.number_of_nodes, 1e-3),
    )

    for i in range(60):
        model = model.update_conduits(dt = 60.0)

        if i % 100 == 0:
            print(f"Iteration {i+1}: Mean Conduit Size = {np.mean(model.conduit_size):.2f}")

    plot_triangle_mesh(grid, model.get_state()['effective_pressure'])
    plot_triangle_mesh(grid, model.conduit_size)
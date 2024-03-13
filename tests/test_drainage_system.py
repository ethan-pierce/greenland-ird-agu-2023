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
        np.full(state.grid.number_of_nodes, 1e-3),
        grid.at_node['surface_melt_rate']
    )

def test_base_potential(grid, model):
    """Test the base potential calculation."""
    assert model.base_potential.shape == (grid.number_of_nodes,)
    assert np.all(model.state.overburden_pressure <= model.base_potential)

def test_route_flow(grid, model):
    """Test the flow routing."""
    assert model.discharge.shape == (grid.number_of_links,)
    assert np.all(model.discharge >= 0)
    assert model.flow_direction.shape == (grid.number_of_links,)
    assert np.all(np.isin(model.flow_direction, [-1, 1]))

def test_hydraulic_gradient(grid, model):
    """Test the hydraulic gradient calculation."""
    gradient = model._calc_hydraulic_gradient(jnp.full(grid.number_of_links, 1.0))

    assert gradient.shape == (grid.number_of_links,)
    assert np.all(np.sign(gradient) == np.sign(model.flow_direction * model.discharge))

def test_solve_for_potential(grid, model):
    """Test the potential field solution."""
    conduit_size = jnp.full(grid.number_of_links, 1.0)
    gradient = model._calc_hydraulic_gradient(conduit_size)
    potential = model._solve_for_potential(gradient)

    assert potential.shape == (grid.number_of_nodes,)

def test_effective_pressure(grid, model):
    """Test the effective pressure calculation."""
    conduit_size = jnp.full(grid.number_of_links, 1.0)
    gradient = model._calc_hydraulic_gradient(conduit_size)
    potential = model._solve_for_potential(gradient)
    pressure = model._calc_effective_pressure(potential)

    assert pressure.shape == (grid.number_of_nodes,)

def test_conduit_size(grid, model):
    """Test the conduit size calculation."""
    conduit_size = jnp.full(grid.number_of_links, 1.0)
    gradient = model._calc_hydraulic_gradient(conduit_size)
    potential = model._solve_for_potential(gradient)
    pressure = model._calc_effective_pressure(potential)
    updated_conduit_size = model._calc_conduit_size(gradient, pressure)

    assert updated_conduit_size.shape == (grid.number_of_links,)
    assert np.all(updated_conduit_size > 0)

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
        np.full(state.grid.number_of_links, 1e-3),
        grid.at_node['surface_melt_rate']
    )
    initial_conduit_size = model.discharge / np.max(model.discharge) * 0.1 + 1e-3
    model = eqx.tree_at(
        lambda tree: tree.conduit_size,
        model,
        initial_conduit_size
    )

    S = initial_conduit_size
    for i in range(5):
        S += model._calc_conduit_roc(S) * 1.0

        plot_links(grid, S, title = 'Conduit size (m$^2$)')
        
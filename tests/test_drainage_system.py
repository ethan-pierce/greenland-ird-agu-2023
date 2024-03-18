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

    import jaxopt
    
    def residual(phi):
        grad_phi = model.grid.calc_grad_at_link(phi)

        pressure = jnp.maximum(
            model.grid.map_mean_of_link_nodes_to_link(model.base_potential - phi), 
            0.0
        )

        melt_opening = jnp.abs(model.opening_coeff * model.discharge * grad_phi * model.flow_direction)
        gap_opening = jnp.abs(model.state.sliding_velocity / model.state.sec_per_a) * model.step_height
        closure = model.closure_coeff * pressure**model.n * model.conduit_size

        conduit_size = (melt_opening + gap_opening) / (gap_opening / model.scale_cutoff + closure)

        divergence = model.grid.calc_flux_div_at_node(
            model.flow_coeff * conduit_size**model.flow_exp * jnp.sqrt(jnp.abs(grad_phi)) * model.flow_direction
        )

        residual = divergence - model.total_melt_rate

        return jnp.sum(residual**2)

    solver = jaxopt.ScipyBoundedMinimize(
        fun = residual,
        method = 'L-BFGS-B'
    )
    lower_bounds = model.state.water_density * model.state.gravity * model.state.bedrock_elevation
    upper_bounds = model.base_potential
    bounds = (lower_bounds, upper_bounds)
    solution = solver.run(jnp.zeros(grid.number_of_nodes), bounds = bounds).params

    gradient = model.grid.calc_grad_at_link(solution)
    pressure = jnp.maximum(
        model.grid.map_mean_of_link_nodes_to_link(model.base_potential - solution), 
        0.0
    )
    melt_opening = jnp.abs(model.opening_coeff * model.discharge * gradient * model.flow_direction)
    gap_opening = jnp.abs(model.state.sliding_velocity / model.state.sec_per_a) * model.step_height
    closure = model.closure_coeff * pressure**model.n * model.conduit_size

    conduit_size = (melt_opening + gap_opening) / (gap_opening / model.scale_cutoff + closure)

    divergence = model.grid.calc_flux_div_at_node(
        model.flow_coeff * conduit_size**model.flow_exp * jnp.sqrt(jnp.abs(gradient)) * model.flow_direction
    )

    residual = divergence - model.total_melt_rate

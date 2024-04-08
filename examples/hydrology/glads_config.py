"""Test the ConduitHydrology implementation."""

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx

import matplotlib.pyplot as plt

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
        triangle_opts = 'pqDevjza1000000q26',
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
        rng.random(g.number_of_nodes) + (g.node_x - np.max(g.node_x)) * 1e-3,
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

def model(state, grid):
    """Create an instance of the SubglacialDrainageSystem model."""
    return SubglacialDrainageSystem(
        state, 
        grid.at_node['surface_melt_rate'],
        state.overburden_pressure * 0.2,
        np.full(grid.number_of_nodes, 0.05),
        np.full(grid.number_of_links, 0.0),
    )

if __name__ == '__main__':
    print('Generating grid...')
    grid = make_grid()

    print('Instantiating model...')
    state = state(grid)
    model = model(state, grid)

    init_forcing = model.build_forcing_vector(model.potential, model.sheet_thickness, model.channel_size)
    _, updated_forcing = model.assemble_linear_system(model.potential, model.sheet_thickness, model.channel_size)

    altered = jnp.where(init_forcing != updated_forcing, 1, 0)

    im = plt.scatter(
        grid.node_x,
        grid.node_y,
        c = jnp.where(model.grid.cell_at_node != -1, altered[model.grid.cell_at_node], jnp.nan)
    )
    plt.colorbar(im)
    plt.show()


    # print('Running model...')
    # for i in range(50):
    #     model = model.update(60.0 * 60.0 * 6)

    #     if i % 10 == 0:
    #         print('Completed iteration', i)

    # plot_triangle_mesh(
    #     grid,
    #     model.sheet_thickness,
    #     subplots_args = {'figsize': (18, 4)}
    # )

    # plot_triangle_mesh(
    #     grid,
    #     model.grid.map_mean_of_links_to_node(
    #         jnp.abs(model.sheet_discharge_on_links(model.potential, model.sheet_thickness))
    #     ),
    #     subplots_args = {'figsize': (18, 4)}
    # )

    # plot_links(
    #     grid,
    #     jnp.log10(model.channel_size),
    #     subplots_args = {'figsize': (18, 4)}
    # )

    # Q = jnp.abs(model.channel_discharge(model.potential, model.channel_size))
    # plot_links(
    #     grid,
    #     jnp.where(Q > jnp.percentile(Q, 90), 2, jnp.where(Q > jnp.percentile(Q, 80), 1, 0)),
    #     subplots_args = {'figsize': (18, 4)}
    # )

    # plot_triangle_mesh(
    #     grid,
    #     model.potential,
    #     subplots_args = {'figsize': (18, 4)}
    # )
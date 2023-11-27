import pytest
import numpy as np
import jax
import jax.numpy as jnp

from landlab import RasterModelGrid
from landlab.plot import imshow_grid
from utils import StaticGrid
from components import ModelState


@pytest.fixture
def grid():
    rmg = RasterModelGrid((5, 10))

    rmg.add_field('ice_thickness', 500 - rmg.node_x * 10, at = 'node')
    rmg.add_field('surface_elevation', rmg.at_node['ice_thickness'][:], at = 'node')
    rmg.add_field(
        'sliding_velocity_vector', 
        (rmg.node_x * 10, np.ones(rmg.number_of_nodes)),
        at = 'node'
    )
    rmg.add_field('geothermal_heat_flux', np.full(rmg.number_of_nodes, 0.05), at = 'node')
    rmg.add_field('water_pressure', rmg.at_node['ice_thickness'][:] * 917 * 9.81 * 0.8, at = 'node')

    return rmg

@pytest.fixture
def state(grid):
    state = ModelState.from_grid(grid)
    return state
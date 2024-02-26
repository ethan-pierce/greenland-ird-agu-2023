""'Unit tests for components/steady_state_hydrology.py'""

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest
from landlab import TriangleModelGrid

from components import SteadyStateHydrology
from utils import StaticGrid, freeze_grid
from utils.plotting import plot_triangle_mesh, plot_links

import matplotlib.pyplot as plt

@pytest.fixture
def grid():
    g = TriangleModelGrid(
        (
            [-1001, 1000, 1000, -1000],
            [1000, 1000, -1000, -1000]
        ),
        triangle_opts = 'pqDevjza1000q20',
        sort = True
    )

    g.add_ones('melt_rate', at = 'node')
    g.at_node['melt_rate'] *= (np.max(g.node_x) - g.node_x) / 31556926 * g.cell_area_at_node
    
    return g

@pytest.fixture
def hydro(grid):
    static = freeze_grid(grid)
    return SteadyStateHydrology(
        grid = static, 
        melt_rate = grid.at_node['melt_rate']
    )

def test_adjacency_matrix(hydro, grid):
    assert hydro.adjacency_matrix.shape == (grid.number_of_nodes, grid.number_of_nodes)

    assert_array_equal(
        np.sum(hydro.adjacency_matrix, axis = 1),
        np.count_nonzero(grid.adjacent_nodes_at_node + 1, axis = 1)
    )

def test_route_discharge(hydro, grid):
    solution = hydro._route_discharge()
    assert solution.shape == (grid.number_of_nodes,)


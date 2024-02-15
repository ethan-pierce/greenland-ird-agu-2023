""'Unit tests for components/steady_state_hydrology.py'""

import numpy as np
from numpy.testing import assert_array_almost_equal
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
    g.at_node['melt_rate'] *= (np.max(g.node_x) - g.node_x) * g.cell_area_at_node
    
    return g

@pytest.fixture
def hydro(grid):
    static = freeze_grid(grid)
    return SteadyStateHydrology(
        grid = static, 
        melt_rate = grid.at_node['melt_rate']
    )

def test_route_discharge(hydro, grid):
    solution = hydro._route_discharge()
    
    plot_triangle_mesh(grid, solution)
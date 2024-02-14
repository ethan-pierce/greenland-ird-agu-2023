""'Unit tests for utils/unstructured_tvd.py'""

import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
from landlab import RasterModelGrid, TriangleMeshGrid

from utils import StaticGrid, TVDAdvector, freeze_grid
from utils.plotting import plot_triangle_mesh

@pytest.fixture
def grid():
    g = TriangleMeshGrid(
        (
            [-10, 10, 10, -10],
            [10, 10, -10, -10]
        ),
        triangle_opts = 'pqDevjza0.25'
    )

    g.add_field(
        "velocity",
        np.ones(g.number_of_links),
        at = 'link'
    )
    g.add_field(
        "tracer",
        np.zeros(g.number_of_nodes),
        at = 'node'
    )
    g.at_node['tracer'][
        (g.node_x < -5)
        & (g.node_x > -10)
        & (g.node_y < 5)
        & (g.node_y > -5)
    ] = 1.0

    return g

@pytest.fixture
def static_grid(grid):
    return freeze_grid(grid)

@pytest.fixture
def tvd(grid, static_grid):
    return TVDAdvector(static_grid, grid.at_link['velocity'], grid.at_node['tracer'])

def test_tvd_init(tvd, grid, static_grid):
    """Test that the TVDAdvector is initialized correctly."""
    assert tvd.grid == static_grid
    assert tvd.velocity.shape == (grid.number_of_links,)
    assert tvd.tracer.shape == (grid.number_of_nodes,)

def test_upwind_ghost_at_link(tvd):
    """Test that the upwind points are correctly identified at each link."""
    assert tvd.upwind_ghost_at_link.shape == (tvd.grid.number_of_links, 2)
    assert tvd.upwind_ghost_at_link[0] == pytest.approx([7.166, 1.920], rel = 1e-3)

def test_upwind_real_at_link(tvd):
    """Test that the upwind cells are correctly identified at each link."""
    assert tvd.upwind_real_idx.shape == (tvd.grid.number_of_links,)
    assert tvd.upwind_real_coords.shape == (tvd.grid.number_of_links, 2)

def test_upwind_shift_vector(tvd):
    """Test that the shift vector is correctly calculated at each link."""
    assert tvd.upwind_shift_vector.shape == (tvd.grid.number_of_links, 2)
    assert tvd.upwind_shift_vector[0] == pytest.approx([0.305, -0.02453], rel = 1e-3)
import pytest
import numpy as np
from numpy.testing import assert_array_equal
import jax
from landlab import RasterModelGrid
from utils import StaticGrid, TVDAdvection

def test_tvd_algorithm():
    rmg = RasterModelGrid((3, 3))
    grid = StaticGrid.from_grid(rmg)

    # Looks weird but necessary to test gradient ratio
    field = rmg.add_ones('field', at = 'node')
    field[:5] = grid.node_x[:5] * 2

    velocity = rmg.add_ones('velocity', at = 'link')
    velocity[5:7] *= -1

    tvd = TVDAdvection(grid, velocity, field)
    assert_array_equal(
        tvd.upwind_links,
        [-1, 0, -1, -1, -1, 6, -1, 2, 3, 4, -1, 10]
    )

    assert_array_equal(
        tvd.gradient_ratio,
        [1, 1, 1, 1, 1, -0.5, 1, 0, 0, 1, 1, 1]
    )

    assert_array_equal(
        tvd.flux_limiter,
        [1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1]
    )

    for i in range(10):
        tvd = tvd.update(1.0)

    assert_array_equal(
        tvd.field,
        [0, 2, 4, 0, 29526, 1, 1, 1, 1]
    )

import pytest
import numpy as np
from numpy.testing import assert_array_equal
from landlab import RasterModelGrid
from utils import StaticGrid, TVDAdvection

def test_tvd_constructor():
    rmg = RasterModelGrid((3, 3))
    grid = StaticGrid.from_grid(rmg)

    field = rmg.add_ones('field', at = 'node')

    positive_velocity = rmg.add_ones('positive_velocity', at = 'link')
    negative_velocity = rmg.add_ones('negative_velocity', at = 'link')
    negative_velocity *= -1

    tvd_pos = TVDAdvection(grid, positive_velocity, field)
    assert_array_equal(
        tvd_pos.upwind_links,
        [-1, 0, -1, -1, -1, -1, 5, 2, 3, 4, -1, 10]
    )

    assert_array_equal(
        tvd_pos.gradient_ratio,
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    )

    tvd_neg = TVDAdvection(grid, negative_velocity, field)
    assert_array_equal(
        tvd_neg.upwind_links,
        [1, -1, 7, 8, 9, 6, -1, -1, -1, -1, 11, -1]
    )

    assert_array_equal(
        tvd_neg.gradient_ratio,
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    )


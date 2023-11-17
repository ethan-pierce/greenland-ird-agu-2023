import pytest
import numpy as np
from numpy.testing import assert_array_equal
import dataclasses
import jax
from landlab import RasterModelGrid
from utils import StaticGrid

def test_static_grid():
    rmg = RasterModelGrid((3, 3))
    grid = StaticGrid.from_grid(rmg)

    for field in dataclasses.fields(grid):
        value = getattr(grid, field.name)
        if not isinstance(value, int) and not isinstance(value, tuple):
            assert isinstance(value, jax.Array)

def test_mapping_between_elements():
    rmg = RasterModelGrid((3, 3))
    grid = StaticGrid.from_grid(rmg)

    node_array = rmg.add_ones('field', at = 'node')
    node_array[4] = 3.

    assert_array_equal(
        grid.map_mean_of_link_nodes_to_link(node_array),
        [1, 1, 1, 2, 1, 2, 2, 1, 2, 1, 1, 1]
    )

    link_array = rmg.add_ones('field', at = 'link')
    link_array[5:7] = 3

    assert_array_equal(
        grid.map_mean_of_links_to_node(link_array),
        [1, 1, 1, 1.5, 2, 1.5, 1, 1, 1]
    )

def test_gradients():
    rmg = RasterModelGrid((3, 3))
    grid = StaticGrid.from_grid(rmg)

    node_array = rmg.add_ones('field', at = 'node')
    node_array[4] = 3.

    assert_array_equal(
        grid.calc_grad_at_link(node_array),
        [0, 0, 0, 2, 0, 2, -2, 0, -2, 0, 0, 0]
    )

    link_array = rmg.add_ones('field', at = 'link')
    link_array[5] = 3
    link_array[6] = -3

    assert_array_equal(
        grid.sum_at_nodes(link_array),
        [-2, -1, 0, -3, 6, -3, 0, 1, 2]
    )

    assert_array_equal(
        grid.calc_flux_div_at_node(link_array),
        [0, 0, 0, 0, 6, 0, 0, 0, 0]
    )

    assert_array_equal(
        grid.calc_flux_div_at_node(link_array, dirichlet_boundary = 1),
        [1, 1, 1, 1, 6, 1, 1, 1, 1]
    )

    bvs = np.full(9, 1)
    assert_array_equal(
        grid.calc_flux_div_at_node(link_array, dirichlet_boundary = bvs),
        [1, 1, 1, 1, 6, 1, 1, 1, 1]
    )

def test_jit_compile():
    rmg = RasterModelGrid((3, 3))

    @jax.jit
    def make_grid():
        return StaticGrid.from_grid(rmg)

    grid = make_grid()
    
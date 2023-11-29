import pytest
import numpy as np
from numpy.testing import assert_almost_equal
import jax.numpy as jnp
import equinox as eqx

from .fixtures import grid, state
from components import FrozenFringe

def test_frozen_fringe(grid, state):
    state = eqx.tree_at(
        lambda tree: tree.till_thickness,
        state,
        jnp.full(grid.number_of_nodes, 10.)
    )
    model = FrozenFringe(grid, state)

    assert_almost_equal(
        model.entry_pressure,
        68e3
    )

    assert_almost_equal(
        model.base_temperature,
        272.939,
        3
    )

    assert_almost_equal(
        model.bulk_conductivity,
        4.562,
        3
    )

    assert_almost_equal(
        np.mean(model.thermal_gradient),
        -0.0445,
        4
    )

    assert_almost_equal(
        np.mean(model.supercooling),
        1.0007,
        4
    )

    assert_almost_equal(
        np.mean(model.saturation),
        0.00095,
        5
    )
    
    assert_almost_equal(
        np.mean(model.nominal_heave_rate),
        1.35e-9,
        11
    )

    assert_almost_equal(
        np.mean(model.flow_resistivity),
        0.0573,
        4
    )

    assert_almost_equal(
        np.mean(model.heave_rate),
        -2.573e-7,
        10
    )

    assert_almost_equal(
        np.mean(model.fringe_growth_rate),
        6.83e-4,
        6
    )

    for i in range(10):
        model = model.update(100.0)

    assert_almost_equal(
        np.mean(model.state.fringe_thickness),
        0.074,
        4
    )

    assert_almost_equal(
        np.mean(model.state.till_thickness),
        10 - 0.074,
        3
    )
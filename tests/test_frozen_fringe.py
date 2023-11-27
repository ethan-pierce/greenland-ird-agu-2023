import pytest
import numpy as np
from numpy.testing import assert_almost_equal

from .fixtures import grid, state
from components import FrozenFringe

def test_frozen_fringe(grid, state):
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

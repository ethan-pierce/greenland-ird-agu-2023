import pytest
import numpy as np
from numpy.testing import assert_almost_equal

from components import GlacialEroder
from .fixtures import grid, state

def test_glacial_eroder(grid, state):
    model = GlacialEroder(grid, state)

    for i in range(10):
        model = model.update(1.0)

    assert_almost_equal(
        np.mean(model.state.till_thickness),
        1.49e-3,
        5
    )
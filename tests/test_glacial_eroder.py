import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

from components import GlacialEroder
from .fixtures import grid, state

def test_glacial_eroder(grid, state):
    model = GlacialEroder(
        grid,
        state,
        state.sliding_velocity,
        state.bedrock_slope
    )

    for i in range(10):
        model = model.update(1.0)

    assert_array_almost_equal(
        model.state.till_thickness,
        []
    )
import pytest
import numpy as np
from numpy.testing import assert_approx_equal

from .fixtures import grid, state

def test_model_state(grid, state):
    assert True

def test_calc_shear_stress(grid, state):
    assert_approx_equal(np.mean(state.shear_stress), 269038.20, 2)
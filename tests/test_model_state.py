import pytest
import numpy as np

from .fixtures import grid, state

def test_model_state(grid, state):
    assert True

def test_calc_shear_stress(grid, state):
    assert np.mean(state.shear_stress) == 269038.15625 
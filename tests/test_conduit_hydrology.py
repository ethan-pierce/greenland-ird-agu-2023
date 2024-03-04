"""Test the ConduitHydrology implementation."""

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import jax
import jax.numpy as jnp
import pytest

from landlab import TriangleModelGrid
from utils import StaticGrid, freeze_grid
from components import ConduitHydrology

@pytest.fixture
def grid():
    """Create a simple unstructured grid."""
    g = TriangleModelGrid(
        (
            [-11, 10, 10, -10],
            [10, 10, -10, -10]
        ),
        triangle_opts = 'pqDevjza0.25q27',
        sort = True
    )

    return g

@pytest.fixture
def conduits(grid):
    """Create an instance of the ConduitHydrology model."""
    return ConduitHydrology(
        freeze_grid(grid)
    )

def test_init(conduits):
    """Test the initialization of the ConduitHydrology model."""
    assert isinstance(conduits, ConduitHydrology)
    assert isinstance(conduits.grid, StaticGrid)
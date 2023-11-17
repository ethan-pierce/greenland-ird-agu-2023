import pytest
import numpy as np

from landlab import RasterModelGrid
from utils import StaticGrid
from components import ModelState

def test_model_state():
    rmg = RasterModelGrid((3, 3))
    grid = StaticGrid.from_grid(rmg)

    # TODO
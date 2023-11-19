import pytest
import numpy as np

from landlab import RasterModelGrid
from utils import StaticGrid
from components import ModelState, GlacialEroder

def test_glacial_eroder():
    rmg = RasterModelGrid((3, 3))
    grid = StaticGrid.from_grid(rmg)

    # TODO
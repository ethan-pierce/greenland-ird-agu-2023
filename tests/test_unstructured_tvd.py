""'Unit tests for utils/unstructured_tvd.py'""

import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
from landlab import RasterModelGrid, TriangleMeshGrid

from utils import StaticGrid, TVDAdvector

from landlab import imshow_grid
import matplotlib.pyplot as plt

def test_tvd_algorithm():
    rmg = RasterModelGrid((3, 20))

    # Looks weird but necessary to test gradient ratio
    field = rmg.add_zeros('field', at = 'node')
    field[24:26] = 2.0
    field[rmg.boundary_nodes] = 0.0

    imshow_grid(rmg, field)
    plt.show()

    velocity = rmg.add_zeros('velocity', at = 'link')
    velocity[rmg.horizontal_links] = 0.2

    tvd = TVDAdvector(rmg, 'velocity')

    colors = plt.cm.jet(np.linspace(0, 1, 50))
    for i in range(50):
        field = tvd.update(field, dt = 0.25)
        field[rmg.boundary_nodes] = 0.0

        if i % 5 == 0:
            plt.plot(field[rmg.node_y == 1], color = colors[i])

    plt.show()
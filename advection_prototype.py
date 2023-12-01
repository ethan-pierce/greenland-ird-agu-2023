import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pickle

from utils import StaticGrid

with open('examples/ird/meshes/rolige-brae.grid', 'rb') as g:
    tmg = pickle.load(g)

grid = StaticGrid.from_grid(tmg)

print(grid.number_of_nodes)
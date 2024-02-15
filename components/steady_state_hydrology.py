"""Solves for the steady state subglacial hydrology, using Schoof's (2010) model."""

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import jaxopt

from utils import StaticGrid

class SteadyStateHydrology(eqx.Module):
    """
    Class for solving the steady state subglacial hydrology using Schoof's (2010) model.
    
    Attributes:
        TODO: Add attributes here
        
    Methods:
        TODO: Add methods here
    """
    
    grid: StaticGrid
    melt_rate: jax.Array = eqx.field(converter = jnp.asarray)

    adjacency_matrix: jax.Array = eqx.field(converter = jnp.asarray, init = False)

    def __post_init__(self):
        """Initialize the component."""
        self._build_adjacency_matrix()
        
    def _build_adjacency_matrix(self):
        """Build the adjacency matrix for the grid."""
        adjacency = np.zeros((self.grid.number_of_nodes, self.grid.number_of_nodes))

        adjacency[
            np.arange(self.grid.number_of_nodes)[:, np.newaxis], 
            self.grid.adjacent_nodes_at_node
        ] = np.where(self.grid.adjacent_nodes_at_node != -1, 1, 0)

        self.adjacency_matrix = jnp.asarray(adjacency)

    def _route_discharge(self):
        """Route meltwater input at nodes through grid links."""
        solution = jaxopt.linear_solve.solve_cg(
            lambda x: jnp.dot(self.adjacency_matrix, x),
            self.melt_rate
        )
        return solution

    

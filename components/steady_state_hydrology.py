"""Solves for the steady state subglacial hydrology, using Schoof's (2010) model."""

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import lineax as lx

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
    operator: lx.MatrixLinearOperator = eqx.field(init = False)

    def __post_init__(self):
        """Initialize the component."""
        self._build_adjacency_matrix()
        self.operator = lx.MatrixLinearOperator(self.adjacency_matrix)
        
    def _build_adjacency_matrix(self):
        """Build the adjacency matrix for the grid."""
        adjacency = np.zeros((self.grid.number_of_nodes, self.grid.number_of_nodes))

        for i in range(self.grid.number_of_nodes):
            for j in self.grid.adjacent_nodes_at_node[i]:
                if j != -1:
                    adjacency[i, j] = 1
                    adjacency[j, i] = 1

        self.adjacency_matrix = jnp.asarray(adjacency)

    def _route_discharge(self):
        """Route meltwater input at nodes through grid links."""
        solution = lx.linear_solve(
            operator = self.operator, 
            vector = self.melt_rate,
            solver = lx.AutoLinearSolver(well_posed = True),
            throw = True
        )
        return solution.value

    

    
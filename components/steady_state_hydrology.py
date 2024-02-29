"""Solves for the steady state subglacial hydrology, using Schoof's (2010) model."""

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import lineax as lx
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
    overburden: jax.Array = eqx.field(converter = jnp.asarray)

    # adjacency_matrix: jax.Array = eqx.field(converter = jnp.asarray, init = False)
    # operator: lx.MatrixLinearOperator = eqx.field(init = False)

    def __post_init__(self):
        """Initialize the component."""
        pass
        # self._build_adjacency_matrix()
        # self.operator = lx.MatrixLinearOperator(self.adjacency_matrix)
        
    def _build_adjacency_matrix(self):
        """Build the adjacency matrix for the grid."""
        adjacency = np.zeros((self.grid.number_of_nodes, self.grid.number_of_nodes))

        for i in range(self.grid.number_of_nodes):
            adj_nodes = self.grid.adjacent_nodes_at_node    [i]
            valid_adj_nodes = adj_nodes[adj_nodes != -1]

            links_i = self.grid.links_at_node[i]
            links_j = self.grid.links_at_node[valid_adj_nodes]
            common_links = np.intersect1d(links_i, links_j)
            valid_links = common_links[common_links != -1]

            lengths = self.grid.length_of_link[valid_links]

            adjacency[i, valid_adj_nodes] = 1 / lengths
            adjacency[valid_adj_nodes, i] = 1 / lengths
                    
        self.adjacency_matrix = jnp.asarray(adjacency)

    def _discharge_residual(self, discharge: jax.Array):
        """Return the excess discharge at grid nodes."""
        discharge = jnp.where(self.grid.status_at_link == 4, 0.0, discharge)

        direction = jnp.where(
            self.overburden[self.grid.node_at_link_head] 
            > self.overburden[self.grid.node_at_link_tail],
            1,
            -1
        )

        net_flux = jnp.sum(
            (direction * discharge)[self.grid.links_at_node],
            axis = 1
        )

        return net_flux - self.melt_rate 

    def _route_discharge(self):
        """Route meltwater input at nodes through grid links."""
        solver = jaxopt.GaussNewton(
            residual_fun = self._discharge_residual,
            verbose = True,
            tol = 1e-10
        )
    
        return solver.run(jnp.zeros(self.grid.number_of_links)).params

    

    
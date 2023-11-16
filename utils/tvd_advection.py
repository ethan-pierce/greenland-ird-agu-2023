"""Total-Variation-Diminishing (TVD) Advection algorithm in Landlab."""

import jax
import jax.numpy as jnp
import equinox as eqx
from utils import StaticGrid

class TVDAdvection(eqx.Module):
    """Total-Variation-Diminishing Advection algorithm."""

    grid: StaticGrid
    velocity: jax.Array = eqx.field(converter = jnp.asarray)
    field: jax.Array = eqx.field(converter = jnp.asarray)

    upwind_links: jax.Array = eqx.field(converter = jnp.asarray, init = False)
    gradient_ratio: jax.Array = eqx.field(converter = jnp.asarray, init = False)

    def __post_init__(self):

        # Populate upwind_links
        n_parallel =  len(self.grid.parallel_links_at_link)
        inverse_velocity = jnp.asarray(self.velocity <= 0).astype(int)
        self.upwind_links = self.grid.parallel_links_at_link[jnp.arange(n_parallel), inverse_velocity]

        # Define upwind to local gradient ratio
        local_diff = (
            self.velocity[self.grid.node_at_link_head] - self.velocity[self.grid.node_at_link_tail]
        )
        self.gradient_ratio = jnp.where(
            (local_diff != 0) & (self.upwind_links != -1),
            jnp.divide(local_diff[self.upwind_links], local_diff),
            1.0
        )
        
    
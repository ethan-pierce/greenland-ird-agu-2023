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
    flux_limiter: jax.Array = eqx.field(converter = jnp.asarray, init = False)

    def __post_init__(self):
        self.upwind_links = self.define_upwind_links()
        self.gradient_ratio = self.calc_gradient_ratio()
        self.flux_limiter = self.flux_lim_vanleer(self.gradient_ratio)     

    def define_upwind_links(self) -> jax.Array:
        """Identify links upwind of each link."""
        n_parallel =  len(self.grid.parallel_links_at_link)
        inverse_velocity = jnp.asarray(self.velocity <= 0).astype(int)
        return self.grid.parallel_links_at_link[jnp.arange(n_parallel), inverse_velocity]
        
    def calc_gradient_ratio(self) -> jax.Array:
        """Calculate the ratio between local and upwind gradients."""
        local_diff = (
            self.field[self.grid.node_at_link_head] - self.field[self.grid.node_at_link_tail]
        )
        return jnp.where(
            (local_diff != 0) & (self.upwind_links != -1),
            jnp.divide(local_diff[self.upwind_links], local_diff),
            1.0
        )

    def flux_lim_vanleer(self, ratio: jax.Array) -> jax.Array:
        """Van Leer flux limiter."""
        return (ratio + jnp.abs(ratio)) / (1.0 + jnp.abs(ratio))

    def map_to_links_linear_upwind(self) -> jax.Array:
        """Assign values to links from upstream nodes."""
        positive_velocity = self.velocity > 0
        return jnp.where(
            self.velocity > 0,
            self.field[self.grid.node_at_link_tail],
            self.field[self.grid.node_at_link_head]
        )

    def map_to_links_lax_wendroff(self, courant: float) -> jax.Array:
        """Assign values to links using a weighted combination of nodes."""
        return 0.5 * (
            (1 + courant) * self.field[self.grid.node_at_link_tail]
            + (1 - courant) * self.field[self.grid.node_at_link_head]
        )

    def calc_rate_of_change(self, dt: float) -> jax.Array:
        """Calculate the rate of change at nodes."""
        courant = dt * self.velocity / self.grid.length_of_link
        field_at_links_high = self.map_to_links_lax_wendroff(courant)
        field_at_links_low = self.map_to_links_linear_upwind()
        
        field_at_links = (
            self.flux_limiter * field_at_links_high
            + (1 - self.flux_limiter) * field_at_links_low
        )

        flux = self.velocity * field_at_links

        return -self.grid.calc_flux_div_at_node(flux)

    def update(self, dt: float):
        """Run one step of advection and return a new TVDAdvection object."""
        rate_of_change = self.calc_rate_of_change(dt)
        new_field = self.field + rate_of_change * dt

        return TVDAdvection(
            self.grid,
            self.velocity,
            new_field
        )
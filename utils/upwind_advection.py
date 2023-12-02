"""Upwind advection algorithm for unstructured grids, written in JAX."""

import jax
import jax.numpy as jnp
import equinox as eqx
from utils import StaticGrid

class UpwindAdvection(eqx.Module):
    """Upwind advection algorithm."""

    grid: StaticGrid

    field: jax.Array = eqx.field(converter = jnp.asarray)
    control: jax.Array = eqx.field(converter = jnp.asarray)
    velocity: jax.Array = eqx.field(converter = jnp.asarray)

    def calc_rate_of_change(self) -> jax.Array:
        """Calculate the rate of change of the field due to advection."""
        upwind_field = self.grid.map_value_at_max_node_to_link(
            self.control,
            self.field
        )
        flux = upwind_field * self.velocity

        return self.grid.calc_flux_div_at_node(flux)

    def update(self, dt: float):
        """Run one advection step and return the updated field."""
        rate_of_change = self.calc_rate_of_change()
        new_field = self.field + rate_of_change * dt

        return new_field
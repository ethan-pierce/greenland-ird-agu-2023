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
        return -self.grid.calc_flux_div_at_node(flux)
    
    def calc_div_grad_form(self) -> jax.Array:
        """Calculate the div form: u dot grad(h) + h div(u)."""
        div = self.grid.calc_flux_div_at_node(self.velocity * self.grid.length_of_face[self.grid.face_at_link])
        deformation = self.field * div

        grad = self.velocity * self.grid.calc_grad_at_link(self.field)
        advection = self.grid.sum_at_nodes(grad)

        return advection + deformation

    def update(self, dt: float):
        """Run one advection step and return the updated field."""
        rate_of_change = self.calc_rate_of_change()
        new_field = self.field + rate_of_change * dt

        return new_field
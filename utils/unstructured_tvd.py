"""Total-variation diminishing (TVD) scheme for an unstructured grid."""

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from landlab import TriangleMeshGrid
from utils import StaticGrid

class TVDAdvector(eqx.Module):
    """JAX- and Equinox-backed implementation of first-order TVD advection in Landlab."""

    grid: StaticGrid
    velocity: jax.Array = eqx.field(converter = jnp.asarray)
    tracer: jax.Array = eqx.field(converter = jnp.asarray)

    def initialize(self):
        """Initialize the TVDAdvector."""
        pass

    def update(self, field, dt: float):
        """Advect the tracer over dt seconds and return the resulting field."""
        if dt > self._calc_stable_dt():
            raise ValueError(f"Maximum stable timestep is {self._calc_stable_dt()} seconds.")

        div = self.grid.calc_flux_div_at_node(self.calc_flux(field))
        return field - div * dt

    def _identify_upwind_points(self):
        """Identify the upwind points for each link."""
        pass

    def _interp_gradient(self, field):
        """Interpolate the component-wise gradient of a scalar field at nodes."""
        mag, comps = self.grid.calc_slope_at_node(
            elevs = field,
            ignore_closed_nodes = False,
            return_components = True
        )

        return jnp.asarray(comps).T

    def _van_leer(self, r):
        """Van Leer limiter function."""
        return (r + jnp.abs(r)) / (1 + r)

    def _superbee(self, r):
        """Superbee limiter function."""
        return jnp.max(jnp.asarray([0, jnp.minimum(2 * r, 1), jnp.minimum(r, 2)]))

    def _calc_stable_dt(self, cfl: float = 0.2):
        """Calculate the stable timestep for advection."""
        return cfl * jnp.min(self.grid.length_of_link) / (2 * jnp.max(jnp.abs(self.velocity)))

    def calc_flux_limited(self, field):
        """Calculate the flux-limited field at links."""
        gradient = self._interp_gradient(field)

        limited_field = np.zeros(self.grid.number_of_links)
        for link in jnp.arange(self.grid.number_of_links):
            if self.velocity[link] >= 0:
                c = self.grid.node_at_link_tail[link]
                d = self.grid.node_at_link_head[link]
            else:
                c = self.grid.node_at_link_head[link]
                d = self.grid.node_at_link_tail[link]

            if field[d] == field[c]:
                limited_field[link] = field[c]
                continue

            grad_c = gradient[c]

            vec = jnp.asarray(
                [self.grid.node_x[d] - self.grid.node_x[c], 
                self.grid.node_y[d] - self.grid.node_y[c]]
            )

            slope_factor = (
                jnp.dot(2 * grad_c, vec) / 
                (field[d] - field[c])
            )
        
            flux_limiter = self._superbee(slope_factor)

            limited_field[link] = (
                field[c] 
                + (1 / 2) * flux_limiter 
                * (field[d] - field[c])
            )

        return jnp.asarray(limited_field)
        
    def calc_flux(self, field):
        """Calculate the flux of a field at links."""
        return self.velocity * self.calc_flux_limited(field)

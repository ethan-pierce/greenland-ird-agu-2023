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

    upwind_ghost_at_link: jax.Array = eqx.field(init = False, converter = jnp.asarray)
    upwind_real_at_link: jax.Array = eqx.field(init = False, converter = jnp.asarray)

    def __post_init__(self):
        """Initialize the TVDAdvector."""
        self.upwind_ghost_at_link = self._identify_upwind_ghosts()
        self.upwind_real_at_link = self._get_nearest_upwind_real()

    def update(self, field, dt: float):
        """Advect the tracer over dt seconds and return the resulting field."""
        if dt > self._calc_stable_dt():
            raise ValueError(f"Maximum stable timestep is {self._calc_stable_dt()} seconds.")

        div = self.grid.calc_flux_div_at_node(self.calc_flux(field))
        return field - div * dt

    def _identify_upwind_ghosts(self):
        """Establish the upwind ghost nodes for each link."""
        head_x = self.grid.node_x[self.grid.node_at_link_head]
        head_y = self.grid.node_y[self.grid.node_at_link_head]
        tail_x = self.grid.node_x[self.grid.node_at_link_tail]
        tail_y = self.grid.node_y[self.grid.node_at_link_tail]

        ghost_if_head_upwind = jnp.asarray([
            tail_x - (head_x - tail_x),
            tail_y - (head_y - tail_y)
        ])

        ghost_if_tail_upwind = jnp.asarray([
            head_x - (tail_x - head_x),
            head_y - (tail_y - head_y)
        ])

        # Convention: consider zero velocity to be positive
        return jnp.where(
            self.velocity >= 0,
            ghost_if_tail_upwind,
            ghost_if_head_upwind
        ).T

    def _get_nearest_upwind_real(self):
        """Identify the upwind cells for each link."""
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

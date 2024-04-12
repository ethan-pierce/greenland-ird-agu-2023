"""Total-variation diminishing (TVD) scheme for an unstructured grid."""

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from scipy.spatial import KDTree
from landlab import TriangleModelGrid
from utils import StaticGrid

def identify_upwind_ghosts(grid, velocity):
    head_x = grid.node_x[grid.node_at_link_head]
    head_y = grid.node_y[grid.node_at_link_head]
    tail_x = grid.node_x[grid.node_at_link_tail]
    tail_y = grid.node_y[grid.node_at_link_tail]

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
        velocity >= 0,
        ghost_if_tail_upwind,
        ghost_if_head_upwind
    ).T

def get_nearest_upwind_real(grid, ghosts):
    """Identify the nearest upwind real nodes for each link."""
    points = jnp.asarray(
        [grid.node_x, grid.node_y]
    ).T

    tree = KDTree(points)
    _, indices = tree.query(ghosts)

    return indices, points[indices]

class TVDAdvector(eqx.Module):
    """JAX- and Equinox-backed implementation of first-order TVD advection in Landlab."""

    grid: StaticGrid
    velocity: jax.Array = eqx.field(converter = jnp.asarray)
    tracer: jax.Array = eqx.field(converter = jnp.asarray)

    # These should be calculated once, outside of jax.jit, and passed as static args
    upwind_ghost_at_link: jax.Array = eqx.field(converter = jnp.asarray)
    upwind_real_idx: jax.Array = eqx.field(converter = jnp.asarray)
    upwind_real_coords: jax.Array = eqx.field(converter = jnp.asarray)

    upwind_shift_vector: jax.Array = eqx.field(init = False, converter = jnp.asarray)
    upwind_values: jax.Array = eqx.field(init = False, converter = jnp.asarray)

    def __post_init__(self):
        """Initialize the TVDAdvector."""
        self.upwind_shift_vector = self._calc_upwind_shift_vector()
        self.upwind_values = self._interp_upwind_values()

    @jax.jit
    def update(self, dt: float):
        """Advect the tracer over dt seconds and return the resulting field."""
        face_flux_at_links = self.calc_flux()
        sum_at_nodes = jnp.sum(face_flux_at_links[self.grid.links_at_node], axis = 1)

        div = jnp.where(
            self.grid.cell_area_at_node != 0,
            sum_at_nodes / self.grid.cell_area_at_node,
            0.0
        )

        return eqx.tree_at(
            lambda tree: tree.tracer,
            self,
            self.tracer + dt * div
        )

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
        """Identify the nearest upwind real nodes for each link."""
        points = jnp.asarray(
            [self.grid.node_x, self.grid.node_y]
        ).T

        tree = KDTree(points)
        _, indices = tree.query(self.upwind_ghost_at_link)

        return indices, points[indices]

    def _calc_upwind_shift_vector(self):
        """Calculate the shift vector between the upwind ghosts and reals."""
        return (
            self.upwind_real_coords - self.upwind_ghost_at_link
        )

    def _interp_gradient(self):
        """Calculate the magnitude and (x, y) components of the gradient of a field at nodes."""
        return self.grid.calc_gradient_vector_at_node(self.tracer)

    def _interp_upwind_values(self):
        """Interpolate the values of a field at the upwind ghost nodes."""
        mag, comps = self._interp_gradient()

        return (
            self.tracer[self.upwind_real_idx]
            + jnp.sum(self.upwind_shift_vector * comps[self.upwind_real_idx], axis = 1)
        )

    def _van_leer(self, r):
        """Van Leer limiter function."""
        return (r + jnp.abs(r)) / (1 + jnp.abs(r))

    def _superbee(self, r):
        """Superbee limiter function."""
        return jnp.maximum(
            jnp.zeros_like(r), 
            jnp.maximum(jnp.minimum(2 * r, 1), jnp.minimum(r, 2))
        )

    def _calc_face_flux(self):
        """Calculate the flux-limited field at links."""
        center = jnp.where(
            self.velocity >= 0,
            self.tracer[self.grid.node_at_link_tail],
            self.tracer[self.grid.node_at_link_head]
        )
        downwind = jnp.where(
            self.velocity >= 0,
            self.tracer[self.grid.node_at_link_head],
            self.tracer[self.grid.node_at_link_tail]
        )
        upwind = self.upwind_values

        r_factor = jnp.where(
            downwind != center,
            (center - upwind) / (downwind - center),
            0.0
        )

        return center + 0.5 * self._van_leer(r_factor) * (downwind - center)

    def calc_flux(self):
        """Calculate the flux at links."""
        return self.velocity * self._calc_face_flux()

    # BUG: This will severely underestimate the stable timestep
    def calc_stable_dt(self, cfl: float = 0.2):
        """Calculate the stable timestep for advection."""
        return cfl * jnp.min(self.grid.length_of_link) / (2 * jnp.max(jnp.abs(self.velocity)))

        
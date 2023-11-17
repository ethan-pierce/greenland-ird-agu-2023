"""Immutable Landlab grid to support automatic differentiation."""

import dataclasses
import jax
import jax.numpy as jnp
import equinox as eqx

class StaticGrid(eqx.Module):
    """Handles grid elements, connectivity, and computes gradients."""

    # Basic information
    shape: tuple
    number_of_nodes: int
    number_of_corners: int
    number_of_links: int
    number_of_faces: int
    number_of_patches: int
    number_of_cells: int

    # Geometry
    node_x: jax.Array = eqx.field(converter = jnp.asarray)
    node_y: jax.Array = eqx.field(converter = jnp.asarray)
    x_of_corner: jax.Array = eqx.field(converter = jnp.asarray)
    y_of_corner: jax.Array = eqx.field(converter = jnp.asarray)
    length_of_link: jax.Array = eqx.field(converter = jnp.asarray)
    length_of_face: jax.Array = eqx.field(converter = jnp.asarray)
    area_of_patch: jax.Array = eqx.field(converter = jnp.asarray)
    cell_area_at_node: jax.Array = eqx.field(converter = jnp.asarray)

    # Boundary conditions
    core_nodes: jax.Array = eqx.field(converter = jnp.asarray)
    boundary_nodes: jax.Array = eqx.field(converter = jnp.asarray)
    status_at_node: jax.Array = eqx.field(converter = jnp.asarray)

    # Connectivity
    node_at_link_head: jax.Array = eqx.field(converter = jnp.asarray)
    node_at_link_tail: jax.Array = eqx.field(converter = jnp.asarray)
    links_at_node: jax.Array = eqx.field(converter = jnp.asarray)
    link_dirs_at_node: jax.Array = eqx.field(converter = jnp.asarray)
    parallel_links_at_link: jax.Array = eqx.field(converter = jnp.asarray)
    face_at_link: jax.Array = eqx.field(converter = jnp.asarray)
    cell_at_node: jax.Array = eqx.field(converter = jnp.asarray)

    @classmethod
    def from_grid(cls, grid):
        """Instantiate a StaticGrid from an existing Landlab grid."""
        fields = {
            field.name: getattr(grid, field.name) for field in dataclasses.fields(StaticGrid)
        }
        return cls(**fields)

    def map_mean_of_links_to_node(self, array):
        """Map an array of values from links to nodes."""
        return jnp.mean(array[self.links_at_node], axis = 1)

    def map_mean_of_link_nodes_to_link(self, array):
        """Map an array of values from nodes to links."""
        return 0.5 * (array[self.node_at_link_head] + array[self.node_at_link_tail])

    def sum_at_nodes(self, array):
        """At each node, sum incoming and outgoing values of an array defined on links."""
        return jnp.sum(self.link_dirs_at_node * array[self.links_at_node], axis = 1)
    
    def calc_grad_at_link(self, array):
        """At each link, calculate the gradient of an array defined on nodes."""
        return jnp.divide(
            array[self.node_at_link_head] - array[self.node_at_link_tail],
            self.length_of_link
        )

    def calc_flux_div_at_node(self, array, dirichlet_boundary = 0.0):
        """At each node, calculate the divergence of an array of fluxes defined on links."""
        return jnp.where(
            self.status_at_node == 0,
            jnp.divide(self.sum_at_nodes(array), self.cell_area_at_node),
            dirichlet_boundary
        )

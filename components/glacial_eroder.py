"""This component handles subglacial sediment production."""

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from utils import StaticGrid
from components import ModelState

class GlacialEroder(eqx.Module):
    """Erode bedrock beneath an ice mass by abrasion and quarrying."""

    grid: StaticGrid
    state: ModelState

    sliding_velocity: jax.Array = eqx.field(converter = jnp.asarray, init = False)
    bedrock_slope: jax.Array = eqx.field(converter = jnp.asarray, init = False)
    abrasion_coefficient: float = 4e-7 # m / a
    quarrying_coefficient: float = 2e-6 # m / a

    def __post_init__(self):
        self.sliding_velocity = self.grid.map_mean_of_links_to_node(self.state.sliding_velocity)
        self.bedrock_slope = self.grid.calc_slope_at_node(self.state.bedrock_elevation)

    def calc_abrasion_rate(self):
        """Calculate the rate of erosion from abrasion."""
        return self.abrasion_coefficient * self.sliding_velocity**2

    def calc_quarrying_rate(self):
        """Calculate the rate of erosion from quarrying."""
        return (
            self.quarrying_coefficient 
            * self.state.effective_pressure**3
            * self.sliding_velocity
            * self.bedrock_slope**2
        )

    def update(self, dt: float):
        """Update the model state to reflect erosion over one time step of dt years."""
        abrasion = self.calc_abrasion_rate()
        quarrying = self.calc_quarrying_rate()

        total_erosion = (abrasion + quarrying) * dt

        updated_state = eqx.tree_at(
            lambda tree: tree.till_thickness, 
            self.state,
            self.state.till_thickness + total_erosion
        )

        return GlacialEroder(self.grid, updated_state)
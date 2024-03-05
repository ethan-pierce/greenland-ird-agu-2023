"""Model subglacial hydrology through a series of conduits."""

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import jaxopt

from landlab.components import FlowAccumulator

from utils import StaticGrid
from components import ModelState

class ConduitHydrology(eqx.Module):
    """Implement a model of subglacial hydrology through a series of conduits."""

    state: ModelState
    grid: StaticGrid = eqx.field(init = False)
    
    conduit_area: jnp.array = eqx.field(converter = jnp.asarray)
    melt_flux: jnp.array = eqx.field(converter = jnp.asarray, init = False)
    potential: jnp.array = eqx.field(converter = jnp.asarray, init = False)
    geometric_gradient: jnp.array = eqx.field(converter = jnp.asarray, init = False)

    flow_coeff: float = 4.05e-2
    flow_exp: float = 5 / 4 

    def __post_init__(self):
        """Initialize remaining model fields."""
        self.grid = self.state.grid

        self.melt_flux = self.state.melt_rate * self.grid.cell_area_at_node / self.state.sec_per_a

        self.potential = (
            self.state.water_density * self.state.gravity * self.state.bedrock_elevation
            + self.state.water_pressure
        )

        self.geometric_gradient = (
            -self.state.ice_density * self.state.gravity * self.state.surface_slope
            -(self.state.water_density - self.state.ice_density) * self.state.gravity 
            * self.state.bedrock_slope
        )

    def route_flow(self, landlab_grid) -> jnp.array:
        """Route discharge based on the hydraulic potential field."""
        fa = FlowAccumulator(
            landlab_grid, 
            surface = self.potential,
            runoff_rate = self.melt_flux,
            flow_director = 'FlowDirectorMFD'
        )

        area, discharge = fa.accumulate_flow(update_depression_finder = False)

        return discharge

    
    
    





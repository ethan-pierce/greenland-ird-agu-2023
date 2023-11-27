"""Models sediment entrainment within basal ice layers."""

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from utils import StaticGrid
from components import ModelState

class FrozenFringe(eqx.Module):
    """Entrain sediment in frozen fringe and dispersed basal ice layers."""

    grid: StaticGrid
    state: ModelState

    # Physical constants
    surface_energy: float = 0.034
    pore_throat_radius: float = 1e-6
    melt_temperature: float = 273
    ice_conductivity: float = 2
    sediment_conductivity: float = 6.27
    porosity: float = 0.4
    entry_pressure: float = eqx.field(init = False)
    base_temperature: float = eqx.field(init = False)
    bulk_conductivity: float = eqx.field(init = False)
    thermal_gradient: float = eqx.field(init = False)

    # Model fields
    supercooling: jnp.array = eqx.field(converter = jnp.asarray, init = False)

    def __post_init__(self):
        self.entry_pressure = 2 * self.surface_energy / self.pore_throat_radius
        self.base_temperature = (
            self.melt_temperature
            -(
                (self.entry_pressure * self.melt_temperature)
                / (self.state.ice_density * self.state.ice_latent_heat)
            )
        )
        self.bulk_conductivity = (
            (1 - self.porosity) * self.sediment_conductivity
            + self.porosity * self.ice_conductivity
        )
        self.thermal_gradient = (
            -(self.state.geothermal_heat_flux + self.state.frictional_heat_flux / self.state.sec_per_a)
            / self.bulk_conductivity
        )
        self.supercooling = 1 - (
            (self.thermal_gradient * self.state.fringe_thickness)
            / (self.melt_temperature - self.base_temperature)
        )

    def update(self, dt: float):
        """Advance the model by one step of dt years."""
        return FrozenFringe(self.grid, self.state)
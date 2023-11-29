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
    permeability: float = 4.1e-17
    alpha: float = 3.1
    beta: float = 1.3
    grain_size: float = 4e-5
    film_thickness: float = 1e-8
    entry_pressure: float = eqx.field(init = False)
    base_temperature: float = eqx.field(init = False)
    bulk_conductivity: float = eqx.field(init = False)
    thermal_gradient: float = eqx.field(init = False)

    # Model fields
    supercooling: jnp.array = eqx.field(converter = jnp.asarray, init = False)
    saturation: jnp.array = eqx.field(converter = jnp.asarray, init = False)
    nominal_heave_rate: jnp.array = eqx.field(converter = jnp.asarray, init = False)
    flow_resistivity: jnp.array = eqx.field(converter = jnp.asarray, init = False)
    heave_rate: jnp.array = eqx.field(converter = jnp.asarray, init = False)
    fringe_growth_rate: jnp.array = eqx.field(converter = jnp.asarray, init = False)

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
            -(self.state.geothermal_heat_flux + self.state.frictional_heat_flux)
            / self.bulk_conductivity
            / self.state.sec_per_a # Because Qg and Qf are in terms of years, not seconds
        )
        self.supercooling = 1 - (
            (self.thermal_gradient * self.state.fringe_thickness)
            / (self.melt_temperature - self.base_temperature)
        )
        self.saturation = 1 - self.supercooling**(-self.beta)
        self.nominal_heave_rate = -(
            (self.state.water_density**2 * self.state.ice_latent_heat * self.thermal_gradient * self.permeability)
            / (self.state.ice_density * self.melt_temperature * self.state.water_viscosity)
        )
        self.flow_resistivity = -(
            (self.state.water_density**2 * self.permeability * self.thermal_gradient * self.grain_size**2)
            / (self.state.ice_density**2 * (self.melt_temperature - self.base_temperature) * self.film_thickness**3)
        )
        self.heave_rate = self.calc_heave_rate()
        self.fringe_growth_rate = self.calc_fringe_growth_rate()

    def calc_heave_rate(self):
        """Calculate the rate of vertical heave acting on the frozen fringe."""
        A = (
            self.supercooling 
            + self.porosity * (1 - self.supercooling 
            + ((1 / (1 - self.beta)) * (self.supercooling**(1 - self.beta) - 1)))
        )

        B = (
            ((1 - self.porosity)**2 / (self.alpha + 1)) 
            * (self.supercooling**(self.alpha + 1) - 1)
        )

        C = (
            ((2 * (1 - self.porosity) * self.porosity) / (self.alpha - self.beta + 1)) 
            * (self.supercooling**(self.alpha - self.beta + 1) - 1)
        )

        D = (
            (self.porosity**2 / (self.alpha - 2 * self.beta + 1)) 
            * (self.supercooling**(self.alpha - 2 * self.beta + 1) - 1)
        )

        heave_rate = (
            self.nominal_heave_rate * (A - self.state.effective_pressure / self.entry_pressure)
            / (B + C + D + self.flow_resistivity)
        )

        return heave_rate

    def calc_fringe_growth_rate(self):
        """Calculate the rate of vertical sediment entrainment in basal ice layers."""
        melt_rate = self.state.melt_rate / self.state.sec_per_a

        return jnp.where(
            self.saturation > 0,
            (-melt_rate - self.heave_rate) / (self.porosity * self.saturation),
            0
        )

    def update(self, dt: float):
        """Advance the model by one step of dt years."""
        real_growth_rate = jnp.minimum(
            self.fringe_growth_rate,
            self.state.till_thickness
        )

        fringe_thickness = self.state.fringe_thickness + real_growth_rate * dt
        fringe_thickness = jnp.where(
            fringe_thickness < self.state.min_fringe_thickness,
            self.state.min_fringe_thickness,
            fringe_thickness
        )

        updated_state = eqx.tree_at(
            lambda tree: tree.fringe_thickness, 
            self.state,
            fringe_thickness
        )

        till_thickness = self.state.till_thickness - real_growth_rate * dt
        till_thickness = jnp.where(
            till_thickness < 0,
            0,
            till_thickness
        )

        updated_state = eqx.tree_at(
            lambda tree: tree.till_thickness,
            updated_state,
            till_thickness
        )

        return FrozenFringe(self.grid, updated_state)
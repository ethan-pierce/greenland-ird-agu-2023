"""Models sediment entrainment within basal ice layers."""

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx

from utils import StaticGrid, TVDAdvector
from components import ModelState

class FluxIntegrator(eqx.Module):
    """Integrate sediment fluxes at the terminus of a glacier."""

    n_steps: int
    advector: TVDAdvector
    node_is_terminus: jax.Array = eqx.field(converter = jnp.asarray)
    current_step: int = 0

    fluxes: jax.Array = eqx.field(converter = jnp.asarray, init = False)

    def __post_init__(self):
        self.fluxes = jnp.zeros(self.n_steps)

    def clear_fringe(self, state: ModelState) -> ModelState:
        """Remove frozen fringe at the boundaries of the grid."""
        cleared_fringe = jnp.where(
            state.grid.status_at_node == 0,
            state.fringe_thickness,
            state.min_fringe_thickness
        )
        return eqx.tree_at(
            lambda t: t.fringe_thickness,
            state,
            cleared_fringe
        )

    def update(self, state: ModelState) -> tuple:
        """Integrate fluxes at the terminus, then clear boundary nodes."""
        updated_fluxes = self.fluxes.at[self.current_step].set(
            jnp.sum(state.fringe_thickness * self.node_is_terminus)
        )

        updated_state = self.clear_fringe(state)

        updated_integrator = eqx.tree_at(
            lambda t: (t.current_step, t.fluxes),
            self,
            (self.current_step + 1, updated_fluxes)
        )

        return (updated_state, updated_integrator)

    
class FrozenFringe(eqx.Module):
    """Entrain sediment in frozen fringe and dispersed basal ice layers."""

    state: ModelState
    grid: StaticGrid = eqx.field(init = False)

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
    critical_depth: float = 10
    cluster_volume_fraction: float = 0.64
    till_grain_radius: float = 4e-5
    entry_pressure: float = eqx.field(init = False)
    base_temperature: float = eqx.field(init = False)
    bulk_conductivity: float = eqx.field(init = False)
    thermal_gradient: float = eqx.field(init = False)
    nominal_heave_rate: jnp.array = eqx.field(converter = jnp.asarray, init = False)
    flow_resistivity: jnp.array = eqx.field(converter = jnp.asarray, init = False)
    
    def __post_init__(self):
        self.grid = self.state.grid
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
        self.nominal_heave_rate = -(
            (self.state.water_density**2 * self.state.ice_latent_heat * self.thermal_gradient * self.permeability)
            / (self.state.ice_density * self.melt_temperature * self.state.water_viscosity)
        )
        self.flow_resistivity = -(
            (self.state.water_density**2 * self.permeability * self.thermal_gradient * self.grain_size**2)
            / (self.state.ice_density**2 * (self.melt_temperature - self.base_temperature) * self.film_thickness**3)
        )

    def calc_supercooling(self, fringe_thickness: jnp.array):
        supercooling = 1 - (
            (self.thermal_gradient * self.state.fringe_thickness)
            / (self.melt_temperature - self.base_temperature)
        )
        return supercooling
    
    def calc_saturation(self, supercooling: jnp.array):
        saturation = 1 - supercooling**(-self.beta)
        return saturation

    def calc_heave_rate(self, supercooling: jnp.array):
        """Calculate the rate of vertical heave acting on the frozen fringe."""
        A = (
            supercooling 
            + self.porosity * (1 - supercooling 
            + ((1 / (1 - self.beta)) * (supercooling**(1 - self.beta) - 1)))
        )

        B = (
            ((1 - self.porosity)**2 / (self.alpha + 1)) 
            * (supercooling**(self.alpha + 1) - 1)
        )

        C = (
            ((2 * (1 - self.porosity) * self.porosity) / (self.alpha - self.beta + 1)) 
            * (supercooling**(self.alpha - self.beta + 1) - 1)
        )

        D = (
            (self.porosity**2 / (self.alpha - 2 * self.beta + 1)) 
            * (supercooling**(self.alpha - 2 * self.beta + 1) - 1)
        )

        heave_rate = (
            self.nominal_heave_rate * (A - self.state.effective_pressure / self.entry_pressure)
            / (B + C + D + self.flow_resistivity)
        )

        return heave_rate

    def calc_fringe_growth_rate(self, fringe_thickness: jnp.array):
        """Calculate the rate of vertical sediment entrainment in basal ice layers."""
        supercooling = self.calc_supercooling(fringe_thickness)
        saturation = self.calc_saturation(supercooling)
        heave_rate = self.calc_heave_rate(supercooling)
        melt_rate = self.state.melt_rate / self.state.sec_per_a

        return jnp.where(
            saturation > 0,
            (-melt_rate - heave_rate) / (self.porosity * saturation),
            0
        )

    def calc_dispersed_growth_rate(self, supercooling: jnp.array):
        """Calculate the rate of vertical sediment entrainment in dispersed basal ice layers."""
        temp_at_top_of_fringe = (
            self.melt_temperature 
            - (self.melt_temperature - self.base_temperature)
            * supercooling
        )
        gradient = (self.melt_temperature - temp_at_top_of_fringe) / self.critical_depth
        permeability = (
            (self.till_grain_radius**2 * (1 - self.cluster_volume_fraction)**3)
            / (45 * self.cluster_volume_fraction**2)
        )
        rate_coeff = (
            (permeability * self.state.ice_density * self.state.ice_latent_heat)
            / 
            (
                self.state.water_viscosity 
                * self.melt_temperature 
                * (2 * self.ice_conductivity + self.sediment_conductivity)
            )
        )
        return (
            gradient * (rate_coeff * 3 * self.ice_conductivity) 
            / (1 + rate_coeff * self.state.ice_density * self.state.ice_latent_heat)
        )

    @jax.jit
    def update(self, dt: float):
        """Advance the model by one step of dt years."""
        dt_s = dt * self.state.sec_per_a

        # solver = optx.Newton(rtol = 1e-6, atol = 1e-6)
        # max_growth = optx.root_find(
        #     lambda h, _: (
        #         h - self.state.fringe_thickness - dt * self.calc_fringe_growth_rate(h)
        #     ),
        #     solver,
        #     self.state.fringe_thickness,
        #     args = None
        # ).value

        max_growth = (
            dt_s * self.calc_fringe_growth_rate(self.state.fringe_thickness)
            * (self.grid.status_at_node == 0)
        )

        real_growth = jnp.where(
            max_growth > 0,
            jnp.minimum(max_growth, self.state.till_thickness),
            jnp.maximum(max_growth, -self.state.fringe_thickness)
        )

        fringe_thickness = self.state.fringe_thickness + real_growth
        fringe_thickness = jnp.where(
            fringe_thickness < self.state.min_fringe_thickness,
            self.state.min_fringe_thickness,
            fringe_thickness
        )

        dispersed_thickness = (
            self.state.dispersed_thickness 
            + self.calc_dispersed_growth_rate(self.calc_supercooling(fringe_thickness))
            * dt_s
        )

        till_thickness = self.state.till_thickness - real_growth
        till_thickness = jnp.where(till_thickness < 0, 0, till_thickness)

        updated_state = eqx.tree_at(
            lambda tree: (
                tree.fringe_thickness, tree.dispersed_thickness, tree.till_thickness
            ),
            self.state,
            (fringe_thickness, dispersed_thickness, till_thickness)
        )

        return FrozenFringe(updated_state)
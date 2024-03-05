"""Model subglacial hydrology through a series of conduits."""

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import jaxopt

from landlab import ModelGrid
from landlab.components import FlowAccumulator

from utils import StaticGrid
from components import ModelState

class ConduitHydrology(eqx.Module):
    """Implement a model of subglacial hydrology through a series of conduits."""

    state: ModelState
    grid: StaticGrid = eqx.field(init = False)
    landlab_grid: ModelGrid = eqx.field(static = True)
    
    conduit_area: jnp.array = eqx.field(converter = jnp.asarray)
    melt_flux: jnp.array = eqx.field(converter = jnp.asarray, init = False)
    potential: jnp.array = eqx.field(converter = jnp.asarray, init = False)
    geometric_gradient: jnp.array = eqx.field(converter = jnp.asarray, init = False)
    discharge: jnp.array = eqx.field(converter = jnp.asarray, init = False)

    opening_coeff: float = 1.3455e-9 # J^-1 m^3
    closure_coeff: float = 7.11e-24 # Pa^-3 s^-1
    flow_coeff: float = 4.05e-2 # m^9/4 Pa^-1/2 s^-1
    flow_exp: float = 5 / 4
    step_height: float = 0.03 # m
    scale_cutoff: float = 5.74 # m^2
    n: int = 3

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

        self.discharge = self.route_flow(self.landlab_grid)

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

    def calc_hydraulic_gradient(self, effective_pressure: jnp.array) -> jnp.array:
        """Compute the hydraulic gradient."""
        return (
            self.grid.map_mean_of_links_to_node(
                self.grid.calc_grad_at_link(effective_pressure)
            ) 
            + self.geometric_gradient
        )

    def calc_pressure_residual(self, effective_pressure: jnp.array) -> jnp.array:
        """Compute the residual of the pressure field."""

        # Boundary condition: Pw = 0
        effective_pressure = jnp.where(
            self.grid.status_at_node != 0,
            self.state.overburden_pressure,
            effective_pressure
        )

        gradient = self.calc_hydraulic_gradient(effective_pressure)

        cavity_opening = (
            jnp.abs(
                self.grid.map_mean_of_links_to_node(
                    self.state.sliding_velocity / self.state.sec_per_a
                )
            )
            * self.step_height
        )

        conduit_size = (
            (self.opening_coeff * self.discharge * gradient + cavity_opening)
            /
            (cavity_opening / self.scale_cutoff + self.closure_coeff * effective_pressure**self.n)
        )

        # Regularization to avoid division by close-to-zero
        conduit_size = jnp.where(
            conduit_size < 1e-6,
            1e-6,
            conduit_size
        )

        residual = (
            self.discharge 
            - self.opening_coeff * conduit_size**self.flow_exp 
            * jnp.abs(gradient)**(-1/2) * gradient
        )

        return residual

    def solve_for_pressure(self) -> jnp.array:
        """Solve for the effective pressure."""
        solver = jaxopt.GaussNewton(
            residual_fun = self.calc_pressure_residual,
            maxiter = 30,
            tol = 1e-3,
            verbose = True
        )
        return solver.run(self.state.water_pressure).params



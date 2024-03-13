"""Model subglacial hydrology in an evolving network of conduits."""

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import jaxopt

from landlab import ModelGrid
from landlab.components import FlowAccumulator, FlowDirectorMFD

from utils import StaticGrid
from components import ModelState

class SubglacialDrainageSystem(eqx.Module):
    """Model subglacial conduit evolution, discharge, and pressure."""

    state: ModelState
    grid: StaticGrid = eqx.field(init = False)
    landlab_grid: ModelGrid = eqx.field(static = True)
    
    conduit_size: jnp.array = eqx.field(converter = jnp.asarray)
    surface_melt_rate: jnp.array = eqx.field(converter = jnp.asarray)
    total_melt_rate: jnp.array = eqx.field(converter = jnp.asarray, init = False)
    base_potential: jnp.array = eqx.field(converter = jnp.asarray, init = False)
    geometric_gradient: jnp.array = eqx.field(converter = jnp.asarray, init = False)
    discharge: jnp.array = eqx.field(converter = jnp.asarray, init = False)
    flow_direction: jnp.array = eqx.field(converter = jnp.asarray, init = False)
    inflow_outflow: jnp.array = eqx.field(converter = jnp.asarray, init = False)

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

        self.total_melt_rate = (
            self.state.melt_rate 
            * (self.state.water_density / self.state.ice_density)
            / self.state.sec_per_a
            + self.surface_melt_rate
        ) * self.grid.cell_area_at_node

        self.base_potential = (
            self.state.water_density * self.state.gravity * self.state.bedrock_elevation
            + self.state.overburden_pressure
        )

        self.geometric_gradient = (
            -self.state.ice_density * self.state.gravity * self.state.surface_slope
            -(self.state.water_density - self.state.ice_density) * self.state.gravity 
            * self.state.bedrock_slope
        )

        # Flow goes from high potential to low potential, but links go from tail to head
        self.flow_direction = jnp.where(
            self.base_potential[self.grid.node_at_link_head] >
            self.base_potential[self.grid.node_at_link_tail],
            -1,
            1
        )

        adjacent_potential = jnp.mean(
            jnp.where(
                self.grid.adjacent_nodes_at_node != -1,
                self.base_potential[self.grid.adjacent_nodes_at_node],
                0.0
            ),
            axis = 1
        )

        # 1 denotes inflow, -1 denotes outflow
        self.inflow_outflow = jnp.where(
            self.base_potential > adjacent_potential,
            1 * (self.grid.status_at_node > 0),
            -1 * (self.grid.status_at_node > 0)
        )

        self.discharge = self._route_flow(self.landlab_grid)

    def _route_flow(self, landlab_grid) -> jnp.array:
        """Route discharge based on the hydraulic potential field."""
        fa = FlowAccumulator(
            landlab_grid, 
            surface = self.base_potential,
            runoff_rate = self.total_melt_rate,
            flow_director = 'FlowDirectorMFD'
        )

        _, discharge = fa.accumulate_flow(update_depression_finder = False)

        # Enforce discharge = 0 at inflow boundaries
        discharge = jnp.where(
            self.inflow_outflow == 1,
            0.0,
            discharge
        )

        return self.grid.map_mean_of_link_nodes_to_link(discharge)

    def _calc_hydraulic_gradient(self, conduit_size: jnp.array) -> jnp.array:
        """Calculate the hydraulic gradient through a conduit."""
        gradient = (
            (self.discharge / (self.flow_coeff * conduit_size**self.flow_exp))**2
        ) * self.flow_direction

        return gradient

    def _matvec(self, potential: jnp.array) -> jnp.array:
        """Calculate the residual for the potential field."""
        potential = jnp.where(
            self.inflow_outflow == -1,
            self.state.water_density * self.state.gravity * self.state.bedrock_elevation,
            potential
        )

        return self.grid.calc_flux_div_at_node(
            self.grid.calc_grad_at_link(potential)
        )

    def _solve_for_potential(self, gradient: jnp.array) -> jnp.array:
        """Solve for the hydraulic potential field."""
        solution = jaxopt.linear_solve.solve_bicgstab(
            matvec = self._matvec,
            b = self.grid.calc_flux_div_at_node(gradient),
            atol = 1e-3
        )

        return solution

    def _calc_effective_pressure(self, potential: jnp.array) -> jnp.array:
        """Calculate the effective pressure."""
        water_pressure = (
            potential -
            self.state.water_density * self.state.gravity * self.state.bedrock_elevation
        )

        water_pressure = jnp.minimum(
            jnp.maximum(
                water_pressure,
                0.0
            ),
            self.state.overburden_pressure
        )

        return self.state.overburden_pressure - water_pressure

    def _calc_conduit_roc(self, conduit_size: jnp.array) -> jnp.array:
        """Calculate the rate at which conduits are expanding or contracting."""
        gradient = self._calc_hydraulic_gradient(conduit_size)
        potential = self._solve_for_potential(gradient)
        effective_pressure = self.grid.map_mean_of_link_nodes_to_link(
            self._calc_effective_pressure(potential)
        )
        effective_pressure = jnp.where(
            effective_pressure < 1.0, 1.0, effective_pressure
        )

        melt_opening = self.opening_coeff * self.discharge * gradient * self.flow_direction
        closure = self.closure_coeff * effective_pressure**self.n * conduit_size
 
        return melt_opening - closure

    @jax.jit
    def update(self, dt: float):
        """Advance the model by dt seconds."""
        updated_conduit_size = jnp.maximum(
            self.conduit_size + self._calc_conduit_roc(self.conduit_size) * dt,
            1e-3
        )

        return eqx.tree_at(
            lambda tree: tree.conduit_size,
            self,
            updated_conduit_size
        )
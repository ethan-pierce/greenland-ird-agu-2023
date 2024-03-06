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
    
    conduit_size: jnp.array = eqx.field(converter = jnp.asarray)
    total_melt_rate: jnp.array = eqx.field(converter = jnp.asarray, init = False)
    base_potential: jnp.array = eqx.field(converter = jnp.asarray, init = False)
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

        self.total_melt_rate = (
            self.state.melt_rate 
            * (self.state.water_density / self.state.ice_density)
            / self.state.sec_per_a
        ) + 3.5e-7 # Pretend this is ~3 cm per day of surface melt

        self.base_potential = (
            self.state.water_density * self.state.gravity * self.state.bedrock_elevation
            + self.state.overburden_pressure
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
            surface = self.base_potential,
            runoff_rate = self.total_melt_rate,
            flow_director = 'FlowDirectorMFD'
        )

        area, discharge = fa.accumulate_flow(update_depression_finder = False)

        return discharge

    def calc_hydraulic_gradient(self, conduit_size: jnp.array) -> jnp.array:
        """Calculate the hydraulic gradient through a conduit."""
        gradient = (
            (self.discharge * self.flow_coeff * conduit_size**self.flow_exp)**2
        )

        return gradient

    def solve_for_potential(self, gradient_at_nodes: jnp.array) -> jnp.array:
        """Solve for the hydraulic potential field."""
        gradient = self.grid.map_mean_of_link_nodes_to_link(gradient_at_nodes)
        
        inactive_links = (
            (self.grid.status_at_node[self.grid.node_at_link_head] != 0)
            |
            (self.grid.status_at_node[self.grid.node_at_link_tail] != 0)
        )

        gradient = jnp.where(
            inactive_links,
            self.grid.map_mean_of_link_nodes_to_link(self.geometric_gradient),
            gradient
        )

        div_f = self.grid.calc_flux_div_at_node(gradient)

        laplace = lambda x: self.grid.calc_flux_div_at_node(
            self.grid.calc_grad_at_link(x)
        )

        solution = jaxopt.linear_solve.solve_cg(
            matvec = laplace,
            b = div_f,
            tol = 1e-3
        )

        return solution

    def calc_effective_pressure(self, potential: jnp.array) -> jnp.array:
        """Calculate the effective pressure."""
        return self.geometric_gradient - potential

    def calc_conduits_roc(self, conduit_size: jnp.array) -> jnp.array:
        """Calculate the rate of closure of the conduits."""
        gradient = self.calc_hydraulic_gradient(conduit_size)
        potential = self.solve_for_potential(gradient)
        pressure = self.calc_effective_pressure(potential)

        melt_opening = self.opening_coeff * self.discharge * gradient

        gap_opening = (
            jnp.abs(
                self.grid.map_mean_of_links_to_node(
                    self.state.sliding_velocity / self.state.sec_per_a
                )
            )
            * self.step_height
            * (1 - jnp.tanh(conduit_size / self.scale_cutoff))
        )

        closure = self.closure_coeff * pressure**self.n * conduit_size

        return melt_opening + gap_opening - closure

    @jax.jit
    def update_conduits(self, dt: float):
        """Update the size of the conduits and return an updated copy of this object."""
        k1 = self.calc_conduits_roc(self.conduit_size)
        k2 = self.calc_conduits_roc(self.conduit_size + dt / 2 * k1)
        k3 = self.calc_conduits_roc(self.conduit_size + dt / 2 * k2)
        k4 = self.calc_conduits_roc(self.conduit_size + dt * k3)

        new_conduit_size = self.conduit_size + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        return eqx.tree_at(
            lambda tree: tree.conduit_size,
            self,
            new_conduit_size
        )

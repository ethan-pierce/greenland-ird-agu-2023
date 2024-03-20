"""Model subglacial hydrology in an evolving network of conduits."""

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import lineax as lx
import optimistix as optx

from landlab import ModelGrid
from landlab.components import FlowAccumulator, FlowDirectorMFD

from utils import StaticGrid
from components import ModelState

from jax import config
config.update("jax_enable_x64", True)

class SubglacialDrainageSystem(eqx.Module):
    """Model subglacial conduit evolution, discharge, and pressure."""

    state: ModelState
    grid: StaticGrid = eqx.field(init = False)
    landlab_grid: ModelGrid = eqx.field(static = True)
    
    surface_melt_rate: jnp.array = eqx.field(converter = jnp.asarray)
    specific_melt_rate: jnp.array = eqx.field(converter = jnp.asarray, init = False)
    base_potential: jnp.array = eqx.field(converter = jnp.asarray, init = False)
    flow_direction: jnp.array = eqx.field(converter = jnp.asarray, init = False)
    inflow_outflow: jnp.array = eqx.field(converter = jnp.asarray, init = False)
    discharge: jnp.array = eqx.field(converter = jnp.asarray, init = False)
    potential: jnp.array = eqx.field(converter = jnp.asarray, init = False)

    sheet_conductivity: float = 1e-2 # m^7/4 kg^-1/2
    sheet_flow_exp: float = 5/4
    bedrock_step_height: float = 0.1 # m
    cavity_spacing: float = 2.0 # m
    cavity_closure_coeff: float = 5e-25 # Pa^-3 s^-1
    n: int = 3 # Glen's flow law

    def __post_init__(self):
        """Initialize remaining model fields."""
        self.grid = self.state.grid

        self.specific_melt_rate = (
            self.state.melt_rate 
            * (self.state.water_density / self.state.ice_density)
            / self.state.sec_per_a
            + self.surface_melt_rate
        )

        self.base_potential = (
            self.state.water_density * self.state.gravity * self.state.bedrock_elevation
            + self.state.overburden_pressure
        )

        self.potential = jnp.zeros(self.grid.number_of_nodes)

        self._set_flow_direction()
        self.discharge = self._route_flow(self.landlab_grid)

    def _set_flow_direction(self):
        """Set the flow direction and inflow/outflow at each node."""

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

    def _route_flow(self, landlab_grid) -> jnp.array:
        """Route discharge based on the hydraulic potential field."""
        fa = FlowAccumulator(
            landlab_grid, 
            surface = self.base_potential,
            runoff_rate = self.specific_melt_rate,
            flow_director = 'FlowDirectorMFD'
        )

        _, discharge = fa.accumulate_flow(update_depression_finder = False)

        discharge = jnp.where(
            self.grid.status_at_node != 0,
            0.0,
            discharge
        )

        return self.grid.map_mean_of_link_nodes_to_link(discharge)

    def _calc_sheet_thickness(self, potential: jnp.array) -> jnp.array:
        """Calculate the mean thickness of flow across the distributed system."""
        sliding = (
            jnp.abs(self.grid.map_mean_of_links_to_node(self.state.sliding_velocity))
            / self.state.sec_per_a
        )
        pressure = self.base_potential - potential

        thickness = (
            (sliding**2 * self.bedrock_step_height) 
            / (self.cavity_closure_coeff * pressure**self.n * self.cavity_spacing**2)
        )

        return jnp.where(
            thickness < 0.0, 0.0, thickness
        )

    def _potential_residual(self, potential: jnp.array) -> jnp.array:
        """Calculate the residual of the potential field."""
        potential = jnp.where(
            self.inflow_outflow == -1,
            self.base_potential - self.state.overburden_pressure,
            potential
        )
        h = self.grid.map_mean_of_link_nodes_to_link(
            self._calc_sheet_thickness(potential)
        )

        gradient = self.grid.calc_grad_at_link(potential)

        flux = (
            -self.sheet_conductivity 
            * h**self.sheet_flow_exp
            * jnp.abs(gradient)**(-1/2)
            * gradient
        )
        flux = jnp.where(
            (self.inflow_outflow[self.grid.node_at_link_head] == 1)
            | (self.inflow_outflow[self.grid.node_at_link_tail] == 1),
            0.0,
            flux
        )

        forcing = self.discharge

        return jnp.abs(flux - forcing)

    def _solve_for_potential(self) -> jnp.array:
        """Solve for the hydraulic potential in the distributed system."""
        residual = lambda phi, args: self._potential_residual(phi)
        solver = optx.LevenbergMarquardt(
            rtol = 1e-6, atol = 1e-6,
            linear_solver = lx.QR(),
            norm = optx.two_norm,
            verbose = frozenset({'step', 'loss', 'step_size'})
        )

        solution = optx.least_squares(
            residual,
            solver = solver,
            y0 = self.base_potential - self.state.overburden_pressure,
            args = None
        )

        return solution.value




    # def _calc_forcing(self, sheet_thickness: jnp.array) -> jnp.array:
    #     """Calculate the forcing term for the sheet evolution equation."""
    #     sheet_on_links = self.grid.map_mean_of_link_nodes_to_link(sheet_thickness)
    #     flux = (
    #         self.flow_direction *
    #         (
    #             self.discharge 
    #             / 
    #             (self.sheet_conductivity * sheet_on_links**self.sheet_flow_exp)
    #         )**2
    #     )

    #     flux = jnp.where(
    #         (self.inflow_outflow[self.grid.node_at_link_head] == 1) |
    #         (self.inflow_outflow[self.grid.node_at_link_tail] == 1),
    #         0.0,
    #         flux
    #     )

    #     return self.grid.calc_flux_div_at_node(flux)
    
    # def _calc_laplacian(self, potential: jnp.array) -> jnp.array:
    #     """Calculate the Laplacian of the potential, applying boundary conditions."""
    #     potential = jnp.where(
    #         self.inflow_outflow == -1,
    #         self.base_potential - self.state.overburden_pressure,
    #         potential
    #     )

    #     gradient = self.grid.calc_grad_at_link(potential)

    #     return self.grid.calc_flux_div_at_node(gradient)

    # def _solve_for_potential(self, sheet_thickness: jnp.array) -> jnp.array:
    #     """Solve for the potential of the sheet."""
    #     forcing = self._calc_forcing(sheet_thickness)
    #     residual = lambda phi, args: forcing + self._calc_laplacian(phi)
    #     solver = optx.LevenbergMarquardt(
    #         rtol = 1e-5, atol = 1e-3,
    #         verbose = frozenset({'step', 'loss', 'step_size'})
    #     )

    #     solution = optx.least_squares(
    #         residual,
    #         solver = solver,
    #         y0 = self.base_potential - self.state.overburden_pressure,
    #         args = None
    #     )
    #     return solution.value

    # def _calc_sheet_growth_rate(self, sheet_thickness: jnp.array, potential: jnp.array) -> jnp.array:
    #     """Calculate the growth rate of the sheet."""
    #     step_differential = jnp.where(
    #         sheet_thickness < self.bedrock_step_height,
    #         (self.bedrock_step_height - sheet_thickness) / self.cavity_spacing,
    #         0.0
    #     )
    #     sliding_at_nodes = self.grid.map_mean_of_links_to_node(self.state.sliding_velocity)
    #     opening = jnp.abs(sliding_at_nodes) / self.state.sec_per_a * step_differential

    #     pressure = (self.base_potential - potential)**self.n
    #     closure = self.cavity_closure_coeff * pressure * sheet_thickness

    #     return opening - closure

    # @jax.jit
    # def _update_sheet_flow(self, dt: float):
    #     """Update the distributed components of the drainage system."""
    #     potential = self._solve_for_potential(self.sheet_thickness)
        
    #     k1 = self._calc_sheet_growth_rate(self.sheet_thickness, potential)
    #     k2 = self._calc_sheet_growth_rate(self.sheet_thickness + 0.5 * dt * k1, potential)
    #     k3 = self._calc_sheet_growth_rate(self.sheet_thickness + 0.5 * dt * k2, potential)
    #     k4 = self._calc_sheet_growth_rate(self.sheet_thickness + dt * k3, potential)

    #     new_sheet_thickness = self.sheet_thickness + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    #     new_sheet_thickness = jnp.where(
    #         new_sheet_thickness < 0.0,
    #         0.0,
    #         new_sheet_thickness
    #     )

    #     updated = eqx.tree_at(lambda tree: tree.potential, self, potential)

    #     return eqx.tree_at(
    #         lambda tree: tree.sheet_thickness, updated, new_sheet_thickness
    #     )
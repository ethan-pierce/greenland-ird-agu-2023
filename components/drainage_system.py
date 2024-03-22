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
    potential: jnp.array = eqx.field(converter = jnp.asarray)
    sheet_thickness: jnp.array = eqx.field(converter = jnp.asarray)

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

        self._set_flow_direction()
        self.discharge = self._route_flow(self.landlab_grid, self.base_potential)

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

    def _route_flow(self, landlab_grid, potential: jnp.array) -> jnp.array:
        """Route discharge based on the hydraulic potential field."""
        fa = FlowAccumulator(
            landlab_grid, 
            surface = potential,
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

    def _assemble_elliptic_operator(self, sheet_thickness: jnp.array, previous_potential: jnp.array):
        """Assemble the linear operator for the elliptic equation in potential."""
        matrix = np.zeros((self.grid.number_of_nodes, self.grid.number_of_nodes))
        
        for i in range(self.grid.number_of_nodes):
            if self.inflow_outflow[i] == 1:
                matrix[i, i] = 1
            else:
                j = self.grid.adjacent_nodes_at_node[i][self.grid.adjacent_nodes_at_node[i] != -1]
                all_links = np.intersect1d(self.grid.links_at_node[i], self.grid.links_at_node[j])
                valid_links = all_links[all_links != -1]
                lengths = self.grid.length_of_face[self.grid.face_at_link[valid_links]]
                prev_gradients = np.power(
                    np.abs((previous_potential[i] - previous_potential[j]) / lengths),
                    -1/2
                )
                sheets = 0.5 * (sheet_thickness[i] + sheet_thickness[j])

                matrix[i, j] += -0.01 * sheets**(5/4) * lengths * prev_gradients
                matrix[i, i] -= np.sum(-0.01 * sheets**(5/4) * lengths * prev_gradients)

        return jnp.asarray(matrix)

    def _solve_for_potential(self, sheet_thickness: jnp.array, previous_potential: jnp.array):
        """Solve the elliptic equation for potential."""
        matrix = self._assemble_elliptic_operator(sheet_thickness, previous_potential)
        operator = lx.MatrixLinearOperator(matrix)
        forcing = jnp.where(
            self.inflow_outflow == 1,
            self.base_potential - self.state.overburden_pressure,
            self.specific_melt_rate
        )
        solution = lx.linear_solve(operator, forcing)

        return solution.value

    def _update_sheet_thickness(self, dt: float, sheet_thickness: jnp.array, potential: jnp.array):
        """Update the mean thickness of flow in the distributed system."""
        sliding = jnp.abs(
            self.grid.map_mean_of_links_to_node(
                self.state.sliding_velocity / self.state.sec_per_a
            )
        )

        pressure = self.base_potential - potential

        dhdt = lambda h: dt * (
            sliding * (self.bedrock_step_height - h) / self.cavity_spacing
            - self.cavity_closure_coeff * pressure**self.n * h
        )

        residual = lambda h, _: h - sheet_thickness - dhdt(h)

        solver = optx.Newton(rtol = 1e-5, atol = 1e-5)
        solution = optx.root_find(residual, solver, sheet_thickness, args = None)

        return solution.value

    def update(self, dt: float):
        """Advance the model by one step of dt seconds."""
        potential = self._solve_for_potential(self.sheet_thickness, self.potential)
        sheet_thickness = self._update_sheet_thickness(dt, self.sheet_thickness, potential)

        updated = eqx.tree_at(lambda t: t.potential, self, potential)
        return eqx.tree_at(lambda t: t.sheet_thickness, updated, sheet_thickness)
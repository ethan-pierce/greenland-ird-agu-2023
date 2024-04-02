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
    channel_size: jnp.array = eqx.field(converter = jnp.asarray)
    sheet_thickness: jnp.array = eqx.field(converter = jnp.asarray)

    links_between_nodes: jnp.array = eqx.field(converter = jnp.asarray, init = False)

    sheet_conductivity: float = 1e-2 # m^7/4 kg^-1/2
    flow_exp: float = 5/4
    channel_conductivity: float = 0.1 # m^3/2 kg^-1/2
    bedrock_step_height: float = 0.1 # m
    cavity_spacing: float = 2.0 # m
    closure_coeff: float = 5e-25 # Pa^-3 s^-1
    pressure_melt_coeff: float = 7.5e-8 # K Pa^-1
    heat_capacity: float = 4.22e3 # J kg^-1 K^-1
    n: int = 3 # Glen's flow law
    min_channel_size: float = 0.0 # m^2

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

        self.links_between_nodes = self.grid.build_link_between_nodes_array()

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

    def _calc_exchange(self, potential: jnp.array, sheet_thickness: jnp.array) -> jnp.array:
        """Calculate the exchange of water between sheets and channels."""
        flow_over_patch = self.grid.map_mean_of_patch_nodes_to_patch(
            -self.sheet_conductivity * sheet_thickness**self.flow_exp
        )
        _, comps = self.grid.calc_grad_at_patch(potential)
        grad_at_patches = jnp.asarray([comps[0], comps[1]]).T * flow_over_patch[:, None]
        normal_at_links = self.grid.get_normal_at_links()

        def q_dot_n(link):
            a = jnp.dot(
                grad_at_patches[self.grid.patches_at_link][link, 0],
                normal_at_links[link, 0]
            )
            b = jnp.dot(
                grad_at_patches[self.grid.patches_at_link][link, 1],
                normal_at_links[link, 1]
            )
            return jnp.nansum(jnp.asarray([a, b]))

        return jax.vmap(q_dot_n)(jnp.arange(self.grid.number_of_links))

    def _assemble_full_operator(
        self, 
        previous_potential: jnp.array,
        channel_size: jnp.array,
        sheet_thickness: jnp.array
    ):
        """Assemble the linear operator for the elliptic equation in potential."""
        previous_gradients = self.grid.calc_grad_at_link(previous_potential)
        channel_discharge = (
            -self.channel_conductivity * channel_size**self.flow_exp * previous_gradients
        )
        sheet_discharge = (
            -self.sheet_conductivity 
            * self.grid.map_mean_of_link_nodes_to_link(sheet_thickness)**self.flow_exp 
            * jnp.abs(previous_gradients)**(-1/2)
            * previous_gradients
        )
        dissipation_coeff = (
            ((1 / self.state.ice_density) - (1 / self.state.water_density))
            * (1 / self.state.ice_latent_heat)
        )

        matrix = np.zeros((self.grid.number_of_nodes, self.grid.number_of_nodes))

        for i in range(self.grid.number_of_nodes):
            if self.inflow_outflow[i] == 1:
                matrix[i, i] = 1
            else:
                j = self.grid.adjacent_nodes_at_node[i][self.grid.adjacent_nodes_at_node[i] != -1]
                all_links = np.intersect1d(self.grid.links_at_node[i], self.grid.links_at_node[j])
                valid_links = all_links[all_links != -1]

                face_lengths = self.grid.length_of_face[self.grid.face_at_link[valid_links]]
                link_lengths = self.grid.length_of_link[valid_links]
                
                channel_size_on_links = 0.5 * (channel_size[i] + channel_size[j])
                sheet_thickness_on_links = 0.5 * (sheet_thickness[i] + sheet_thickness[j])
                
                sheet_flux = (
                    -self.sheet_conductivity 
                    * sheet_thickness_on_links**self.flow_exp
                    * jnp.abs(previous_gradients[valid_links])**(-1/2)
                    * face_lengths
                    / link_lengths
                )

                channel_flux = (
                    -self.channel_conductivity
                    * channel_size_on_links**self.flow_exp
                    * face_lengths
                    / link_lengths
                )

                channel_dissipation = (
                    dissipation_coeff
                    * channel_discharge[valid_links]
                    * face_lengths
                )

                sheet_dissipation = (
                    dissipation_coeff
                    * sheet_discharge[valid_links]
                    * self.cavity_spacing
                    * face_lengths
                )

                matrix[i, j] -= sheet_flux + channel_flux + channel_dissipation + sheet_dissipation
                matrix[i, i] += jnp.sum(
                    sheet_flux + channel_flux + channel_dissipation + sheet_dissipation
                )

        return jnp.asarray(matrix)

    @jax.jit
    def _jit_assemble_operator(
        self,
        previous_potential: jnp.array,
        channel_size: jnp.array,
        sheet_thickness: jnp.array
    ):
        """JIT-compatible assembly of the linear operator."""
        previous_gradients = self.grid.calc_grad_at_link(previous_potential)
        channel_discharge = (
            -self.channel_conductivity * channel_size**self.flow_exp * previous_gradients
        )
        sheet_discharge = (
            -self.sheet_conductivity 
            * self.grid.map_mean_of_link_nodes_to_link(sheet_thickness)**self.flow_exp 
            * jnp.abs(previous_gradients)**(-1/2)
            * previous_gradients
        )
        dissipation_coeff = (
            ((1 / self.state.ice_density) - (1 / self.state.water_density))
            * (1 / self.state.ice_latent_heat)
        )

        def assemble_off_diag(out, i, j):
            link = self.links_between_nodes[i, j]
            face_length = self.grid.length_of_face[self.grid.face_at_link[link]]
            link_length = self.grid.length_of_link[link]
            channel_size_on_link = 0.5 * (channel_size[i] + channel_size[j])
            sheet_thickness_on_link = 0.5 * (sheet_thickness[i] + sheet_thickness[j])

            sheet_flux = (
                -self.sheet_conductivity 
                * sheet_thickness_on_link**self.flow_exp
                * jnp.abs(previous_gradients[link])**(-1/2)
                * face_length
                / link_length
            )

            channel_flux = (
                -self.channel_conductivity
                * channel_size_on_link**self.flow_exp
                * face_length
                / link_length
            )

            channel_dissipation = jnp.abs(
                dissipation_coeff
                * channel_discharge[link]
                * face_length
            )

            sheet_dissipation = jnp.abs(
                dissipation_coeff
                * sheet_discharge[link]
                * self.cavity_spacing
                * face_length
            )

            return out.at[j].set(
                -(sheet_flux + channel_flux + channel_dissipation + sheet_dissipation)
            ).at[i].add(
                sheet_flux + channel_flux + channel_dissipation + sheet_dissipation
            )

        def assemble_interior(i):
            out = jnp.zeros(self.grid.number_of_nodes)

            for j in self.grid.adjacent_nodes_at_node[i]:
                out = jnp.where(
                    j != -1,
                    assemble_off_diag(out, i, j),
                    out
                )

            return out

        def assemble_boundary(i):
            return jnp.zeros(self.grid.number_of_nodes).at[i].set(1)

        def assemble_row(i):
            return jax.lax.cond(
                self.inflow_outflow[i] == 1,
                assemble_boundary,
                assemble_interior,
                i
            )

        return jax.vmap(assemble_row)(jnp.arange(self.grid.number_of_nodes))
        
    def _calc_forcing(
        self, 
        previous_potential: jnp.array,
        channel_size: jnp.array,
        sheet_thickness: jnp.array
    ):
        """Calculate the forcing term for the elliptic equation in potential."""
        previous_gradients = self.grid.calc_grad_at_link(previous_potential)

        channel_source = jnp.sum(
            self._calc_exchange(previous_potential, sheet_thickness)[self.grid.links_at_node],
            axis = 1
        )

        channel_closure = (
            self.closure_coeff 
            * (self.base_potential - previous_potential)**self.n
            * self.grid.map_mean_of_links_to_node(channel_size)
        )

        interior_forcing = channel_closure + channel_source + self.specific_melt_rate

        return jnp.where(
            self.inflow_outflow == 1,
            self.base_potential - self.state.overburden_pressure,
            interior_forcing
        )

    def _solve_for_potential(
        self,
        previous_potential: jnp.array,
        channel_size: jnp.array,
        sheet_thickness: jnp.array
    ):
        """Solve the elliptic equation for potential."""
        matrix = self._jit_assemble_operator(previous_potential, channel_size, sheet_thickness)
        operator = lx.MatrixLinearOperator(matrix)
        forcing = self._calc_forcing(previous_potential, channel_size, sheet_thickness)
        solution = lx.linear_solve(operator, forcing)

        return solution.value
    
    def _update_channel_size(
        self, 
        dt: float, 
        channel_size: jnp.array, 
        sheet_thickness: jnp.array,
        potential: jnp.array
    ):
        """Update the cross-sectional area of channels on every link."""
        dSdt = lambda S: dt * (
            (
                jnp.abs(
                    self.channel_conductivity * S**self.flow_exp * self.grid.calc_grad_at_link(potential)
                )
                + jnp.abs(
                    self.sheet_conductivity 
                    * self.cavity_spacing
                    * self.grid.map_mean_of_link_nodes_to_link(sheet_thickness)**self.flow_exp 
                    * self.grid.calc_grad_at_link(potential)
                )
            )
            / (self.state.ice_density * self.state.ice_latent_heat)
            - self.closure_coeff 
            * self.grid.map_mean_of_link_nodes_to_link(self.base_potential - potential)**self.n 
            * S
        )

        residual = lambda S, _: S - channel_size - dSdt(S)

        solver = optx.Newton(rtol = 1e-5, atol = 1e-5)
        solution = optx.root_find(residual, solver, channel_size, args = None)

        channel_size = jnp.where(
            (self.inflow_outflow[self.grid.node_at_link_head] == -1) &
            (self.inflow_outflow[self.grid.node_at_link_tail] == -1),
            0.0,
            solution.value
        )

        return jnp.where(
            channel_size < self.min_channel_size,
            self.min_channel_size,
            channel_size
        )

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
            - self.closure_coeff * pressure**self.n * h
        )

        residual = lambda h, _: h - sheet_thickness - dhdt(h)

        solver = optx.Newton(rtol = 1e-5, atol = 1e-5)
        solution = optx.root_find(residual, solver, sheet_thickness, args = None)

        return solution.value

    # @jax.jit
    def update(self, dt: float):
        """Advance the model by one step."""
        potential = self._solve_for_potential(
            self.potential, self.channel_size, self.sheet_thickness
        )
        sheet_thickness = self._update_sheet_thickness(dt, self.sheet_thickness, potential)
        channel_size = self._update_channel_size(dt, self.channel_size, sheet_thickness, potential)

        updated = eqx.tree_at(
            lambda t: (t.potential, t.channel_size, t.sheet_thickness),
            self,
            (potential, channel_size, sheet_thickness)
        )

        return updated

# THE FOLLOWING FUNCTIONS MODEL THE DISTRIBUTED DOMAIN ONLY
# def update_sheet_flow(self, dt: float):
#     """Advance the model by one step in the distributed system."""
#     potential = self._solve_for_potential(self.sheet_thickness, self.potential)
#     sheet_thickness = self._update_sheet_thickness(dt, self.sheet_thickness, potential)

#     updated = eqx.tree_at(lambda t: t.potential, self, potential)
#     return eqx.tree_at(lambda t: t.sheet_thickness, updated, sheet_thickness)

# def _assemble_elliptic_operator(self, sheet_thickness: jnp.array, previous_potential: jnp.array):
#     """Assemble the linear operator for the elliptic equation in potential."""
#     matrix = np.zeros((self.grid.number_of_nodes, self.grid.number_of_nodes))
    
#     for i in range(self.grid.number_of_nodes):
#         if self.inflow_outflow[i] == 1:
#             matrix[i, i] = 1
#         else:
#             j = self.grid.adjacent_nodes_at_node[i][self.grid.adjacent_nodes_at_node[i] != -1]
#             all_links = np.intersect1d(self.grid.links_at_node[i], self.grid.links_at_node[j])
#             valid_links = all_links[all_links != -1]
#             lengths = self.grid.length_of_face[self.grid.face_at_link[valid_links]]
#             prev_gradients = np.power(
#                 np.abs((previous_potential[i] - previous_potential[j]) / lengths),
#                 -1/2
#             )
#             sheets = 0.5 * (sheet_thickness[i] + sheet_thickness[j])

#             matrix[i, j] += -0.01 * sheets**(5/4) * lengths * prev_gradients
#             matrix[i, i] -= np.sum(-0.01 * sheets**(5/4) * lengths * prev_gradients)

#     return jnp.asarray(matrix)

# def _solve_for_potential(self, sheet_thickness: jnp.array, previous_potential: jnp.array):
#     """Solve the elliptic equation for potential."""
#     matrix = self._assemble_elliptic_operator(sheet_thickness, previous_potential)
#     operator = lx.MatrixLinearOperator(matrix)
#     forcing = jnp.where(
#         self.inflow_outflow == 1,
#         self.base_potential - self.state.overburden_pressure,
#         self.specific_melt_rate
#     )
#     solution = lx.linear_solve(operator, forcing)

#     return solution.value

# THE FOLLOWING FUNCTIONS HANDLE THE PROBLEM IN A NON-LINEAR LEAST SQUARES FORM
    # def _channel_residual(
    #     self, 
    #     potential: jnp.array, 
    #     args: tuple
    # ) -> jnp.array:
    #     """Calculate the residual in the channelized system."""
    #     channel_size, sheet_thickness = args
    #     exchange = self._calc_exchange(sheet_thickness, potential)
    #     pressure = self.grid.map_mean_of_link_nodes_to_link(
    #         self.base_potential - potential
    #     )
    #     closure = self.closure_coeff * pressure**self.n * channel_size

    #     gradient = self.grid.calc_grad_at_link(potential)

    #     channel_discharge = (
    #         -self.channel_conductivity * channel_size**self.flow_exp * gradient
    #     )

    #     sheet_contribution = (
    #         -self.sheet_conductivity 
    #         * self.cavity_spacing 
    #         * self.grid.map_mean_of_link_nodes_to_link(sheet_thickness)**self.flow_exp
    #         * gradient
    #     )

    #     dissipation = jnp.abs(channel_discharge * gradient) + jnp.abs(sheet_contribution * gradient)

    #     density_coeff = (
    #         ((1 / self.state.ice_density) - (1 / self.state.water_density)) 
    #         * (1 / self.state.ice_latent_heat)
    #     )

    #     flux_div = self.grid.map_mean_of_link_nodes_to_link(
    #         self.grid.calc_flux_div_at_node(channel_discharge)
    #     )

    #     return flux_div + dissipation * density_coeff - closure - exchange

    # def _total_residual(
    #     self, 
    #     potential: jnp.array, 
    #     args: tuple
    # ) -> jnp.array:
    #     """Calculate the total residual in the system."""
    #     channel_size, sheet_thickness = args
    #     channel_residual = self._channel_residual(potential, args)
    #     channel_res_nodes = self.grid.map_mean_of_links_to_node(channel_residual)

    #     gradient = self.grid.calc_grad_at_link(potential)
    #     h_on_links = self.grid.map_mean_of_link_nodes_to_link(sheet_thickness)
    #     sheet_discharge = (
    #         -self.sheet_conductivity * h_on_links**self.flow_exp * gradient
    #     )
    #     flux_div = self.grid.calc_flux_div_at_node(sheet_discharge)
    #     sheet_residual = flux_div - self.specific_melt_rate / self.cavity_spacing

    #     return jnp.abs(channel_res_nodes) + jnp.abs(sheet_residual)


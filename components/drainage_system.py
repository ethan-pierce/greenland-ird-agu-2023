"""Model subglacial hydrology in an evolving network of conduits."""

import jax
import jax.numpy as jnp
import equinox as eqx
import lineax as lx
import optimistix as optx

from utils import StaticGrid
from components import ModelState

from jax import config
config.update("jax_enable_x64", True)

class SubglacialDrainageSystem(eqx.Module):
    """Model subglacial conduit evolution, discharge, and pressure."""

    state: ModelState
    grid: StaticGrid = eqx.field(init = False)
    
    surface_melt_rate: jnp.array = eqx.field(converter = jnp.asarray)
    potential: jnp.array = eqx.field(converter = jnp.asarray)
    sheet_thickness: jnp.array = eqx.field(converter = jnp.asarray)
    channel_size: jnp.array = eqx.field(converter = jnp.asarray)

    links_between_nodes: jnp.array = eqx.field(converter = jnp.asarray, init = False)

    sheet_conductivity: float = 1e-2 # m^7/4 kg^-1/2
    sheet_flow_exp: float = 5/4
    channel_conductivity: float = 0.1 # m^3/2 kg^-1/2
    channel_flow_exp: float = 3
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
        self.links_between_nodes = self.grid.build_link_between_nodes_array()

    @property
    def base_potential(self) -> jnp.array:
        """Calculate the base potential from bedrock elevation."""
        return self.state.water_density * self.state.gravity * self.state.bedrock_elevation

    @property
    def overburden_potential(self) -> jnp.array:
        """Calculate the overburden pressure potential."""
        return (
            self.base_potential +
            self.state.ice_density * self.state.gravity * self.state.ice_thickness
        )

    @property
    def melt_input(self) -> jnp.array:
        """Calculate the melt input to the system."""
        return (
            self.state.melt_rate / self.state.sec_per_a
            + self.surface_melt_rate
        )

    def set_flow_direction(self, potential: jnp.array) -> jnp.array:
        """Set the flow direction and inflow/outflow at each node."""

        # Flow goes from high potential to low potential, but links go from tail to head
        flow_direction = jnp.where(
            potential[self.grid.node_at_link_head] > potential[self.grid.node_at_link_tail],
            -1,
            1
        )

        return flow_direction
    
    def set_inflow_outflow(self, potential: jnp.array) -> jnp.array:
        """Determine whether boundary nodes expect inflow or outflow."""
        max_adj_potential = jnp.max(
            jnp.where(
                self.grid.adjacent_nodes_at_node != -1,
                potential[self.grid.adjacent_nodes_at_node],
                -jnp.inf
            ),
            axis = 1
        )

        # 1 denotes inflow, -1 denotes outflow
        inflow_outflow = jnp.where(
            potential > max_adj_potential,
            1 * (self.grid.status_at_node > 0),
            -1 * (self.grid.status_at_node > 0)
        )

        return inflow_outflow

    def calc_water_pressure(self, potential: jnp.array) -> jnp.array:
        """Calculate the water pressure from potential."""
        return potential - self.base_potential

    def calc_effective_pressure(self, potential: jnp.array) -> jnp.array:
        """Calculate the effective pressure from potential."""
        return self.overburden_potential - potential

    def sheet_discharge_on_links(self, potential: jnp.array, sheet_thickness: jnp.array) -> jnp.array:
        """Interpolate discharge from the distributed system onto grid links."""
        gradient = self.grid.calc_grad_at_link(potential)
        sheet_thickness_on_links = self.grid.map_mean_of_link_nodes_to_link(sheet_thickness)

        return (
            -self.sheet_conductivity 
            * sheet_thickness_on_links**self.sheet_flow_exp 
            * jnp.abs(gradient)**(-1/2) 
            * gradient
        )

    def channel_discharge(self, potential: jnp.array, channel_size: jnp.array) -> jnp.array:
        """Calculate the discharge in active channels."""
        gradient = self.grid.calc_grad_at_link(potential)

        return (
            -self.channel_conductivity 
            * channel_size**self.channel_flow_exp 
            * gradient
        )

    def calc_sheet_opening(self, sheet_thickness: jnp.array) -> jnp.array:
        """Calculate the opening rate of distributed sheet flow."""
        sliding = self.grid.map_mean_of_links_to_node(
            jnp.abs(self.state.sliding_velocity) / self.state.sec_per_a
        )

        return jnp.where(
            sheet_thickness < self.bedrock_step_height,
            sliding * (self.bedrock_step_height - sheet_thickness) / self.cavity_spacing,
            0.0
        )

    def calc_sheet_closure(self, potential: jnp.array, sheet_thickness: jnp.array) -> jnp.array:
        """Calculate the closure rate of distributed sheet flow."""
        N = self.calc_effective_pressure(potential)

        return (
            self.closure_coeff 
            * sheet_thickness 
            * jnp.where(N > 0, N, 0)**self.n
        )

    def calc_channel_closure(self, potential: jnp.array, channel_size: jnp.array) -> jnp.array:
        """Calculate the closure rate of active channels."""
        N = self.grid.map_mean_of_link_nodes_to_link(
            self.calc_effective_pressure(potential)
        )

        return (
            self.closure_coeff
            * channel_size
            * jnp.where(N > 0, N, 0)**self.n
        )

    def energy_dissipation(
        self,
        potential: jnp.array,
        sheet_thickness: jnp.array,
        channel_size: jnp.array
    ) -> jnp.array:
        """Calculate the dissipation of potential energy in the channelized system."""
        gradient = self.grid.calc_grad_at_link(potential)
        sheet_discharge = self.sheet_discharge_on_links(potential, sheet_thickness)
        channel_discharge = self.channel_discharge(potential, channel_size)

        return (
            jnp.abs(channel_discharge * gradient) 
            + jnp.abs(self.cavity_spacing * sheet_discharge * gradient)
        )

    def sensible_heat(
        self,
        potential: jnp.array,
        sheet_thickness: jnp.array,
        channel_size: jnp.array
    ) -> jnp.array:
        """Calculate the sensible heat change in the channelized system."""
        heat_coeff = (
            -self.pressure_melt_coeff * self.heat_capacity * self.state.water_density
        )
        sheet_discharge = self.sheet_discharge_on_links(potential, sheet_thickness)
        channel_discharge = self.channel_discharge(potential, channel_size)
        water_pressure = self.calc_water_pressure(potential)
        pressure_gradient = self.grid.calc_grad_at_link(water_pressure)

        total_discharge = jnp.where(
            (channel_size > 0) | (pressure_gradient * sheet_discharge > 0),
            channel_discharge + self.cavity_spacing * sheet_discharge,
            channel_discharge
        )

        return heat_coeff * total_discharge * pressure_gradient

    def exchange_term(self, potential: jnp.array, sheet_thickness: jnp.array) -> jnp.array:
        """Calculate the exchange of water between sheets and channels."""
        mag, comps = self.grid.calc_grad_at_patch(potential)
        grad_at_patches = jnp.asarray([comps[0], comps[1]]).T
        sheet_thickness_on_patches = self.grid.map_mean_of_patch_nodes_to_patch(sheet_thickness)
        discharge = (
            -self.sheet_conductivity 
            * sheet_thickness_on_patches[:, None]**self.sheet_flow_exp 
            * grad_at_patches
        )
        normal_at_links = self.grid.get_normal_at_links()

        def discharge_dot_normal(link):
            head_xy = jnp.asarray([
                self.grid.node_x[self.grid.node_at_link_head[link]],
                self.grid.node_y[self.grid.node_at_link_head[link]]
            ])
            first_patch_xy = self.grid.xy_of_patch[self.grid.patches_at_link[link, 0]]
            second_patch_xy = self.grid.xy_of_patch[self.grid.patches_at_link[link, 1]]

            first_patch = jnp.where(
                jnp.dot(first_patch_xy - head_xy, normal_at_links[link, 0]) > 0,
                jnp.dot(discharge[self.grid.patches_at_link[link, 0]], normal_at_links[link, 0]),
                jnp.dot(discharge[self.grid.patches_at_link[link, 0]], normal_at_links[link, 1])
            )

            second_patch = jax.lax.cond(
                self.grid.patches_at_link[link, 1] != -1,
                lambda: jnp.where(
                    jnp.dot(second_patch_xy - head_xy, normal_at_links[link, 1]) > 0,
                    jnp.dot(discharge[self.grid.patches_at_link[link, 1]], normal_at_links[link, 1]),
                    jnp.dot(discharge[self.grid.patches_at_link[link, 1]], normal_at_links[link, 0])
                ),
                lambda: 0.0
            )

            return first_patch + second_patch
        
        return jax.vmap(discharge_dot_normal)(jnp.arange(self.grid.number_of_links))

    def update_sheet_flow(self, potential: jnp.array, sheet_thickness: jnp.array, dt: float) -> jnp.array:
        """Update the thickness of distributed sheet flow."""
        dhdt = lambda h: (
            self.calc_sheet_opening(h) - self.calc_sheet_closure(potential, h)
        )

        residual = lambda h, _: h - sheet_thickness - dt * dhdt(h)

        solver = optx.Newton(rtol = 1e-5, atol = 1e-5)
        solution = optx.root_find(residual, solver, sheet_thickness, args = None)

        return solution.value

    def update_channel_flow(
        self, 
        potential: jnp.array, 
        sheet_thickness: jnp.array,
        channel_size: jnp.array, 
        dt: float
    ) -> jnp.array:
        """Update the cross-sectional area of active channels."""
        dSdt = lambda S: (
            (
                self.energy_dissipation(potential, self.sheet_thickness, S)
                - self.sensible_heat(potential, self.sheet_thickness, S)
            ) / (self.state.ice_density * self.state.ice_latent_heat)
            - self.calc_channel_closure(potential, S)
        )

        residual = lambda S, _: S - channel_size - dt * dSdt(S)

        solver = optx.Newton(rtol = 1e-5, atol = 1e-5)
        solution = optx.root_find(residual, solver, channel_size, args = None)

        return jnp.where(
            solution.value < self.min_channel_size,
            self.min_channel_size,
            solution.value
        )








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
            channel_size_on_link = channel_size[link]
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

            sensible_heat = (
                -self.heat_capacity * self.pressure_melt_coeff * self.state.water_density
                * (
                    channel_discharge[link] 
                    + jnp.where(
                        (channel_size[link] > 0) | 
                        (sheet_discharge[link] * previous_gradients[link] > 0),
                        self.cavity_spacing * sheet_discharge[link],
                        0.0
                    )
                )
            ) * dissipation_coeff

            return jax.lax.cond(
                self.inflow_outflow[j] == 1,
                lambda _: out,
                lambda _: out.at[j].set(
                    (sheet_flux + channel_flux + channel_dissipation + sheet_dissipation - sensible_heat)
                ).at[i].add(
                    -(sheet_flux + channel_flux + channel_dissipation + sheet_dissipation - sensible_heat)
                ),
                0
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
                self.inflow_outflow[i] == -1,
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

        sheet_source = jnp.where(
            sheet_thickness < self.bedrock_step_height,
            self.grid.map_mean_of_links_to_node(
                jnp.abs(self.state.sliding_velocity / self.state.sec_per_a)
            )
            * (self.bedrock_step_height - sheet_thickness) 
            / self.cavity_spacing,
            0.0
        )

        sheet_closure = (
            self.closure_coeff
            * sheet_thickness
            * (self.base_potential - previous_potential)**self.n
        )

        interior_forcing = (
            channel_closure 
            + channel_source 
            + self.specific_melt_rate
            + sheet_closure
            - sheet_source
        )

        return jnp.where(
            self.inflow_outflow == -1,
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
        grav_potential = self.state.water_density * self.state.gravity * self.state.bedrock_elevation

        dSdt = lambda S: dt * (
            (
                jnp.abs(
                    -self.channel_conductivity * S**self.flow_exp * self.grid.calc_grad_at_link(potential)
                )
                + jnp.abs(
                    -self.sheet_conductivity 
                    * self.cavity_spacing
                    * self.grid.map_mean_of_link_nodes_to_link(sheet_thickness)**self.flow_exp 
                    * self.grid.calc_grad_at_link(potential)
                )
                - (
                    -self.heat_capacity 
                    * self.pressure_melt_coeff 
                    * self.state.water_density
                    * (
                        -self.channel_conductivity * S**self.flow_exp * self.grid.calc_grad_at_link(potential)
                        + jnp.where(
                            (S > 0) | 
                            (S * self.grid.calc_grad_at_link(potential - grav_potential) > 0),
                            -self.cavity_spacing * self.sheet_conductivity * S**self.flow_exp * self.grid.calc_grad_at_link(potential),
                            0.0
                        )
                    )
                )
            )
            / (self.state.ice_density * self.state.ice_latent_heat)
            - self.closure_coeff 
            * self.grid.map_mean_of_link_nodes_to_link(self.base_potential - potential)**self.n 
            * S
        )

        residual = lambda S, _: S - channel_size - dSdt(S)

        solver = optx.Newton(
            rtol = 1e-5, atol = 1e-5,
            linear_solver = lx.AutoLinearSolver(well_posed = None)
        )
        solution = optx.root_find(residual, solver, channel_size, args = None)

        channel_size = jnp.where(
            (self.inflow_outflow[self.grid.node_at_link_head] != 0) &
            (self.inflow_outflow[self.grid.node_at_link_tail] != 0),
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

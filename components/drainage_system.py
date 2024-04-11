"""Model subglacial hydrology in an evolving network of conduits."""

import jax
import jax.numpy as jnp
import equinox as eqx
import lineax as lx
import optimistix as optx
import optax
from functools import partial

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
    boundary_tags: jnp.array = eqx.field(converter = jnp.asarray, init = False)
    status_at_link: jnp.array = eqx.field(converter = jnp.asarray, init = False)

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

        # 1 if inflow, -1 if outflow, 0 if interior
        self.boundary_tags = self.set_inflow_outflow(self.base_potential)

        # 1 if connected to Neumann boundary node, 0 otherwise
        self.status_at_link = jnp.where(
            (self.boundary_tags[self.grid.node_at_link_head] == 1) |
            (self.boundary_tags[self.grid.node_at_link_tail] == 1),
            1,
            0
        )

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

    def calc_grad_at_link(self, array: jnp.array) -> jnp.array:
        """Calculate the gradient of an array at links, respecting Neumann boundaries."""
        grad = self.grid.calc_grad_at_link(array)
        return jnp.where(
            self.status_at_link == 0,
            grad,
            0.0
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
        min_adj_potential = jnp.min(
            jnp.where(
                self.grid.adjacent_nodes_at_node != -1,
                potential[self.grid.adjacent_nodes_at_node],
                jnp.inf
            ),
            axis = 1
        )

        # 1 denotes inflow, -1 denotes outflow
        inflow_outflow = jnp.where(
            potential <= min_adj_potential,
            -1 * (self.grid.status_at_node > 0),
            1 * (self.grid.status_at_node > 0)
        )

        return inflow_outflow

    def calc_water_pressure(self, potential: jnp.array) -> jnp.array:
        """Calculate the water pressure from potential."""
        return potential - self.base_potential

    def calc_effective_pressure(self, potential: jnp.array) -> jnp.array:
        """Calculate the effective pressure from potential."""
        return self.overburden_potential - potential

    def sheet_discharge_coeff(self, potential: jnp.array, sheet_thickness: jnp.array) -> jnp.array:
        """Calculate the discharge coefficient for distributed sheet flow."""
        gradient = self.calc_grad_at_link(potential)
        sheet_thickness_on_links = self.grid.map_mean_of_link_nodes_to_link(sheet_thickness)

        coeff = (
            -self.sheet_conductivity 
            * sheet_thickness_on_links**self.sheet_flow_exp 
            * jnp.abs(gradient)**(-1/2)
        )

        return jnp.where(
            gradient == 0,
            0.0,
            coeff
        )

    def sheet_discharge_on_links(self, potential: jnp.array, sheet_thickness: jnp.array) -> jnp.array:
        """Interpolate discharge from the distributed system onto grid links."""
        return (
            self.sheet_discharge_coeff(potential, sheet_thickness)
            * self.calc_grad_at_link(potential)
        )

    def channel_discharge_coeff(self, potential: jnp.array, channel_size: jnp.array) -> jnp.array:
        """Calculate the discharge coefficient for active channels."""
        gradient = self.calc_grad_at_link(potential)
        coeff = (
            -self.channel_conductivity 
            * channel_size**self.channel_flow_exp 
            # * jnp.abs(gradient)**(-1/2)
        )

        return jnp.where(
            gradient == 0,
            0.0,
            coeff
        )

    def channel_discharge(self, potential: jnp.array, channel_size: jnp.array) -> jnp.array:
        """Calculate the discharge in active channels."""
        return (
            self.channel_discharge_coeff(potential, channel_size) * self.calc_grad_at_link(potential)
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
        sheet_discharge_on_links = self.sheet_discharge_on_links(potential, sheet_thickness)
        channel_discharge = self.channel_discharge(potential, channel_size)
        gradient = self.calc_grad_at_link(potential)

        return (
            jnp.abs(self.cavity_spacing * sheet_discharge_on_links * gradient)
            + jnp.abs(channel_discharge * gradient)
        )

    def sensible_heat_coeff(
        self,
        potential: jnp.array,
        sheet_thickness: jnp.array,
        channel_size: jnp.array
    ) -> jnp.array:
        """Calculate the coefficient of sensible heat change in the channelized system."""
        heat_coeff = (
            -self.pressure_melt_coeff * self.heat_capacity * self.state.water_density
        )
        sheet_discharge = self.sheet_discharge_on_links(potential, sheet_thickness)
        channel_discharge = self.channel_discharge(potential, channel_size)
        water_pressure = self.calc_water_pressure(potential)
        pressure_gradient = self.calc_grad_at_link(water_pressure)

        total_discharge = jnp.where(
            (channel_size > 0) | ((pressure_gradient * sheet_discharge) > 0),
            channel_discharge + self.cavity_spacing,
            channel_discharge
        )

        return heat_coeff * total_discharge * pressure_gradient

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
        pressure_gradient = self.calc_grad_at_link(water_pressure)

        total_discharge = jnp.where(
            (channel_size > 0) | ((pressure_gradient * sheet_discharge) > 0),
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
        
        exchange = jax.vmap(discharge_dot_normal)(jnp.arange(self.grid.number_of_links))

        return jnp.where(
            self.status_at_link == 0,
            exchange,
            0.0
        )

    def update_sheet_flow(self, potential: jnp.array, sheet_thickness: jnp.array, dt: float) -> jnp.array:
        """Update the thickness of distributed sheet flow."""
        dhdt = lambda h: (
            self.calc_sheet_opening(h) - self.calc_sheet_closure(potential, h)
        )

        residual = lambda h, _: h - sheet_thickness - dt * dhdt(h)

        solver = optx.Newton(rtol = 1e-5, atol = 1e-5)
        solution = optx.root_find(residual, solver, sheet_thickness, args = None)

        return jnp.where(
            self.boundary_tags == 0,
            solution.value,
            0.0
        )

    def update_channel_flow(
        self, 
        potential: jnp.array, 
        sheet_thickness: jnp.array,
        channel_size: jnp.array, 
        dt: float
    ) -> jnp.array:
        """Update the cross-sectional area of active channels."""
        dSdt = lambda S: jnp.where(
            self.status_at_link == 0,
            (
                self.energy_dissipation(potential, sheet_thickness, S)
                - self.sensible_heat(potential, sheet_thickness, S)
            ) / (self.state.ice_density * self.state.ice_latent_heat)
            - self.calc_channel_closure(potential, S),
            0.0
        )

        residual = lambda S, _: S - channel_size - dt * dSdt(S)

        solver = optx.Newton(rtol = 1e-5, atol = 1e-5)
        solution = optx.root_find(residual, solver, channel_size, args = None)

        return solution.value

    def build_forcing_vector(
        self,
        potential: jnp.array,
        sheet_thickness: jnp.array,
        channel_size: jnp.array
    ) -> jnp.array:
        """Calculate the forcing vector for potential at grid cells."""
        sheet_opening = self.calc_sheet_opening(sheet_thickness)
        sheet_closure = self.calc_sheet_closure(potential, sheet_thickness)

        channel_closure = jnp.abs(
            self.grid.sum_at_nodes(
                self.calc_channel_closure(potential, channel_size)
            )
        )

        heat_coeff = (
            (1 / self.state.ice_density) - (1 / self.state.water_density)
        ) * (1 / self.state.ice_latent_heat)
        sensible_heat = self.grid.map_mean_of_links_to_node(
            self.sensible_heat(potential, sheet_thickness, channel_size)
        )
        dissipation = self.grid.map_mean_of_links_to_node(
            self.energy_dissipation(
                potential, sheet_thickness, channel_size
            )
        )
        channel_opening = jnp.abs(
            self.grid.sum_at_nodes(
                (dissipation - sensible_heat) * heat_coeff
            )
        )
        
        forcing_at_nodes = (
            (self.melt_input - sheet_opening + sheet_closure) 
            * self.grid.cell_area_at_node
            + channel_closure
            # Not a typo, channel opening is already formulated as a negative term
            + channel_opening
        )

        return forcing_at_nodes[self.grid.node_at_cell]

    def get_coeffs_at_link(
        self, cell: int, j: int,
        potential: jnp.array,
        sheet_thickness: jnp.array,
        channel_size: jnp.array
    ) -> float:
        """Get the coefficients of the linear system at the link between a cell and neighboring node 'j'."""
        i = self.grid.node_at_cell[cell]
        link = self.links_between_nodes[i, j]
        link_len = self.grid.length_of_link[link]
        face_len = self.grid.length_of_face[self.grid.face_at_link[link]]

        gradient = self.calc_grad_at_link(potential)[link]

        sheet_flux = (
            self.sheet_discharge_coeff(potential, sheet_thickness)[link]
            * face_len
            / link_len
        )

        channel_flux = (
            self.channel_discharge_coeff(potential, channel_size)[link]
            * face_len
            / link_len
        )        

        return jnp.where(
            self.status_at_link[link] == 0,
            sheet_flux + channel_flux,
            0.0
        )

    def assemble_row(
        self, 
        cell: int, 
        potential: jnp.array, 
        sheet_thickness: jnp.array, 
        channel_size: jnp.array
    ) -> tuple:
        """Assemble the matrix entries and any addition to the forcing vector at one cell."""
        i = self.grid.node_at_cell[cell]
        adj_nodes = self.grid.adjacent_nodes_at_node[i]

        get_coeffs = jax.vmap(
            lambda j: self.get_coeffs_at_link(cell, j, potential, sheet_thickness, channel_size)
        )

        coeffs = jnp.where(
            adj_nodes != -1,
            get_coeffs(adj_nodes),
            0.0
        )

        added_forcings = jnp.where(
            (self.boundary_tags[adj_nodes] == -1) & (adj_nodes != -1),
            -coeffs * self.base_potential[adj_nodes],
            0.0
        )

        forcing = jnp.sum(added_forcings)
        
        empty_row = jnp.zeros(self.grid.number_of_cells)

        def assign_coeffs(row, jidx):
            j = adj_nodes[jidx]
            jcell = self.grid.cell_at_node[j]

            # If jcell != -1, it is an interior cell
            # If jcell == -1, it is a boundary cell
            # but recall that the coefficients at Neumann boundaries are always zero
            row = jnp.where(
                jcell != -1,
                row.at[jcell].add(coeffs[jidx]).at[cell].add(-coeffs[jidx]),
                row.at[cell].add(-coeffs[jidx])
            )
            return row, None

        row, _ = jax.lax.scan(
            assign_coeffs,
            empty_row,
            jnp.arange(len(adj_nodes))
        )
    
        return (forcing, row)

    def assemble_linear_system(
        self,
        previous_potential: jnp.array,
        sheet_thickness: jnp.array,
        channel_size: jnp.array
    ) -> tuple:
        """Assemble the linear system for potential. Return the matrix 'A' and forcing 'b'."""

        forcing, matrix = jax.vmap(self.assemble_row, in_axes = (0, None, None, None))(
            jnp.arange(self.grid.number_of_cells),
            previous_potential,
            sheet_thickness,
            channel_size
        )

        base_forcing = self.build_forcing_vector(previous_potential, sheet_thickness, channel_size)
        forcing = jnp.add(forcing, base_forcing)

        return forcing, matrix

    def solve_for_potential(
        self,
        previous_potential: jnp.array,
        sheet_thickness: jnp.array,
        channel_size: jnp.array
    ) -> jnp.array:
        """Solve the elliptic equation for potential."""
        b, A = self.assemble_linear_system(previous_potential, sheet_thickness, channel_size)
        operator = lx.MatrixLinearOperator(A)
        solution = lx.linear_solve(operator, b)

        boundary_values = jnp.where(
            self.boundary_tags == 1,
            0.0,
            self.base_potential
        )

        return jnp.where(
            self.grid.cell_at_node != -1,
            solution.value[self.grid.cell_at_node],
            boundary_values
        )

    # def optimize_potential(
    #     self,
    #     potential: jnp.array,
    #     sheet_thickness: jnp.array,
    #     channel_size: jnp.array
    # ) -> jnp.array:
    #     """Solve for potential via nonlinear optimization."""
    #     def mass_residual(potential, args):
    #         sheet_thickness, channel_size = args
    #         potential = jnp.where(
    #             self.boundary_tags == -1,
    #             self.base_potential,
    #             potential
    #         )
    #         channel_discharge = self.channel_discharge(potential, channel_size)
    #         return self.grid.sum_at_nodes(channel_discharge * self.set_flow_direction(potential))

    #     solution = optx.least_squares(
    #         lambda phi, args: mass_residual(phi, args),
    #         solver = optx.OptaxMinimiser(optax.adabelief(1e-3), rtol = 1e-8, atol = 1e-8),
    #         y0 = self.grid.node_x,
    #         args = (sheet_thickness, channel_size)
    #     )

    #     return solution.value

    @jax.jit
    def update(self, dt: float):
        """Advance the model by one step."""
        potential = self.solve_for_potential(
            self.potential, self.sheet_thickness, self.channel_size
        )

        sheet_thickness = self.update_sheet_flow(
            potential, self.sheet_thickness, dt
        )

        channel_size = self.update_channel_flow(
            potential, sheet_thickness, self.channel_size, dt
        )

        updated = eqx.tree_at(
            lambda t: (t.potential, t.sheet_thickness, t.channel_size),
            self,
            (potential, sheet_thickness, channel_size)
        )

        return updated

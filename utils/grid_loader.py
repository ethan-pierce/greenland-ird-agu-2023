"""Generate a TriangleMeshGrid, add netCDF data, and pickle it."""

import matplotlib.pyplot as plt

import os
import copy
import numpy as np
import xarray as xr
import rioxarray as rxr
import geopandas as gpd
import shapely
import itertools
from scipy.interpolate import RBFInterpolator
from rasterio.enums import Resampling
from landlab import TriangleMeshGrid


class GridLoader:
    """Constructs a Landlab grid and adds gridded data to it."""

    def __init__(
        self, 
        shapefile: str, 
        quality: int = 20, 
        max_area: float = 1000000, 
        buffer: float = 0.0, 
        tolerance: float = 0.0,
        centered: bool = True,
        generate_grid: bool = True
    ):
        """Initialize a new GridLoader object."""
        quality_flag = "q" + str(quality)
        area_flag = "a" + str(max_area)
        self.triangle_opts = "pDevjz" + quality_flag + area_flag
        self.centered = centered
        
        self.geoseries = gpd.read_file(shapefile)
        self.original_geoseries = self.geoseries.copy()
        self.crs = str(self.geoseries.crs)

        # These will be overwritten when _generate_grid() is called
        boundary = self._smooth_boundary(self.geoseries.geometry, buffer = buffer)[0]
        self.polygon = shapely.simplify(boundary, tolerance)
        self.original_polygon = copy.deepcopy(self.polygon)

        if generate_grid:
            self._generate_grid(buffer, tolerance)

    def _generate_grid(self, buffer: float, tolerance: float):
        """Generate a mesh and optionally center it at the origin."""
        if self.centered:
            self.xoff = -self.geoseries.centroid.x[0]
            self.yoff = -self.geoseries.centroid.y[0]

            self.geoseries = self.geoseries.translate(
                xoff = self.xoff,
                yoff = self.yoff
            )

        boundary = self._smooth_boundary(self.geoseries.geometry, buffer = buffer)[0]
        self.polygon = shapely.simplify(boundary, tolerance)

        nodes_y = np.array(self.polygon.exterior.xy[1])
        nodes_x = np.array(self.polygon.exterior.xy[0])
        holes = self.polygon.interiors

        self.grid = TriangleMeshGrid(
            (nodes_y, nodes_x), holes = holes, triangle_opts = self.triangle_opts
        )

    def _smooth_boundary(self, polygon, buffer: float) -> shapely.Polygon:
        """Smooth the exterior boundary of the input shapefile."""

        # Dilate by x, erode by 2x, dilate again by x
        new_poly = (
            polygon.buffer(buffer, join_style='round')
            .buffer(-2 * buffer, join_style='round')
            .buffer(buffer, join_style='round')
        )

        return new_poly

    def _open_data(self, path: str, var: str, crs=None, no_data=None) -> xr.DataArray:
        """Read a netCDF file as an xarray Dataset."""
        ds = xr.open_dataset(path)
        da = ds.data_vars[var]

        if crs:
            da.rio.write_crs(crs, inplace=True)

        if no_data:
            da = da.where(da != no_data)
            da.rio.write_nodata(np.nan, inplace=True)

        return da

    def _clip(self, source: xr.DataArray) -> xr.DataArray:
        """Clip data to the shapefile bounds."""
        return source.rio.clip(
            geometries=[self.original_polygon], crs=self.crs, drop=True
        )

    def _clip_to_box(self, source: xr.DataArray) -> xr.DataArray:
        """Clip data to a bounding box around the shapefile."""
        return source.rio.clip_box(
            minx = self.original_geoseries.get_coordinates().x.min(),
            miny = self.original_geoseries.get_coordinates().y.min(),
            maxx = self.original_geoseries.get_coordinates().x.max(),
            maxy = self.original_geoseries.get_coordinates().y.max()
        )

    def _translate(self, source: xr.DataArray) -> xr.DataArray:
        """Translate a DataArray to center at the origin."""
        return source.assign_coords(
            {'x': source.coords['x'] + self.xoff,
             'y': source.coords['y'] + self.yoff}
        )

    def _reproject(self, source: xr.DataArray, resolution: tuple | float, dest: str = "") -> xr.DataArray:
        """Reproject data from source crs to destination crs."""
        if len(dest) == 0:
            dest = self.crs

        return source.rio.reproject(
            dst_crs = dest, 
            resolution = resolution, 
            resampling = Resampling.bilinear,
            nodata = np.nan
        )

    def _interpolate_na(
        self, source: xr.DataArray, method: str = "nearest"
    ) -> xr.DataArray:
        """Interpolate missing data using scipy.interpolate.griddata."""
        return source.rio.interpolate_na(method=method)

    def _rescale(self, source: xr.DataArray, scalar: float) -> xr.DataArray:
        """Multiply a dataarray by a scalar."""
        return source * scalar

    def _interpolate_to_mesh(
        self, source: xr.DataArray, neighbors: int = 9, smoothing: float = 0.0
    ) -> np.ndarray:
        """Interpolate a dataarray to the new grid coordinates."""
        stack = source.stack(z=("x", "y"))
        coords = np.vstack([stack.coords["x"], stack.coords["y"]]).T
        values = source.values.flatten(order="F")

        destination = np.vstack([self.grid.node_x, self.grid.node_y]).T

        interp = RBFInterpolator(
            coords, values, neighbors=neighbors, smoothing=smoothing
        )
        result = interp(destination)

        return result

    def _interpolate_to_grid(
        self, source: xr.DataArray, destination: xr.DataArray
    ):
        """Interpolate one dataarray to match the shape of a second dataarray."""
        return source.interp(
            {'x': destination.coords['x'],
             'y': destination.coords['y']}
        )

    def add_field(
        self,
        path: str,
        nc_name: str,
        ll_name: str,
        resolution: float,
        crs=None,
        no_data=None,
        neighbors=9,
        smoothing=0.0,
        scalar=1.0
    ):
        """Read a field from a netCDF file and add it to the grid."""
        if len(ll_name) == 0:
            ll_name = nc_name

        opened = self._open_data(path, nc_name, crs=crs, no_data=no_data)
        clipped = self._clip(opened)

        if self.centered:
            translated = self._translate(clipped)
        else:
            translated = clipped

        projected = self._reproject(translated, resolution)
        filled = self._interpolate_na(projected)
        rescaled = self._rescale(filled, scalar)
        gridded = self._interpolate_to_mesh(rescaled, neighbors=neighbors, smoothing=smoothing)

        self.grid.add_field(ll_name, gridded, at="node")

    def write_input_nc(
        self, 
        path_to_write: str,
        data_vars: list,
        nc_files: list,
        input_names: list,
        crs: list,
        no_data: list,
        scalars: list,
        resolution: tuple,
        add_igm_aux_vars: True,
        write_output = False,
        yield_output = False
    ):
        data_arrays = []

        for i in range(len(data_vars)):
            opened = self._open_data(nc_files[i], input_names[i], crs = crs[i], no_data = no_data[i])
            clipped = self._clip_to_box(opened)

            if self.centered:
                translated = self._translate(clipped)
            else:
                translated = clipped

            projected = self._reproject(translated, resolution = resolution)
            filled = self._interpolate_na(projected)
            rescaled = self._rescale(filled, scalars[i])

            if i > 0:
                gridded = self._interpolate_to_grid(rescaled, data_arrays[0])
            else:
                gridded = rescaled
            
            assert gridded.rio.resolution()[0] == np.abs(gridded.rio.resolution()[1])

            data_arrays.append(gridded)

        if add_igm_aux_vars:
            thkidx = data_vars.index('thk')
            icemask = np.where(
                data_arrays[thkidx].values[:] > 0.1,
                1.0,
                0.0
            )
            data_vars.append('icemask')
            data_arrays.append((data_arrays[thkidx].dims, icemask))

            icemaskobs = icemask
            data_vars.append('icemaskobs')
            data_arrays.append((data_arrays[thkidx].dims, icemaskobs))

            data_vars.append('thkobs')
            data_arrays.append(data_arrays[thkidx])

            data_vars.append('thkinit')
            data_arrays.append(data_arrays[thkidx])

            usurfidx = data_vars.index('usurf')
            data_vars.append('usurfobs')
            data_arrays.append(data_arrays[usurfidx])

            uvelsurfidx = data_vars.index('uvelsurf')
            data_vars.append('uvelsurfobs')
            data_arrays.append(data_arrays[uvelsurfidx])

            vvelsurfidx = data_vars.index('vvelsurf')
            data_vars.append('vvelsurfobs')
            data_arrays.append(data_arrays[vvelsurfidx])

        dataset = xr.Dataset(
            data_vars = {data_vars[i]: data_arrays[i] for i in range(len(data_vars))},
            coords = data_arrays[0].coords
        )

        if write_output:
            dataset.to_netcdf(path_to_write)

        if yield_output:
            return dataset, data_arrays

def calc_resolution(polygon, n_cells):
    return int(np.sqrt(polygon.area / n_cells))

def main():
    """Generate a mesh and add netCDF data."""
    from utils.plotting import plot_triangle_mesh
    import warnings
    warnings.filterwarnings("ignore")

    bedmachine = "/home/egp/repos/greenland-ird/data/ignore/BedMachineGreenland-v5.nc"
    velocity = "/home/egp/repos/greenland-ird/data/ignore/GRE_G0120_0000.nc"
    geotherm = "/home/egp/repos/greenland-ird/data/ignore/geothermal_heat_flow_map_10km.nc"
    shapefiles = "/home/egp/repos/greenland-ird/data/basin-outlines/"
    paths = []
    for i in os.listdir('/home/egp/repos/greenland-ird/data/basin-outlines/CE/'):
        paths.append('CE/' + i)
    for i in os.listdir('/home/egp/repos/greenland-ird/data/basin-outlines/CW/'):
        paths.append('CW/' + i)
    for i in os.listdir('/home/egp/repos/greenland-ird/data/basin-outlines/SW/'):
        paths.append('SW/' + i)

    for path in paths:
        glacier = path.split('/')[-1].replace('.geojson', '')
        print('Constructing mesh for ', glacier)

        if glacier in ['sydbrae', 'charcot-gletscher', 'graah-gletscher', 'dode-brae']:
            max_area = 1e5
        elif glacier == 'bredegletsjer':
            print('Testing bredegletsjer')
            max_area = 1e4
        else:
            max_area = 1e6

        loader = GridLoader(
            shapefiles + path, 
            centered = False, 
            generate_grid = True, 
            max_area = max_area, 
            quality = 30,
            buffer = 250.0,
            tolerance = 10.0
        )

        if glacier in ['eielson-gletsjer', 'sermeq-avannarleq', 'sermeq-kullajeq']:
            resolution = calc_resolution(loader.polygon, n_cells = 30000)
        else:
            resolution = calc_resolution(loader.polygon, n_cells = 40000)

        loader.add_field(
            bedmachine,
            "thickness",
            "ice_thickness",
            resolution,
            crs = "epsg:3413",
            no_data = -9999.0,
            neighbors = 9,
            smoothing = 0.0
        )
        h = loader.grid.at_node['ice_thickness']
        h[h < 0.5] = 0.0
        print('Added ice thickness to grid nodes.')

        loader.add_field(
            bedmachine,
            "bed",
            "bedrock_elevation",
            resolution,
            crs = "epsg:3413",
            neighbors = 9,
            no_data = -9999.0,
        )
        print("Added bedrock elevation to grid nodes.")

        loader.grid.add_field(
            'surface_elevation', 
            loader.grid.at_node['ice_thickness'][:] + loader.grid.at_node['bedrock_elevation'][:],
            at = 'node'
        )

        loader.add_field(
            geotherm, 
            "GHF", 
            "geothermal_heat_flux", 
            resolution,
            crs = "epsg:3413", 
            neighbors = 100, 
            no_data = np.nan, 
            scalar = 1e-3 * 31556926
        )
        print("Added geothermal heat flux to grid nodes.")

        loader.add_field(
            velocity, 
            "vx", 
            "surface_velocity_x", 
            resolution,
            crs = "epsg:3413", 
            neighbors = 100, 
            no_data = -1
        )
        loader.add_field(
            velocity, 
            "vy", 
            "surface_velocity_y", 
            resolution,
            crs = "epsg:3413", 
            neighbors = 100, 
            no_data = -1
        )
        loader.grid.add_field(
            'surface_velocity_vector', 
            (loader.grid.at_node['surface_velocity_x'][:], loader.grid.at_node['surface_velocity_y'][:]),
            at = 'node'
        )
        print("Added surface velocity to grid nodes.")

        print('Mesh nodes: ', loader.grid.number_of_nodes)
        print('Mesh links: ', loader.grid.number_of_links)
        loader.grid.save('/home/egp/repos/glacierbento/examples/ird/meshes/' + glacier + '.grid', clobber = True)

        # QC
        # plot_triangle_mesh(loader.grid, loader.grid.at_node['ice_thickness'][:], subplots_args = {'figsize': (18, 6)})
        # plot_triangle_mesh(loader.grid, loader.grid.at_node['surface_velocity_x'][:], subplots_args = {'figsize': (18, 6)})

        print('Finished loading data for ' + glacier.replace('-', ' ').capitalize())
    
if __name__ == "__main__":
    main()

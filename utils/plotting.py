"""Utilities for plotting fields on unstructured meshes."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.patches
import matplotlib.collections

import time

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        wrapper.total_time += duration
        print(f"Execution time: {duration}")
        return result

    wrapper.total_time = 0
    return wrapper

def plot_triangle_mesh(
    grid, 
    field, 
    at = 'patch',
    cmap = plt.cm.jet, 
    subplots_args = None,
    set_clim = False,
    norm = None,
    cbar_ticks = None,
    show = True,
    title = None
):
    """Plot a field defined on an unstructured mesh."""
    if isinstance(field, str):
        if field in grid.at_node.keys():
            field = grid.at_node[field][:]
        elif field in grid.at_link.keys():
            field = grid.map_mean_of_links_to_node(field)
        else:
            raise ValueError(
                "Could not find " + field + " at grid nodes or links."
            )

    if hasattr(field, 'shape'):
        if len(np.ravel(field)) == grid.number_of_nodes:
            pass
        elif len(np.ravel(field)) == grid.number_of_links:
            field = grid.map_mean_of_links_to_node(field)
        else:
            raise ValueError(
                "Could not broadcast " + field + " to grid nodes or links."
            )

    if at == 'patch':
        values = grid.map_mean_of_patch_nodes_to_patch(field)

        coords = []
        for patch in range(grid.number_of_patches):
            nodes = []

            for node in grid.nodes_at_patch[patch]:
                nodes.append(
                    [grid.node_x[node], grid.node_y[node]]
                )

            coords.append(nodes)

    elif at == 'cell':
        values = grid.map_node_to_cell(field)

        coords = []
        for cell in range(grid.number_of_cells):
            corners = []

            for corner in grid.corners_at_cell[cell]:
                if corner != -1:
                    corners.append(
                        [grid.x_of_corner[corner], grid.y_of_corner[corner]]
                    )

            coords.append(corners)

    else:
        raise NotImplementedError(
            "For now, plot_triangle_mesh can only plot fields at patches or cells."
        )

    if subplots_args is None:
            subplots_args = {'nrows': 1, 'ncols': 1}

    fig, ax = plt.subplots(**subplots_args)

    import shapely
    hulls = [shapely.get_coordinates(shapely.Polygon(i).convex_hull) for i in coords]
    polys = [plt.Polygon(shp) for shp in hulls]

    if norm is None:
        norm = matplotlib.colors.Normalize(vmin = np.min(field), vmax = np.max(field))

    collection = matplotlib.collections.PatchCollection(polys, cmap=cmap, norm=norm)
    collection.set_array(values)

    if set_clim is not False:
        collection.set_clim(**set_clim)

    im = ax.add_collection(collection)
    ax.autoscale()

    if cbar_ticks is not None:
        plt.colorbar(im, ticks = cbar_ticks)
    else:
        plt.colorbar(im)

    if title is not None:
        plt.title(title)

    if show:
        plt.show()
    
    return fig, ax

def plot_links(
    grid,
    field,
    cmap = plt.cm.jet,
    subplots_args = None,
    show = True,
    title = None
):
    """Plot a field defined on grid links."""
    lines = []

    for link in np.arange(grid.number_of_links):
        head = grid.node_at_link_head[link]
        tail = grid.node_at_link_tail[link]

        xs = (grid.node_x[head], grid.node_y[head])
        ys = (grid.node_x[tail], grid.node_y[tail])

        lines.append([xs, ys])
        
    collection = matplotlib.collections.LineCollection(lines, cmap = cmap)
    collection.set_array(field)

    if subplots_args is None:
        subplots_args = {'nrows': 1, 'ncols': 1}

    fig, ax = plt.subplots(**subplots_args)

    im = ax.add_collection(collection)
    ax.autoscale()

    plt.colorbar(im)

    if title is not None:
        plt.title(title)

    if show:
        plt.show()
    
    return fig, ax
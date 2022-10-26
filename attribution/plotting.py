import cartopy.crs as ccrs
import iris.plot as iplt
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
from iris.cube import Cube
from matplotlib import pyplot as plt

from attribution.utils import compute_cube_regression


def plot_regression_map(
    index_cube,
    index_name,
    predictor=None,
    reg_coefs=None,
    pvalues=None,
    p_lim=None,
    epsg=None,
    ax=None,
    coord_names={"lat": "grid_latitude", "lon": "grid_longitude"},
):
    """Plot a regression map of the regression coefficents between cube variable and the predictor.

    Arguments
    ---------
    cube : iris.cube.Cube
        iris cube holding the data on which to perform the regression.
    predictor : np.ndarray, optional
        1d array with the predictor data.
    reg_coefs : np.ndarray, optional
        2d array of regression coefficients. Using this the regression does not have to be re-run.
    pvalues : np.ndarray, optional
        2d array with pvalues for the regression coefficients.
    epsg : string
        EPSG code for custom plot projection.
    ax : matplotlib.axes.Axes
        Draw the plot on custom axes.
    coord_names : dict, optional
        Name of spatial coordinates in the cube.
    """

    if reg_coefs is None:
        if predictor is None:
            raise ValueError("predictor required if reg_coefs is None")

        # Compute the regression.
        reg_coefs, pvalues = compute_cube_regression(
            index_cube, predictor, broadcast_coef=False
        )
    # We want to compute the regression over every gridpoint
    # For ease of plotting we create a simple iris cube of the regression.
    reg_cube = Cube(
        reg_coefs,
        dim_coords_and_dims=[
            (index_cube.coord(coord_names["lat"]), 0),
            (index_cube.coord(coord_names["lon"]), 1),
        ],
    )
    # Was an axes passed?
    if ax is None:
        # Custom projection?
        if epsg is None:
            projection = ccrs.PlateCarree()
        # Default to PlateCarree
        else:
            projection = ccrs.epsg(epsg)
        # Create the axes.
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": projection})
    # Get the colormap.
    cmap = cm.get_cmap("brewer_RdBu_11")
    # Plot the cube. The CenteredNorm maps data to [0, 1] for the colormap,
    # where 0 of original data is mapped to 0.5, since we have a diverging cmap.
    iplt.contourf(reg_cube, norm=colors.CenteredNorm(), cmap=cmap, axes=ax)
    # Scatter confidence
    if p_lim is not None:
        # Get indices of significant values.
        y, x = np.argwhere(pvalues <= p_lim).T
        # Need their cooridnate values.
        y = index_cube.coord(coord_names["lat"]).points[..., y]
        x = index_cube.coord(coord_names["lon"]).points[..., x]
        # Scatter them
        coord_system = index_cube.coord_system().as_cartopy_projection()
        ax.scatter(
            x,
            y,
            marker="+",
            c="k",
            alpha=0.3,
            label=f"p$\leq${p_lim}",
            transform=coord_system,
        )
    # Title
    t0 = index_cube.coord("time").cell(0).point.strftime("%Y")
    t1 = index_cube.coord("time").cell(-1).point.strftime("%Y")
    ax.set_title(f"GMST to {index_name} regression coef. map\n{t0}-{t1}")
    # Gridlines
    gl = ax.gridlines(draw_labels=True)
    # Remove ylabels on right side.
    gl.right_labels = False
    try:
        fig.tight_layout()
    except NameError:
        pass

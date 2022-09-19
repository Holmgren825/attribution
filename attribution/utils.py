import geopandas as gpd
import iris
import iris.analysis
import iris.analysis.cartography
import iris.coord_categorisation
import iris_utils
import numpy as np
import pandas as pd
import statsmodels.api as sm
from climix.metadata import load_metadata
from iris.exceptions import CoordinateNotFoundError
from tqdm.autonotebook import trange

from attribution.config import init_config


def select_season(cube, season_abbr, season_name="season"):
    """Select data from a cube for a specific seasonself.

    Arguments:
    ----------
    cube : iris.Cube.cube
        Cube hodling the data.
    season_abbr : string
        A string of month abbreviations specifying the season of interest.
        For instance "mjja", "djf".
    season_name : string
        Name of the custom season.

    Returns:
    --------
    season_cube
        Cube with the data selected.
    """
    # First we need a monthly coordinate.
    try:
        iris.coord_categorisation.add_month(cube, "time")
    # If there already is a month coordinate, we pass.
    except ValueError:
        print("Cooridnate month alredy exist.")
    # Then we add the membership
    try:
        iris.coord_categorisation.add_season_membership(
            cube, "time", name=season_name, season=season_abbr
        )
    except ValueError:
        print("Season membership already exist.")

    # Create the constriant
    season_constraint = iris.Constraint(coord_values={season_name: True})

    season_cube = cube.extract(season_constraint)

    return season_cube


def compute_index(cube, index_name, client, spatial_average=False, spatial_max=False):
    """Compute a climate index based on the cube using Climix.

    Arguments:
    ----------
    cube : iris.Cube.cube

    index_name : string
        CF name of the index to compute.
    client : dask.distributed.Client
    spatial_average : bool, default: False
        Whether to return a spatially averaged cube or not.
    spatial_max : bool, default: False
        Whether to return the spatial max for each year or not.

    Returns:
    --------
    index_cube

    """
    # Prepare the catalog
    index_catalog = load_metadata()

    # Select the index
    index = index_catalog.prepare_indices([index_name])[0]

    # Prepare the cube.
    # Can't have a "year" coordinate in climix
    try:
        cube.remove_coord("year")
    # If there is none, do nothing.
    except CoordinateNotFoundError:
        pass

    # Compute the index.
    index_cube = index([cube], client)

    # Do we want to compute the spatial average
    if spatial_average:
        index_cube = compute_spatial_average(index_cube)
    # If not, do we want to compute the spatial max?
    elif spatial_max:
        index_cube = index_cube.collapsed(
            ["grid_latitude", "grid_longitude"], iris.analysis.MAX
        )

    return index_cube


def compute_spatial_average(cube):
    """Compute the spatial average of a cube.

    Arguments:
    ----------
    cube : iris.cube.Cube

    Returns:
    --------
    Spatially averaged cube.
    """

    # Get area weights.
    area_weights = iris_utils.utils.get_weights(cube)

    # Collapse the dimensions.
    averaged_cube = cube.collapsed(
        ["grid_longitude", "grid_latitude"], iris.analysis.MEAN, weights=area_weights
    )

    return averaged_cube


def get_country_shape(shapefile=None, country="Sweden"):
    """Returns a polygon of Sweden mainland which can be used with iris_utils.mask_from_shape.

    Arguments
    ---------
    shapefile : string, optional
        Path to shapefile. Assumes it to be this file
        https://www.naturalearthdata.com/downloads/10m-cultural-vectors/ Admin 0 - Countries.
    country : string
        Name of the country to select.
    """
    if not shapefile:
        # Get the CFG.
        CFG = init_config()
        shapefile = CFG["paths"]["shapefile"]
    # Load it.
    gdf = gpd.read_file(shapefile)
    # Sweden has multiple shapes
    swe_shapes = gdf[gdf.SOVEREIGNT == country].geometry

    # Return the first geometry.
    # TODO This might cause issues with other countries.
    return swe_shapes.iloc[0].geoms[0]


def compute_cube_regression(cube, predictor, broadcast_coef=True):
    """Compute the regression coefficient between the data in cube and a predictor.

    Arguments
    ---------
    cube : iris.cube.Cube
        Cube containing the data on which to perform the regression.
    predictor : array_like
        Data to use a predictor in the linear regression.
    broadcast_coef : bool, default: True
        Return the compressed regression coefficients, broadcasted so that there
        is one value per gridpoint and year.

    Returns
    -------
    The regression coefficient.
    """
    # Make sure data is not lazy
    data = cube.data
    # Shape
    lat_shape = cube.coord("grid_latitude").shape[0]
    lon_shape = cube.coord("grid_longitude").shape[0]
    # Store the results
    coefs = np.zeros((lat_shape, lon_shape))
    pvalues = np.zeros((lat_shape, lon_shape))
    # We use statsmodels
    X = sm.add_constant(predictor)
    # Loop over all gridpoints. Mask later.
    for lat in trange(lat_shape):
        for lon in range(lon_shape):
            # Get the result
            res = sm.OLS(data[:, lat, lon], X).fit()
            coefs[lat, lon] = res.params[-1]
            pvalues[lat, lon] = res.pvalues[-1]

    # Mask the result arrays. The cube should hold the dance.
    coefs = np.ma.masked_array(coefs, cube.data.mask[0, :, :])
    pvalues = np.ma.masked_array(pvalues, cube.data.mask[0, :, :])

    # This returns the compressed and broadcasted array.
    if broadcast_coef:
        coefs = coefs.compressed()
        # Broadcast so that there is one coef for each year.
        coefs = np.broadcast_to(coefs, (predictor.shape[0], coefs.shape[0]))
        return coefs
    # If not, we return the masked arrays.
    else:
        return coefs, pvalues


def get_gmst(cube, path=None, window=4):
    """Get the gmst timeseries for the corresponding cube.

    Arguments
    ---------
    cube : iris.Cube
        Used to get the timespan.
    path : string, Optional.
        Path to local gistemp data.
    window : int
        Size of smoothing window.

    Returns
    -------
    gmst_data :
    """
    url = "https://data.giss.nasa.gov/gistemp/graphs/graph_data/Global_Mean_Estimates_based_on_Land_and_Ocean_Data/graph.txt"
    if not path:
        df = pd.read_csv(
            # Load in the dataset.
            url,
            sep=r"\s+",
            header=2,
        )
    else:
        df = pd.read_csv(
            # Load in the dataset.
            path,
            sep=r"\s+",
            header=2,
        )
    # Clean it a little
    df = df.drop(0)
    df = df.reset_index(drop=True)
    # Cast the years to int.
    df.Year = df.Year.astype(int)

    # Create the smoothed data
    df["4yr_smoothing"] = df["No_Smoothing"].rolling(window).mean()

    # Get the first and last year of the cube
    # Assumes that we have a coordinate year.
    try:
        first_year = cube.coord("year").points[0]
        last_year = cube.coord("year").points[-1]
    except CoordinateNotFoundError:
        first_year = cube.coord("season_year").points[0]
        last_year = cube.coord("season_year").points[-1]

    # Select the timespan
    gmst = df[(df.Year >= first_year) & (df.Year <= last_year)].reset_index(drop=True)

    # Get the smoothed data for our interval as an array.
    gmst_data = gmst["4yr_smoothing"].to_numpy().reshape(-1, 1)

    return gmst_data


def get_monthly_gmst(cube=None, path=None, window=4):
    """Get the monthly gmst timeseries for the corresponding cube.

    Arguments
    ---------
    cube : iris.Cube, optional
        Used to get the timespan.
    path : string, optional.
        Path/url MST data.
    window : int
        Size of smoothing window in years.

    Returns
    -------
    gmst_data : np.ndarray
    """
    url = "https://data.giss.nasa.gov/gistemp/graphs_v4/graph_data/Monthly_Mean_Global_Surface_Temperature/graph.txt"
    if not path:
        df = pd.read_csv(
            # Load in the dataset.
            url,
            sep=r"\s+",
            header=2,
        )
    else:
        df = pd.read_csv(
            # Load in the dataset.
            path,
            sep=r"\s+",
            header=2,
        )
    # Drop the first row.
    df = df.drop(0)

    # Wrangling to get an datetime index.
    date_df = df["Year+Month"].str.split(".", expand=True)

    # Month is in decimal format.
    month = date_df[1].astype(int) / 100

    # We round it to zero and multiply with 12 to get it on the form "m".
    month = np.around((month + 0.04) * 12).astype(int)
    # Replace the old month.
    date_df[1] = month

    # Combine them columns again and convert to datetime.
    df["datetime"] = pd.to_datetime(
        date_df[0].astype(str) + "-" + date_df[1].astype(str), format="%Y-%m"
    )

    # No longer need the old date column
    df = df.drop(columns="Year+Month")
    # And set the new one as the index.
    df = df.set_index("datetime")

    # Smooth it
    df = df.rolling(window * 12, center=False).mean()

    if cube:
        # Select the years of the cube.
        first_year = cube.coord("time").cell(0).point.strftime("%Y-%m")
        last_year = cube.coord("time").cell(-1).point.strftime("%Y-%m")
        # Select timespan
        df = df[first_year:last_year]

    # Return numpy array of Land+Ocean temp.
    gmst_data = df["Land+Ocean"].to_numpy()

    return gmst_data

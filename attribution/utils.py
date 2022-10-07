import geopandas as gpd
import iris
import iris.analysis
import iris.analysis.cartography
import iris.coord_categorisation
import iris.util
import iris_utils
import numpy as np
import pandas as pd
import statsmodels.api as sm
from dask.distributed import get_client
from iris.exceptions import CoordinateNotFoundError
from tqdm.autonotebook import trange

from attribution.config import init_config
from attribution.funcs import q_shift_cube_data, shift_cube_data


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
        # Printing will be annoying when doing this 1000s of times.
        # print("Cooridnate month alredy exist.")
        pass
    # Then we add the membership
    try:
        iris.coord_categorisation.add_season_membership(
            cube, "time", name=season_name, season=season_abbr
        )
    except ValueError:
        # print("Season membership already exist.")
        pass

    # Create the constriant
    season_constraint = iris.Constraint(coord_values={season_name: True})

    season_cube = cube.extract(season_constraint)

    return season_cube


def compute_index(
    cube,
    index,
    client=None,
    spatial_average=False,
    spatial_max=False,
):
    """Compute a climate index based on the cube using Climix.

    Arguments:
    ----------
    cube : iris.cube.Cube
        The data cube.
    index : climix.index.Index
        The prepared climix index.
    client : dask.distributed.Client
        Client passed to climix.
    spatial_average : bool, default: False
        Whether to return a spatially averaged cube or not.
    spatial_max : bool, default: False
        Whether to return the spatial max for each year or not.

    Returns:
    --------
    index_cube : iris.cube.Cube

    """

    if client is None:
        client = get_client()
    # Prepare the cube.
    # Can't have a "year" coordinate.
    try:
        cube.remove_coord("year")
    # If there is none, do nothing.
    except CoordinateNotFoundError:
        pass

    # Cube has to be lazy for climix to work.
    iris_utils.utils.make_lazy(cube)

    # If cube is not 3d, add dummy dimensions until 3d.
    if n := 3 - len(cube.shape):
        for _ in range(n):
            cube = iris.util.new_axis(cube)

    # Compute the index.
    index_cube = index([cube], client)

    # Remove dummy dimensions
    index_cube = iris.util.squeeze(index_cube)
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
    cube : iris.cube.Cube or numpy.ndarray
        Cube containing the data on which to perform the regression, or the cube data.
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
    # Did we get a cube?
    if isinstance(cube, iris.cube.Cube):
        data = cube.data
    else:
        data = cube
    # If cube is 1d, we assume that is only a timeseries.
    if len(data.shape) == 1:
        # We use statsmodels
        X = sm.add_constant(predictor)
        # Compute the regression.
        res = sm.OLS(data, X).fit()
        coefs = res.params[-1]
        pvalues = res.pvalues[-1]

        # This returns the compressed and broadcasted array.
        if broadcast_coef:
            coefs = np.asarray([coefs]).reshape(1, 1)
            # Broadcast so that there is one coef for each year.
            coefs = np.repeat(coefs, predictor.shape[0])
            return coefs
        else:
            # If not, we return the masked arrays.
            return coefs, pvalues

    # If cube is 2d we assume it is an ensemble cube, so dims are ens_id, time
    elif len(data.shape) == 2:
        # If cube is 3d we assume that it is time, lat, lon.
        # Store the results
        coefs = np.zeros(data.shape[0])
        pvalues = np.zeros(data.shape[0])
        # We use statsmodels
        X = sm.add_constant(predictor)
        # Loop over all gridpoints. Mask later.
        for ens in trange(data.shape[0]):
            # Get the result
            res = sm.OLS(data[ens, :], X).fit()
            coefs[ens] = res.params[-1]
            pvalues[ens] = res.pvalues[-1]

        # This returns the compressed and broadcasted array.
        if broadcast_coef:
            # Broadcast so that there is one coef for each year.
            coefs = np.broadcast_to(
                coefs[:, np.newaxis], (data.shape[0], predictor.shape[0])
            )
            return coefs
        else:
            # If not, we return the masked arrays.
            return coefs, pvalues

    # If len is 3 we assume data is time, lat, lon.
    elif len(data.shape) == 3:
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
        else:
            # If not, we return the masked arrays.
            return coefs, pvalues
    else:
        raise ValueError("Cube shape not understood.")


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


def get_monthly_gmst(
    cube=None, path=None, n_hemisphere=False, window=4, return_df=False
):
    """Get the monthly gmst timeseries for the corresponding cube.

    Arguments
    ---------
    cube : iris.Cube, optional
        Used to get the timespan.
    path : string, optional.
        Path/url MST data.
    n_hemisphere : bool, default: False
        Use northern hemisphere data.
    window : int
        Size of smoothing window in years.
    return_df : bool, defaul: False
        Return a dataframe. Otherwise a numpy array.

    Returns
    -------
    gmst_data : np.ndarray
    """
    # Use northern hemisphere data?
    if not n_hemisphere:
        # Path?
        if path is None:
            path = "https://data.giss.nasa.gov/gistemp/graphs_v4/graph_data/Monthly_Mean_Global_Surface_Temperature/graph.txt"
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

    # Use northern hemisphere data instead.
    else:
        if path is None:
            path = "https://data.giss.nasa.gov/gistemp/tabledata_v4/NH.Ts+dSST.csv"
        # Read data.
        df = pd.read_csv(
            # Load in the dataset.
            path,
            sep=",",
            header=1,
            na_values="***",
        )
        # Create the index.
        # From first year to last year of the data, left inclusive so that the
        # start of next year is not included.
        index = pd.date_range(
            str(df.iloc[0, 0]), str(df.iloc[-1, 0] + 1), freq="MS", inclusive="left"
        )
        # Get the data.
        data = df.iloc[:, 1:13].to_numpy().flatten()

        # Create a new dataframe. We name it for ease of use later.
        df = pd.DataFrame(data=data, index=index, columns=["Land+Ocean"])

    # Now we can smooth the data.
    df = df.rolling(window * 12, center=False).mean()

    if cube:
        # Select the years of the cube.
        first_year = cube.coord("time").cell(0).point.strftime("%Y-%m")
        last_year = cube.coord("time").cell(-1).point.strftime("%Y-%m")
        # Select timespan
        df = df[first_year:last_year]

    # Return numpy array of Land+Ocean temp.
    if not return_df:
        gmst_data = df["Land+Ocean"].to_numpy()
    else:
        gmst_data = df["Land+Ocean"]

    return gmst_data


def compute_monthly_regression_coefs(cube, monthly_predictor):
    """Compute the monthly regression coefficients.

    Argruments
    ----------
    cube : iris.cube.Cube
        Cube holding the data.
    monthly_predictor : np.ndarray
        Array of a predictor, with same length as the number of months in cube.

    Returns
    -------
    coefs : np.ndarray
        Monthly regression coefficients.
    p_values : np.ndarray
        P-values for corresponding regression coefficients.
    """
    # Get the median.
    monthly_data = cube.aggregated_by(
        ["year", "month_number"], iris.analysis.MEDIAN
    ).data
    # Group the data in months.
    monthly_data = monthly_data.reshape(-1, 12)
    monthly_predictor = monthly_predictor.reshape(-1, 12)
    # Somewhere to store the coefs.
    betas = np.zeros((12))
    p_values = np.zeros((12))

    # Loop over each month.
    for i, (data_month, gmst_month) in enumerate(
        zip(monthly_data.T, monthly_predictor.T)
    ):
        # Add a constant to the predictor.
        X = sm.add_constant(gmst_month)
        # Fit the OLS.
        res = sm.OLS(data_month, X).fit()
        # Save the results.
        betas[i] = res.params[-1]
        p_values[i] = res.pvalues[-1]

    return betas, p_values


def compute_monthly_q_regression_coefs(cube, monthly_predictor, tqdm=False):
    """Compute the monthly quantile regression coefficients.

    Argruments
    ----------
    cube : iris.cube.Cube
        Cube holding the data.
    monthly_predictor : np.ndarray
        Array of a predictor, with same length as the number of months in cube.
    tqdm : bool, default: True

    Returns
    -------
    coefs : np.ndarray
        Monthly regression coefficients.
    p_values : np.ndarray
        P-values for corresponding regression coefficients.
    """
    # How many years in the cube?
    n_years = (
        cube.coord("time").cell(-1).point.year
        - cube.coord("time").cell(0).point.year
        + 1
    )

    betas = np.zeros((12, 30))
    pvalues = np.zeros((12, 30))

    # 30 quantiles between 0 and 1.
    quantiles = np.linspace(0, 1, num=30)
    # Do This for every month.
    monthly_predictor = monthly_predictor.reshape(-1, 12)
    for month in trange(12, disable=not tqdm):
        # Select monthly data.
        constraint = iris.Constraint(month_number=month + 1)
        # Double data since we dont care about the mask at this stage.
        current_data = cube.extract(constraint).data.data
        # Reshape into year x days per month.
        current_data = current_data.reshape(n_years, -1)
        # What are the quantile values?
        q_vals = np.quantile(current_data, quantiles, axis=1)
        # Select the gmst series for that month.
        X = monthly_predictor[:, month]
        X = sm.add_constant(X)
        # Compute the regression for every quantile.
        for i in range(30):
            res = sm.OLS(q_vals[i], X).fit()
            betas[month, i] = res.params[-1]
            pvalues[month, i] = res.pvalues[-1]

    return betas, pvalues


def daily_resampling_windows(array, n_days, n_years):
    """For each entry (i) in array , return the indices in array that corresponds to a
    2D-window centered on i with a length of buffer_days (both directions) and a
    "height" of buffer_years (both directions).

    Assumes a year in array has 365 days.

    Arguments
    ---------
    array : np.ndarray
        1d array of daily data.
    n_days : int
        Number of days to buffer in each direction.
    n_years : int
        Number of years to buffer in each direction.

    Returns
    -------
    windows : np.ndarray
    first_idx : int
        First index with only valid data.
    last_idx : int
        Last index with only valid data.
    """
    # Daily indices 0..n
    indices = np.arange(array.shape[0]).reshape(-1, 1, 1)
    # Number of neighbouring years e.g. -3, -2, ..., 2, 3
    years = np.arange(-n_years, n_years).reshape(1, -1, 1)
    # Number of neighbouring days.
    days = np.arange(-n_days, n_days).reshape(1, 1, -1)

    # This creates the windows. Broadcasting is awesome
    windows = indices + days + years * 365
    # Flatten the 2nd and 3rd dimension.
    windows = windows.reshape(-1, years.shape[1] * days.shape[2])
    # First index with only valid indices
    # first_idx = np.argwhere(windows == 0)[-1][0]
    first_idx = (n_years + 1) * 365
    # Last index with only valid indices
    # last_idx = np.argwhere(windows == array.shape[0])[0][0]
    last_idx = array.shape[0] - (n_years + 1) * 365

    return windows[first_idx:last_idx], first_idx, last_idx


def random_ts_from_windows(window_views, rng, n_resamples=1000):
    """Create a random timeseries from the sliding window view of a timeseries.

    Arguments
    ---------
    window_views : numpy.ndarray
        A sliding window view of the data.
    rng : numpy.random.default_rng()
        A random number generator.
    n_resamples : int, default: 1000
        How many random realisations to create.

    Returns
    -------
    random_ts : np.ndarray
    """
    # How many years does the sliding window series hold?
    n_years = window_views.shape[0]
    # Size of the window, - 1 since arrays are 0 indexed.
    window_size = window_views.shape[1] - 1
    # Somewhere to store the data
    random_ts = np.zeros((n_resamples, n_years))
    # Now we do a loop to select the random columns for each row.
    for i in range(n_resamples):
        # Generate the random columns for this realisation.
        rand_cols = rng.integers(0, window_size, n_years)
        # Select data.
        random_ts[i, :] = np.take_along_axis(
            window_views, rand_cols[..., np.newaxis], axis=1
        ).flatten()

    return random_ts


def prepare_resampled_cubes(
    resampled_data,
    orig_cube,
    predictor,
    first_idx,
    last_idx,
    delta_temp=-1.0,
    quantile_shift=False,
    season=None,
):
    """Calculate the probability ratio of an event using median scaling/shifting.

    Arguments
    ---------
    resampled_data : np.ndarray
        An numpy array holding a resampled version of the cube data.
    orig_cube : iris.cube.Cube
        Iris cube holding the original timeseries.
    predictor : np.ndarray
        Array of values used a predictor in the regression to the cube data.
    first_idx : int
        Index of first full year in reampled cube.
    last_idx : int
        Index of last full year in reampled cube.
    delta_temp : float, default: -1.0
        Temperature difference used to shift the values using the regression coefficients.
    quantile_shift : bool, default: False
        Whether to perform a quantile or median shift of the daily data.
    season : string
        Season abbreviation, if seasonal data should be selected,

    Returns
    -------
    cube, shifted_cube
    """

    # If we have resampled data, we overwrite the cube data.
    # This assumes that the resampling is done correctly.
    if resampled_data is not None:
        cube = orig_cube.copy()
        cube.data[first_idx:last_idx] = resampled_data
        cube = cube[first_idx:last_idx]
    else:
        cube = orig_cube

    # How should we shift the cube data?
    if quantile_shift:
        # Get the monthly regression coefficients.
        betas, _ = compute_monthly_q_regression_coefs(cube, predictor)
        # Shift the cube data.
        shifted_cube = q_shift_cube_data(cube, betas, delta_temp)
    else:
        # Get the monthly regression coefficients.
        betas, _ = compute_monthly_regression_coefs(cube, predictor)
        # Shift the cube data.
        shifted_cube = shift_cube_data(cube, betas, delta_temp)

    if season is not None:
        cube = select_season(cube, season, season)
        shifted_cube = select_season(shifted_cube, season, season)

    return cube, shifted_cube

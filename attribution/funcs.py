"""Collection of helper functions useful when working with attribution."""
import numpy as np
import pandas as pd
from iris.exceptions import CoordinateNotFoundError


def calc_prob_ratio(
    data, regr_slopes, threshold, temperature, dist, scale_dist=True, axis=None
):
    """Calculate the probability ratio for an event of magnitude (threshold) under the
    current climate (data) and a counterfactual climate (shifted/scaled) according to the
    the relationship with GMST.

    Arguements
    ----------
    data : np.ndarray
        Data to perform the calculation on.
    regr_slopes : numpy.ndarray(float) or float
        Regression coefficient(s) between GMST and the variable.
    threshold : int/float
        Threshold value to the investigated event.
    temperature : float
        Temperature (GMST) used to shift/scale the distribution.
    dist : scipy.stats.rv_contious
        Distribution used to fit the data.
    scale_dist : bool
        Whether to scale or shift the distribution. Default: True.
    axis : int, optional
        Needed for bootstrap?

    Returns
    -------
    The probability ratio for the event, based on the current and counterfactual climate.
    """

    data = data.reshape(-1)
    # Resample the data
    # data = data[..., resample_index]
    # Fit a distribution
    fit = dist.fit(data)

    # Calculate the probability under the current climate.
    p1 = 1 - dist.cdf(threshold, *fit)

    # Select a regression slope randomly - adds the variation of the
    # varying regression to GMST.
    if isinstance(regr_slopes, np.ndarray):
        regr_slope = np.median(regr_slopes)
    # If not, we assume a single slope is passed.
    else:
        regr_slope = regr_slopes
    if scale_dist:
        # Scale the distribution to create the counterfactual climate.
        adjusted_fit = scale_dist_params(temperature, *fit, regr_slope)
    else:
        # Shift the distribution.
        adjusted_fit = shift_dist_params(temperature, *fit, regr_slope)
    # Calculate the probability unde the counterfactual climate.
    p0 = 1 - dist.cdf(threshold, *adjusted_fit)

    return p1 / p0


def scale_dist_params(temperature, shape0, loc0, scale0, regr_slope):
    """Scale the distribution by the location and scale parameters.
    Could add sources here.

    Arguments
    ---------
    temperature : float
        Temperature anomaly to scale the distribution to.
    shape0 : float
        Shape parameter. Unaffected.
    loc0 : float
        Location parameter of the distribution.
    scale0 : float
        Scale parameter of the distribution
    regr_slope : float
        Regression slope between GMST and data of the distribution.

    Returns
    -------
    shape : float
    loc : float
    scale : float
    """

    # Calculate the new location
    loc = loc0 * np.exp(regr_slope * temperature / loc0)
    # And scale
    scale = scale0 * np.exp(regr_slope * temperature / loc0)

    # We return the shape unchanged, this makes it convenient to
    # pass the results to a dist.
    return shape0, loc, scale


def shift_dist_params(temperature, shape0, loc0, scale0, regr_slope):
    """Shift the distribution by the location and scale parameters.
    Could add sources here.

    Arguments
    ---------
    temperature : float
        Temperature anomaly to scale the distribution to.
    shape0 : float
        Shape parameter. Unaffected.
    loc0 : float
        Location parameter of the distribution.
    scale0 : float
        Scale parameter of the distribution
    regr_slope : float
        Regression slope between GMST and data of the distribution.

    Returns
    -------
    shape : float
    loc : float
    scale : float
    """

    # Only the location is shifted.
    loc = loc0 + regr_slope * temperature

    return shape0, loc, scale0


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

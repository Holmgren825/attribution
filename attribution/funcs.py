"""Attribution functions."""
import iris
import numpy as np
import statsmodels.api as sm
from iris_utils import iris_utils
from tqdm.autonotebook import trange

import attribution.utils


def calc_prob_ratio(
    data,
    regr_slopes,
    threshold,
    temperature,
    dist,
    scale_dist=True,
    log_sf=False,
    random_slope=False,
    rng=None,
    axis=None,
):
    """Calculate the probability ratio for an event of magnitude (threshold) under the
    current climate (data) and a counterfactual climate (shifted/scaled) according to the
    the relationship with GMST.

    Arguements
    ----------
    data : np.ndarray
        Data to perform the calculation on.
    regr_slopes : np.ndarray
        Regression coefficients between GMST and the variable.
    threshold : int/float
        Threshold value for the investigated event.
    temperature : float
        Temperature (GMST) used to shift/scale the distribution.
    dist : scipy.stats.rv_contious
        Distribution used to fit the data.
    scale_dist : bool
        Whether to scale or shift the distribution. Default: True.
    log_sf : bool, default: False
        Compute the log of the survival function.
    random_slope : bool, default: False
        Select a random slope from regr_slopes for scaling/shifting the distribution.
    rng : np.random.default_rng, optional
        Random number generator.
    axis : int, optional
        Needed for bootstrap?

    Returns
    -------
    prob_ratio : float
        The probability ratio for the event, based on the current and
        counterfactual climate.
    """

    data = data.reshape(-1)
    # Fit a distribution
    fit = dist.fit(data)

    # Select a regression slope randomly - adds the variation of the
    # varying regression to GMST.
    # If not we pick the median.
    if not random_slope:
        regr_slope = np.median(regr_slopes)
    else:
        # Do we have an rng?
        if not rng:
            rng = np.random.default_rng()
        # Select a random slope.
        regr_slope = rng.choice(regr_slopes)
    # Should the distribution be scaled or shifted?
    if scale_dist:
        # Scale the distribution to create the counterfactual climate.
        adjusted_fit = scale_dist_params(temperature, *fit, regr_slope)
    else:
        # Shift the distribution.
        adjusted_fit = shift_dist_params(temperature, *fit, regr_slope)

    # Calculate prob. ratio
    if log_sf:
        p_func = dist.logsf
        # Calculate the probability under the current climate.
        p1 = p_func(threshold, *fit)
        # Calculate the probability unde the counterfactual climate.
        p0 = p_func(threshold, *adjusted_fit)
        # The ratio is inverted for log_sf.
        prob_ratio = p0 / p1
    else:
        # Survival function.
        p_func = dist.sf
        # Probabilities.
        p1 = p_func(threshold, *fit)
        p0 = p_func(threshold, *adjusted_fit)
        # Ratio.
        prob_ratio = p1 / p0

    return prob_ratio


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


def calc_prob_ratio_dms(
    resampled_data,
    orig_cube,
    predictor,
    index_name,
    threshold,
    dist,
    delta_temp=-1.0,
    season=None,
    log_sf=True,
    client=None,
    axis=None,
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
    index_name : string
        Standard name of climte index to compute.
    threshold : float
        Event threshold.
    dist : scipy.stats.rv_contious
        Distribution used to fit the data.
    delta_temp : float, default: -1.0
        Temperature difference used to shift the values using the regression coefficients.
    season : string
        Season abbreviation, if seasonal data should be selected,
    log_sf : bool, default: True
        Compute the log of the survival function.
    client : dask.distributed.Client

    Returns
    -------
    prob_ratio
    """

    # If we have resampled data, we overwrite the cube data.
    # This assumes that the resampling is done correctly.
    if resampled_data is not None:
        orig_cube.data = resampled_data

    # Get the monthly regression coefficients.
    betas, p_values = attribution.utils.compute_monthly_regression_coefs(
        orig_cube, predictor
    )

    # Shift the cube data.
    shifted_cube = shift_cube_data(orig_cube, betas, delta_temp, tqdm=True)

    if season is not None:
        orig_cube = attribution.utils.select_season(orig_cube, season, season)
        shifted_cube = attribution.utils.select_season(shifted_cube, season, season)
    # Need lazy data.
    iris_utils.utils.make_lazy(orig_cube)
    iris_utils.utils.make_lazy(shifted_cube)

    # Compute the index for both cubes..
    index_cube = attribution.utils.compute_index(orig_cube, index_name, client)
    shifted_index_cube = attribution.utils.compute_index(
        shifted_cube, index_name, client
    )

    # Then we can fit the distributions.
    fit1 = dist.fit(index_cube.data)
    fit0 = dist.fit(shifted_index_cube.data)

    if log_sf:
        p1 = dist.logsf(threshold, *fit1)
        p0 = dist.logsf(threshold, *fit0)

        prob_ratio = np.exp(p0 / p1)

    else:
        p1 = dist.sf(threshold, *fit1)
        p0 = dist.sf(threshold, *fit0)

        prob_ratio = p1 / p0

    return prob_ratio


def shift_cube_data(cube, betas, delta_temp, tqdm=False):
    """Shift the data of a cube with a temperature difference following a monthly regression coefficient.

    Arguments
    ---------
    cube : iris.cube.Cube
        Iris cube holding the data to be shifted.
    betas : np.ndarray
        Numpy array with monthly correlation coefficients.
    delta_temp : float
        Temperature difference used to shift the cube data.
    tqdm : bool, default: True
        Whether to disable the tqdm progressbar or not.
    """
    # How many years in the cube?
    n_years = (
        cube.coord("time").cell(-1).point.year - cube.coord("time").cell(0).point.year
    ) + 1

    shifted_cube = cube.copy()
    # Loop over each month in the cube.
    for month in trange(12, disable=tqdm):
        # We want to select data for a certain month.
        constraint = iris.Constraint(month_number=month + 1)
        # Do this in a subcube.
        subcube = cube.extract(constraint)
        # Get the data of the subcube.
        current_data = subcube.data

        beta = betas[month]
        # Compute the shifted values.
        new_vals = current_data + beta * delta_temp

        # Then we need to find where in the non-subsetted cube the new values resides.
        indices = np.searchsorted(
            shifted_cube.coord("time").points, subcube.coord("time").points
        )
        # We then use these indices to overwrite the daily values with the shifted ones.
        shifted_cube.data[indices] = new_vals

    return shifted_cube

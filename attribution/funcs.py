"""Attribution functions."""
import iris
import numpy as np
from tqdm.autonotebook import trange


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


def calc_prob_ratio_ds(current_cube, counter_cube, dist, threshold, log_sf=True):
    """Calculate the probability ratio of an event based on two cubes.

    Arguments
    ---------
    current_cube : iris.cube.Cube
        Iris cube with a climate index timeseries. Assumes that this is the current climate.
    counter_cube : iris.cube.Cube
        Iris cube with a climate index timeseries. Assumes that this is holds the counterfactual climate.
    dist : scipt.stats.rv_contious
        The distribution used to parametrized the data in the cubes.
    threshold : float
        Event threshold.
    log_sf : bool, default: True
        Whether to caclulate the log of the survival function or not.

    Returns
    -------
    prob_ratio : float
    """

    fit1 = dist.fit(current_cube.data)
    fit0 = dist.fit(counter_cube.data)

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
    tqdm : bool, default: False
        Whether to use the tqdm progressbar or not.
    """

    shifted_cube = cube.copy()
    # Loop over each month in the cube.
    for month in trange(12, disable=not tqdm):
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


def q_shift_cube_data(cube, betas, delta_temp, tqdm=False):
    """Quantile shift the data of a cube according to the temperature difference, following
    the monthly regression coefficients.

    Arguments
    ---------
    cube : iris.cube.Cube
        Iris cube holding the data to be shifted.
    betas : np.ndarray
        Numpy array with monthly correlation coefficients.
    delta_temp : float
        Temperature difference used to shift the cube data.
    tqdm : bool, default: False
        Whether to display the tqdm progressbar or not.
    """
    # How many years in the cube?
    n_years = (
        cube.coord("time").cell(-1).point.year - cube.coord("time").cell(0).point.year
    ) + 1

    shifted_cube = cube.copy()
    # Loop over each month in the cube.
    quantiles = np.linspace(0, 1, num=30)
    # Every month.
    for month in trange(12, disable=not tqdm):
        # We want to select data for a certain month.
        constraint = iris.Constraint(month_number=month + 1)
        # Do this in a subcube.
        subcube = shifted_cube.extract(constraint)
        # Get the data of the subcube.
        current_data = subcube.data.data
        # Reshape to get years and nr. of days in month.
        current_data = current_data.reshape(n_years, -1)
        # 30 quantiles per month for each year.
        bins = np.quantile(current_data, quantiles, axis=1).T
        # Which quantile does the daily data "belong" to,
        # i.e. which coefficient should be used for the shifting?
        which_coef = np.array(list(map(np.searchsorted, bins, current_data)))

        # Compute the shifted values.
        # Shift by -1 degree now.
        # TODO Move this to a variable.
        new_vals = current_data + betas[month, which_coef] * delta_temp

        # Then we need to find where in the non-subsetted cube the current data resides.
        indices = np.searchsorted(
            shifted_cube.coord("time").points, subcube.coord("time").points
        )
        # We then use these indices to overwrite the daily values with the shifted ones.
        shifted_cube.data[indices] = new_vals.T.flatten()

    return shifted_cube

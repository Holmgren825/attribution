"""Attribution functions."""
from functools import partial
from multiprocessing import Pool

import iris
import numpy as np
from tqdm.autonotebook import tqdm as tqdm_bar
from tqdm.autonotebook import trange

from attribution.validation import select_distribution


def calc_prob_ratio(
    data,
    reg_coef,
    threshold,
    temperature,
    dist,
    scale_dist=True,
    log_sf=False,
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

    Returns
    -------
    prob_ratio : float
        The probability ratio for the event, based on the current and
        counterfactual climate.
    """

    data = data.reshape(-1)
    # Fit a distribution
    fit = dist.fit(data)

    # Should the distribution be scaled or shifted?
    if scale_dist:
        # Scale the distribution to create the counterfactual climate.
        adjusted_fit = scale_dist_params(temperature, fit, regr_slope=reg_coef)
    else:
        # Shift the distribution.
        adjusted_fit = shift_dist_params(temperature, fit, regr_slope=reg_coef)

    # Calculate log prob. ratio
    if log_sf:
        p_func = dist.logsf
        # Calculate the probability under the current climate.
        p1 = p_func(threshold, *fit)
        # Calculate the probability unde the counterfactual climate.
        p0 = p_func(threshold, *adjusted_fit)
        # The ratio is inverted for log_sf.
        prob_ratio = p1 - p0
    else:
        # Survival function.
        p_func = dist.sf
        # Probabilities.
        p1 = p_func(threshold, *fit)
        p0 = p_func(threshold, *adjusted_fit)
        # Ratio.
        prob_ratio = p1 / p0

    return prob_ratio


def scale_dist_params(temperature, fit, regr_slope):
    """Scale the distribution by the location and scale parameters.
    Could add sources here.

    Arguments
    ---------
    temperature : float
        Temperature anomaly to scale the distribution to.
    fit : tuple
        Tuple holding the distribution parameters.
    regr_slope : float
        Regression slope between GMST and data of the distribution.

    Returns
    -------
    shape : float
    loc : float
    scale : float
    """

    if len(fit) == 4:
        # Unpack
        a, c, loc0, scale0 = fit
        # Calculate the new location
        loc = loc0 * np.exp(regr_slope * temperature / loc0)
        # And scale
        scale = scale0 * np.exp(regr_slope * temperature / loc0)

        return a, c, loc, scale
    elif len(fit) == 3:
        # Unpack
        shape0, loc0, scale0 = fit
        # Calculate the new location
        loc = loc0 * np.exp(regr_slope * temperature / loc0)
        # And scale
        scale = scale0 * np.exp(regr_slope * temperature / loc0)

        return shape0, loc, scale
    elif len(fit) == 2:
        loc0, scale0 = fit
        # Calculate the new location
        loc = loc0 * np.exp(regr_slope * temperature / loc0)
        # And scale
        scale = scale0 * np.exp(regr_slope * temperature / loc0)

        return loc, scale
    else:
        # Scaling is probably to vaible for one parameter distributions
        loc0 = fit
        # Calculate the new location
        loc = loc0 * np.exp(regr_slope * temperature / loc0)
        return loc


def shift_dist_params(temperature, fit, regr_slope):
    """Shift the distribution by the location and scale parameters.
    Could add sources here.

    Arguments
    ---------
    temperature : float
        Temperature anomaly to scale the distribution to.
    fit : tuple
        Tuple holding the distribution parameters.
    regr_slope : float
        Regression slope between GMST and data of the distribution.

    Returns
    -------
    shape : float
    loc : float
    scale : float
    """
    shift = regr_slope * temperature
    if len(fit) == 4:
        a, c, loc0, scale0 = fit
        loc = loc0 + shift
        return a, c, loc, scale0
    elif len(fit) == 3:
        shape0, loc0, scale0 = fit
        loc = loc0 + shift
        return shape0, loc, scale0
    elif len(fit) == 2:
        loc0, scale0 = fit
        loc = loc0 + shift
        return loc, scale0
    else:
        print(fit)
        loc0 = fit
        loc = loc0 + shift
        return loc


def calc_prob_ratio_ds(cube, dists, threshold_quantile, log_sf=True):
    """Calculate the probability ratio of an event based on two cubes.

    Arguments
    ---------
    cube : iris.cube.Cube
        Iris cube holding a single realisation, and all regression variants, of the climate timeseries.
        Should be generated from prepare_resampled_cube.
    dist : dict{scipy.stats.rv_contious}
        The distributions (scipy.stats.rv_contious) used to parametrized the data in the cubes.
    threshold_quantile : float
        Quantile event threshold.
    log_sf : bool, default: True
        Whether to caclulate the log of the survival function or not.

    Returns
    -------
    prob_ratio : float
    """

    # Current climate realisation should be held in the first place of this cube.
    current_cube = cube[0]
    # The two counterfactual climates are held in the two remaning first indices
    counter_cube = cube[1:]
    # Which distribution fit the data the best?
    dist = select_distribution(current_cube.data, dists)
    # Fit the current data.
    fit1 = dist.fit(current_cube.data)
    # Get the threshold through the inverse survival funciton.
    threshold = dist.isf(threshold_quantile, *fit1)
    # Fit the counterfactual climates.
    fit0 = dist.fit(counter_cube[0].data)
    fit0_sig = dist.fit(counter_cube[1].data)

    # Log sf or not?
    if log_sf:
        p1 = dist.logsf(threshold, *fit1)
        # There are two counterfactual climates, depending on regression significance.
        p0 = dist.logsf(threshold, *fit0)
        # Sig. reg.
        p0_sig = dist.logsf(threshold, *fit0_sig)

        prob_ratio = p1 - p0
        prob_ratio_sig = p1 - p0_sig

    else:
        p1 = dist.sf(threshold, *fit1)
        p0 = dist.sf(threshold, *fit0)
        p0_sig = dist.sf(threshold, *fit0_sig)

        prob_ratio = p1 / p0
        prob_ratio_sig = p1 / p0_sig

    return [prob_ratio, prob_ratio_sig]


def shift_cube_data(cube, betas, p_values, delta_temp, p_lim=0.05, tqdm=False):
    """Shift the data of a cube with a temperature difference following a monthly regression coefficient.

    Arguments
    ---------
    cube : iris.cube.Cube
        Iris cube holding the data to be shifted.
    betas : np.ndarray
        Numpy array with monthly correlation coefficients.
    p_values : np.ndarray
        Numpy array with monthly correlation p-values.
    delta_temp : float
        Temperature difference used to shift the cube data.
    p_lim : float, default: 0.05
        Significance level at which to discard regression coefficients.
        Only coefficients with p <= p_lim are used for the significant regression.
    tqdm : bool, default: False
        Whether to use the tqdm progressbar or not.
    """
    cube = cube.copy()
    # Betas for the original data.
    beta_0 = np.zeros(betas.shape)
    # Stack the betas in one array.
    betas = np.array([beta_0, betas, np.where(p_values > p_lim, 0, betas)])
    # And move the axis around a bit.
    betas = np.moveaxis(betas, (0, 1, 2), (1, 0, 2))

    # Loop over each month in the cube.
    print("Shifting daily data")
    for month in trange(12, disable=not tqdm):
        # We want to select data for a certain month.
        constraint = iris.Constraint(month_number=month + 1)
        # Do this in a subcube.
        subcube = cube.extract(constraint)
        # Get the data of the subcube.
        current_data = subcube.data

        # Get beta for each realisation and variant for the month.
        beta = betas[..., month, np.newaxis]
        # Compute the shifted values.
        new_vals = current_data + beta * delta_temp

        # Then we need to find where in the non-subsetted cube the new values resides.
        indices = np.nonzero(
            cube.coord("time").points[:, None] == subcube.coord("time").points
        )[0]
        # We then use these indices to overwrite the daily values with the shifted ones.
        cube.data[..., indices] = new_vals

    return cube


def _q_shift_helper(realisation, betas, quantiles, delta_temp, n_years):

    for month in range(12):
        # We want to select data for a certain month.
        constraint = iris.Constraint(month_number=month + 1)
        # Do this in a subcube.
        subcube = realisation.extract(constraint)
        # Get the data of the subcube.
        current_data = subcube.data
        # Reshape to get years and nr. of days in month.
        current_data = current_data.reshape(3, n_years, -1)
        # 30 quantiles per month for each year.
        bins = np.quantile(current_data[0], quantiles, axis=1).T
        # Which quantile does the daily data "belong" to,
        # i.e. which coefficient should be used for the shifting?
        which_coef = np.array(list(map(np.searchsorted, bins, current_data[0])))

        # Compute the shifted values.
        coefs = betas[:, month, which_coef]
        new_vals = current_data + coefs * delta_temp

        # Find where in the non-subsetted cube the current data resides.
        indices = np.nonzero(
            realisation.coord("time").points[:, None] == subcube.coord("time").points
        )[0]
        # We then use these indices to overwrite the daily values with the shifted ones.
        realisation.data[:, indices] = new_vals.reshape(3, -1)

    return realisation


def q_shift_cube_data(cube, betas, p_values, delta_temp, p_lim=0.05, tqdm=False):
    """Quantile shift the data of a cube according to the temperature difference, following
    the monthly regression coefficients.

    Arguments
    ---------
    cube : iris.cube.Cube
        Iris cube holding the data to be shifted.
    betas : np.ndarray
        Numpy array with monthly correlation coefficients.
    p_values : np.ndarray
        Numpy array with monthly correlation p-values.
    delta_temp : float
        Temperature difference used to shift the cube data.
    tqdm : bool, default: False
        Whether to display the tqdm progressbar or not.
    """
    shape = cube.shape
    # How many years in the cube?
    n_years = (
        cube.coord("time").cell(-1).point.year - cube.coord("time").cell(0).point.year
    ) + 1

    cube = cube.copy()
    # Betas for the original data.
    beta_0 = np.zeros(betas.shape)
    # Get significant betas
    # Currently we select betas from months (axis=2) where there are ANY significant regressions.
    mask = np.any(p_values <= p_lim, axis=2)
    betas_sig = np.where(~mask.reshape(betas.shape[0], -1, 1), 0, betas)
    # Stack the betas in one array.
    betas = np.array([beta_0, betas, betas_sig])
    # And move the axes around so that realisations is the first axis.
    betas = np.moveaxis(betas, (0, 1, 2, 3), (1, 0, 2, 3))

    # Loop over each month in the cube.
    quantiles = np.linspace(0, 1, num=30)
    print("Shifting monthly quantiles")
    # Helper function to distribute.
    q_shift_helper_p = partial(
        _q_shift_helper, quantiles=quantiles, delta_temp=delta_temp, n_years=n_years
    )

    # Distribute it to the pool.
    with Pool() as p:
        shifted_cubes = list(
            tqdm_bar(
                p.istarmap(
                    q_shift_helper_p, zip(cube.slices_over("realization_index"), betas)
                ),
                total=shape[0],
                disable=not tqdm,
            )
        )

    cube = iris.cube.CubeList(shifted_cubes).merge_cube()

    return cube

"""Attribution functions."""
import numpy as np


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

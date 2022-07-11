"""Collection of helper functions useful when working with attribution."""
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as scstats
from tqdm import tqdm


def bootstrap_fit(data, dist, n_resamples=1000):
    """A very basic way to bootstrap the fit of a scipy.rv_continous distribution.

    Arguments
    ---------
    data : ndarray
    dist : scipy.rv_continous distribution
    n_resamples : int
        How many times should the data be resampled. Default: 1000


    Returns
    -------
    results : ndarray(n_resamples, 3)
    """
    rng = np.random.default_rng()
    # We know how many results we need.
    results = np.zeros((n_resamples, 3))
    # Generate all the resamples before the loop.
    # This is basically sampling with replacement.
    indices = rng.integers(0, data.shape[0], (n_resamples, data.shape[0]))

    # We then loop over the different combinations and fit the distribtuion.
    for i, inds in tqdm(enumerate(indices)):
        res = dist.fit(data[..., inds])
        # Save fit params.
        results[i, :] = res
    return results


def scale_dist_params(T, shape0, loc0, scale0, regr_slope):
    """Scale the distribtuion by the location and scale parameters.
    Could add sources here.

    Arguments
    ---------
    T : float
        Temperature anomaly to scale the distribution to.
    shape0 : float
        Shape parameter. Unaffected.
    loc0 : float
        Location parameter of the distribution.
    scale0 : float
        Scale parameter of the distribution
    regr_slop : float
        Regression slope between GMST and data of the distribution.

    Returns
    -------
    shape : float
    loc : float
    scale : float
    """

    # Calculate the new location
    loc = loc0 * np.exp(regr_slope * T / loc0)
    # And scale
    scale = scale0 * np.exp(regr_slope * T / loc0)

    # We return the shape unchanged, this makes it convenient to
    # pass the results to a dist.
    return shape0, loc, scale


def scale_distributions(fits_ci, reg_results, dist, temp=-1.2, percentiles=[5, 50, 95]):
    """Generate the scaled distributions from the bootstrapped fits and
    the regression results.

    Arguments
    ---------
    fits_ci : array(3x3)
        Array with shape, loc and scale for the 5th, 50th and 95th percentile
        distributions. From the bootstrap.
    reg_results : sklearn.linear_model._base.LinearRegression
        Regression results.
    dist : scipy.rv_continuous
        Distribution used for the fit.
    temp : float
        Temperature used to scale the distributions
    percentiles : list(int)
        List of percentiles which to calculate the regression slopes for.

    Returns
    -------
    dist_dict
    """
    # What distribution?
    slopes = np.percentile(reg_results.coef_, percentiles)

    # We store the scaled dists in a dict
    dist_dict = {}
    # We want to generate distributions for each slope
    for slope, perc in zip(slopes, percentiles):
        # Get the scaled parameters
        scaled_params = [scale_dist_params(temp, *fit, slope) for fit in fits_ci]
        # Then we can generate the distribution
        scaled_dists = [dist(*params) for params in scaled_params]

        dist_dict[f"{perc}th"] = scaled_dists

    return dist_dict


def get_gmst(cube, window=4):
    """Get the gmst timeseries for the corresponding cube.

    Arguments
    ---------
    cube : iris.Cube
        Used to get the timespan.
    window : int
        Size of smoothing window

    Returns
    -------
    gmst_data :
    """
    df = pd.read_csv(
        # Load in the dataset.
        "./data/gistemp.txt",
        sep="\s+",
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
    first_year = cube.coord("year").points[0]
    last_year = cube.coord("year").points[-1]
    # Select the timespan
    gmst = df[(df.Year >= first_year) & (df.Year <= last_year)].reset_index(drop=True)

    # Get the smoothed data for our interval as an array.
    gmst_data = gmst["4yr_smoothing"].to_numpy().reshape(-1, 1)

    return gmst_data


def get_probability_ratios(dists, scaled_dists, threshold):
    """Calculate the probability ratios for distributions of current
    and counterfactual climates.

    Arguments
    ---------
    dists : list(scipy.rv_continous)
        List of distributions of the current climate
    scaled_dists : dist_dict from scale_distributions or list(scipy.rv_continous)
        Either a dictionary of distributions: What the scale_distributions returns,
        or a list of distributions.
    threshold : float
        Event threshold to calculate the probability for.

    Returns
    -------
    prob_ratios
    """

    # If we have a dict, we have to flatten it.
    if isinstance(scaled_dists, dict):
        # Some black magic list comprehension to flatten a list of lists.
        scaled_dists = [dist for dists in list(scaled_dists.values()) for dist in dists]
    # then we can calculate the probabilities for each distribution
    p0 = np.asarray([1 - scaled_dist.cdf(threshold) for scaled_dist in scaled_dists])
    p1 = np.asarray([1 - dist.cdf(threshold) for dist in dists])
    # Get the ratios
    prob_ratios = p1.reshape(-1, 1) / p0.reshape(3, 3)

    return prob_ratios


def plot_distribution(data, dists_ci=None, scaled_dists_ci=None, title=None):
    """Plot the histogram of the data along with distributions.

    Arguments
    ---------
    data : ndarray
        1d array of the sample.
    dists_ci : scipy.rv_continous or list
        Either a single distribtuion or a list of distributions.
        If a list, they are assumed to be in ordered percentiles e.g. 5, 50, 95.
        Optional
    scaled_dists_ci : scipy.rv_continous or dict.
        Either a single distribtuion or a dictionary of scaled distributions.
        If a dict, keys correspond to identifier and value is a list of the #!/usr/bin/env python3
        distributions for that key.
        Optional
    title : string
        Title of the plot.
    """
    # Generate an evenly spaced array of data points in our interval.
    x = np.linspace(data.min(), data.max(), num=200)

    # Create the figure.
    fig, ax = plt.subplots(figsize=(7, 5))

    # Plot the histogram
    ax.hist(data, density=True, histtype="stepfilled", label="Observations", alpha=0.5)
    # Current climate
    # If it is a single distribution.
    if dists_ci and isinstance(dists_ci, scstats.rv_continuous):
        ax.plot(x, dists_ci.pdf(x), label="Current climate")

    # If we have a list of distributions.
    elif dists_ci and isinstance(dists_ci, list):
        # We just assumed the length is three.
        ax.plot(x, dists_ci[1].pdf(x), label="Current climate")
        ax.fill_between(x, dists_ci[0].pdf(x), dists_ci[2].pdf(x), alpha=0.5)

    # Counter factual
    if scaled_dists_ci and isinstance(scaled_dists_ci, scstats.rv_continuous):
        ax.plot(x, scaled_dists_ci.pdf(x), label="Counterfactual climate")

    # If we have a list of distributions.
    elif scaled_dists_ci and isinstance(scaled_dists_ci, dict):
        # We just assumed the length is three.
        for key, dists in scaled_dists_ci.items():
            ax.plot(
                x,
                dists[1].pdf(x),
                label=f"Counterfactual climate\n{key} slope",
            )
            ax.fill_between(x, dists[0].pdf(x), dists[2].pdf(x), alpha=0.5)

    # Add a legend
    ax.set_xlabel("Precipitation flux")
    ax.set_ylabel("Density")
    if title:
        ax.set_title(title)
    plt.legend()

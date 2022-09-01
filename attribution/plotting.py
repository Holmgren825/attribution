import numpy as np
import scipy.stats as scstats
from matplotlib import pyplot as plt


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

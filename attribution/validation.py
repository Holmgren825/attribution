import iris.analysis
import iris.analysis.maths as imaths
import iris.analysis.stats as istats
import iris.coord_categorisation
import iris.cube
import numpy as np
import scipy.stats as stats
from iris_utils.utils import get_weights
from matplotlib import pyplot as plt


def check_categorical_coords(candidate_cube: iris.cube.Cube, ref_cube: iris.cube.Cube):
    """Small helper function which makes sure that the cubes have the
    needed categorical coordinates for aggregation.

    Arguments
    ---------
    candidate_cube : iris.cube.Cube
    ref_cube : iris.cube.Cube
    """
    # Just try to add the coord. This all modifies the cubes in place so
    # we don't need to return anything.
    # Years
    try:
        iris.coord_categorisation.add_year(candidate_cube, "time")
    except ValueError:
        pass
    try:
        iris.coord_categorisation.add_year(ref_cube, "time")
    except ValueError:
        pass
    # Months
    try:
        iris.coord_categorisation.add_month(candidate_cube, "time")
    except ValueError:
        pass
    try:
        iris.coord_categorisation.add_month(ref_cube, "time")
    except ValueError:
        pass


def average_anomaly(
    candiate_cube: iris.cube.Cube, ref_cube: iris.cube.Cube
) -> iris.cube.Cube:
    """Calculate the absolute anomaly between the cubes.

    Arguments
    ---------
    candidate_cube : iris.cube.Cube
        Cube with candidate data, e.g. model output.
        Shape should be equal to ref_cube, or with one extra dimension (e.g. ensemble member).
        In this case the reference cube will be broadcaster.
    ref_cube : iris.cube.Cube
        Cube containing the reference dataset. Should be 3-dimensional.

    Returns
    -------
    results : iris.cube.Cube
    """
    # Get the average day for candidate
    candidate_time_mean = candiate_cube.collapsed("time", iris.analysis.MEAN)
    # and observations.
    ref_time_mean = ref_cube.collapsed("time", iris.analysis.MEAN)

    # Get the absolute value of the difference between the average day (candidate - observation)
    diff_cube = imaths.abs(candidate_time_mean - ref_time_mean)
    # And normalise the difference using the average day of the observation.
    # Essentially how large the average daily anomaly is compared to the average day.
    diff_cube = diff_cube / ref_time_mean

    # Finally we take the area average, giving one value for each possible ensemble member.
    weights = get_weights(diff_cube)
    results = diff_cube.collapsed(
        ["grid_latitude", "grid_longitude"], iris.analysis.MEAN, weights=weights
    )

    return results


def average_monthly_anomaly(
    candidate_cube: iris.cube.Cube, ref_cube: iris.cube.Cube
) -> iris.cube.Cube:
    """Calculate the average monthly anomaly between the cubes.
    Averaged over the cube grid points.

    Arguments
    ---------
    candidate_cube : iris.cube.Cube
        Cube with candidate data, e.g. model output.
        Shape should be equal to ref_cube, or with one extra dimension (e.g. ensemble member).
        In this case the reference cube will be broadcaster.
    ref_cube : iris.cube.Cube
        Cube containing the reference dataset. Should be 3-dimensional.

    Returns
    -------
    average_anomaly : iris.cube.Cube
    """
    # First we test if there is a monthly categorical coordinate.
    check_categorical_coords(candidate_cube, ref_cube)
    # First we have to create monthly means for the candidates and observations.
    candidate_monthly_mean = candidate_cube.aggregated_by("month", iris.analysis.MEAN)
    ref_monthly_mean = ref_cube.aggregated_by("month", iris.analysis.MEAN)

    # Absolute monthly anomalies
    diff_cube = imaths.abs(candidate_monthly_mean - ref_monthly_mean)

    # Normalise the anomaly to the average month of the reference data.
    average_anomaly = diff_cube / ref_monthly_mean
    # Get the average monhtly anomaly.
    average_anomaly = average_anomaly.collapsed("month", iris.analysis.MEAN)

    # Finally get the spatial average.
    weights = get_weights(average_anomaly)
    average_anomaly = average_anomaly.collapsed(
        ["grid_latitude", "grid_longitude"], iris.analysis.MEAN, weights=weights
    )

    return average_anomaly


def seasonality_index(
    candidate_cube: iris.cube.Cube,
    ref_cube: iris.cube.Cube,
    climatological: bool = True,
    kge: bool = True,
) -> iris.cube.Cube:
    """Calculate a seasonality index for the candidate cube.

    Arguments
    ---------
    candidate_cube : iris.cube.Cube
        Cube with candidate data, e.g. model output.
        Shape should be equal to ref_cube, or with one extra dimension (e.g. ensemble member).
        In this case the reference cube will be broadcaster.
    ref_cube : iris.cube.Cube
        Cube containing the reference dataset. Should be 3-dimensional.
    climatological : bool
        Control whether the index is calculated on the climatological season or every season.
    kge : bool
        Calculte a true KGE, or the one presented in BLU report. The latter seems to be a slightly
        more forgiving test.

    Returns
    -------
    results : iris.cube.Cube
    """
    # If there already is a month coordinate we pass.
    check_categorical_coords(candidate_cube, ref_cube)

    # We need monthly means for both cubes.
    aggregated_by_coords = ["month"]
    # If we are note doing an climatological mean.
    if not climatological:
        aggregated_by_coords = ["month", "year"]

    # Get the aggregated average for both cubes.
    candidate_mon_mean = candidate_cube.aggregated_by(
        aggregated_by_coords, iris.analysis.MEAN
    )
    ref_mon_mean = ref_cube.aggregated_by(aggregated_by_coords, iris.analysis.MEAN)

    # Calculate the pearson r coeff along the months.
    # Intuitive it feels like we should have all the monthly means for this?
    corr = istats.pearsonr(candidate_mon_mean, ref_mon_mean, corr_coords="month")
    # The correlation squared
    corr = imaths.exponentiate((corr - 1), 2)

    # Which version of the index are we calculating?
    # This is the pure interpretation of the KGE score.
    if kge:
        # Get the standard deviation along the months
        candidate_mon_std = candidate_cube.aggregated_by("month", iris.analysis.STD_DEV)
        ref_mon_std = ref_cube.aggregated_by("month", iris.analysis.STD_DEV)
        # Get the std fraction
        std_frac = (candidate_mon_std / ref_mon_std) - 1
        # Raise to the power of 2
        std_frac = imaths.exponentiate(std_frac, 2)
        # And get the monthly mean
        std_frac = std_frac.collapsed("month", iris.analysis.MEAN)

        # If we are not calculating a climatological index we need the average month as well.
        if not climatological:
            candidate_mon_mean = candidate_cube.aggregated_by(
                "month", iris.analysis.MEAN
            )
            ref_mon_mean = ref_cube.aggregated_by("month", iris.analysis.MEAN)
        # The monthly mean fraction
        mon_mean_frac = (candidate_mon_mean / ref_mon_mean) - 1
        # Raise to the power of 2
        mon_mean_frac = imaths.exponentiate(mon_mean_frac, 2)
        # Get the monthly average
        mon_mean_frac = mon_mean_frac.collapsed("month", iris.analysis.MEAN)

        # Calculate the inner result of the pearson coeff and the fraction
        inner_result = corr + std_frac + mon_mean_frac
    # While this comes from the BLU paper, which also references the Kling-Gupta..
    else:
        # Get the annual average.
        candidate_mean = candidate_cube.collapsed("time", iris.analysis.MEAN)
        ref_mean = ref_cube.collapsed("time", iris.analysis.MEAN)

        # Candidate diff
        candidate_diff = candidate_mon_mean - candidate_mean
        # Reference diff
        ref_diff = ref_mon_mean - ref_mean
        # Raise both to the power of 2
        candidate_diff = imaths.exponentiate(candidate_diff, 2)
        ref_diff = imaths.exponentiate(ref_diff, 2)
        # Sum them
        candidate_diff_sum = candidate_diff.collapsed("month", iris.analysis.SUM)
        ref_diff_sum = ref_diff.collapsed("month", iris.analysis.SUM)
        # Get the square root
        candidate_diff_sum = imaths.exponentiate(candidate_diff_sum, 1 / 2)
        ref_diff_sum = imaths.exponentiate(ref_diff_sum, 1 / 2)
        # The fraction
        diff_fraction = (candidate_diff_sum / ref_diff_sum) - 1
        diff_fraction = imaths.exponentiate(diff_fraction, 2)

        # Final, inner result
        inner_result = corr + diff_fraction

    # Take the square root of the inner result, and subtract from one.
    result = 1 - imaths.exponentiate(inner_result, 1 / 2)

    # Finally we take the area average of this result, leaves one value per ensemble member.
    weights = get_weights(result)
    result = result.collapsed(
        ["grid_latitude", "grid_longitude"], iris.analysis.MEAN, weights=weights
    )

    return result


def pattern_index(
    candidate_cube: iris.cube.Cube, ref_cube: iris.cube.Cube
) -> iris.cube.Cube:
    """Calculate the pattern index for the candidate cube.
    This is the Pearsons correlations coefficient between annual averages for each grid point.

    Arguments
    ---------
    candidate_cube : iris.cube.Cube
        Cube with candidate data, e.g. model output.
        Shape should be equal to ref_cube, or with one extra dimension (e.g. ensemble member).
        In this case the reference cube will be broadcaster.
    ref_cube : iris.cube.Cube
        Cube containing the reference dataset. Should be 3-dimensional.

    Returns
    -------
    results : iris.cube.Cube
    """

    # First we need to add an categorical year, to get annual averages.
    check_categorical_coords(candidate_cube, ref_cube)
    # We want to get the Pearsons correlations coefficient between the annual averages
    # in each cell.
    candidate_annual_average = candidate_cube.aggregated_by("year", iris.analysis.MEAN)
    ref_annual_average = ref_cube.aggregated_by("year", iris.analysis.MEAN)

    # Get the correlation coefficient.
    # We want to check the correlation of the grid, i.e how the modelled pattern for year one
    # correlates to the pattern for observations year one.
    pearsons_r = istats.pearsonr(
        candidate_annual_average,
        ref_annual_average,
        corr_coords=["grid_latitude", "grid_longitude"],
    )

    # Take the average over the years.
    results = pearsons_r.collapsed(["year"], iris.analysis.MEAN)

    return results


def get_scores(data, bins, score_bins):
    """Maps the index value to a score according to bins/scores

    Arguments
    ---------
    data : array_like
        Input data to be scored.
    bins : array_like
        Array of bins. Has to be 1d and monotonic.
    score_bins : array_like
        Array of scores corresponding to the bins.

    Returns
    -------
    scores
    """

    # We digitize the input data to the bins.
    positions = np.digitize(data, bins)

    # Get the score for each entry in data.
    scores = score_bins[positions]

    return scores


def check_dist_params(fits: np.ndarray, fits_ci: np.ndarray, buffer: float = 0.0):
    """Check which which of a list of fit parameters lies within the confidence
    interval of another fit.

    Arguments
    ---------
    fits : np.ndarray
        m distribution parameters from n different distributions. Shape n x m.
    fits_ci : np.ndarray
        Confidence interval to check the fit distribution params against. Should have the shape 3 x m.
    buffer : float
        Allow the candidate fits to outside the reference CI by some percentage.

    Returns
    -------
    results : np.ndarray of bool
    """
    # Check the shapes
    # We assume that the inputs have
    if not fits.shape[1] == fits_ci.shape[1]:
        raise ValueError("Incompatible number of dist parameters.")

    # If we a clear to check the ranges.
    # The fits should lie within the range of the CI.
    # Maybe add some leeway here, that the values can be outside by some percentage?
    # Get the buffers to add
    buffers_low = np.abs(buffer * fits_ci[0, :])
    buffers_high = np.abs(buffer * fits_ci[2, :])
    # Check the candidates.
    results = np.logical_and(
        fits >= fits_ci[0, :] - buffers_low, fits <= fits_ci[2, :] + buffers_high
    )
    # All the params shoud be inside? Or any?
    results = np.all(results, axis=1)

    return results


def inspect_distributions(
    data: np.ndarray,
    dists: dict,
    plot: bool = True,
    cdf: bool = True,
    n_bins: int = 25,
    figsize: tuple = None,
):
    """Utility to inspect which distributions are suitable to represent the data.
    Computes the Kolmogorov-Smirnof 1 sample test and additionally plots the
    over a histogram of the data.

    Arguments
    ---------
    data : np.ndarray
    dists : dict
        Dictionary holding the distributions to evaluate. A key value pair correspond
        to the name of the distribution and the corresponding scipy.stats.rv_continous distribution.
    plot : bool
        Whether to plot the distribution along with their ks p-values.
    cdf : bool
        Plot the cdf instead of the pdf.
    n_bins : int
        Number of bins for the histogram/ECDF.
    figsize : tuple, optional
        Set the size of the figure.

    Returns
    -------
    KS test results.
    """

    # First compute the ks test
    ks_results = {}
    if plot:
        # Create the figure.
        _, ax = plt.subplots(figsize=figsize)
        # Plot the histogram/ecdf.
        ax.hist(
            data,
            n_bins,
            density=True,
            cumulative=cdf,
            histtype="step",
            label="Empirical",
        )

    for key, dist in dists.items():
        # Fit the distribution to the data.
        fit = dist.fit(data)
        # Compute the KS statistic.
        res = stats.ks_1samp(data, dist.cdf, args=fit)
        # Store the results in the dict.
        ks_results[key] = res
        if plot:
            # Need some x values to plot along.
            x = np.linspace(data.min(), data.max(), 200)
            # Plot cdf or pdf.
            if cdf:
                func = dist.cdf
                ax.set_title("ECDF along with CDFs of distributions.")
            else:
                func = dist.pdf
                ax.set_title("Histogram along with distributions fit to the data.")
            # The actual plot
            ax.plot(x, func(x, *fit), label=f"{key}, ks p: {res.pvalue:.4f}")
            plt.legend()

    return ks_results


def select_distribution(data: np.ndarray, dists: dict):
    # Evaluate distribution for the member
    dist_eval = inspect_distributions(data, dists, plot=False)
    # We select the one with the highest p-value.
    dist_idx = np.asarray([res.pvalue for res in dist_eval.values()]).argmax()
    # Get the key. This is ugly.
    dist = dists[list(dists.keys())[dist_idx]]

    return dist

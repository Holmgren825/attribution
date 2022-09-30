import multiprocessing.pool as mpp
import time
from functools import partial
from multiprocessing import Pool

import numpy as np
import scipy.stats as stats
from scipy.special import ndtr, ndtri
from scipy.stats import _resampling as bs
from tqdm.autonotebook import tqdm

from attribution.funcs import calc_prob_ratio_ds
from attribution.utils import (
    compute_index,
    daily_resampling_windows,
    get_monthly_gmst,
    prepare_resampled_cubes,
)


def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap"""
    if self._state != mpp.RUN:
        raise ValueError("Pool not running")

    if chunksize < 1:
        raise ValueError("Chunksize must be 1+, not {0:n}".format(chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job, mpp.starmapstar, task_batches),
            result._set_length,
        )
    )
    return (item for chunk in result for item in chunk)


mpp.Pool.istarmap = istarmap


def _boot_helper(
    data,
    regr_slopes,
    threshold_quantile,
    delta_temp,
    dists,
    random_slope=False,
    log_sf=True,
    scale_dist=False,
):
    """Bootstrap helper function."""
    dist = select_distribution(data, dists)
    # Fit
    fit = dist.fit(data)
    # Get the threshold through the inverse survival funciton.
    threshold = dist.isf(threshold_quantile, *fit)
    # Calculate the ratio.
    prob_ratio = calc_prob_ratio(
        data=data,
        regr_slopes=regr_slopes,
        threshold=threshold,
        temperature=delta_temp,
        random_slope=random_slope,
        log_sf=log_sf,
        # This is important for temperature.
        scale_dist=scale_dist,
        dist=dist,
    )
    return prob_ratio


def prob_ratio_ci(
    data,
    reg_coefs,
    threshold_quantile,
    delta_temp,
    dists,
    ensemble=False,
    alpha=0.05,
    n_resamples=1000,
    log_sf=True,
    random_slope=False,
    scale_dist=False,
    client=None,
):
    """Bootstrapping the confidence interval of statistic.
    Essentially a copy of the internals of scipy.stats._bootstrap, but with some added paralellisation.

    Arguments
    ---------
    data : ndarray
        Data on which to calculate probability ratio.
    reg_coefs : ndarray
        Regression coefficients for the dataset. If data is a single timeseries,
        reg_coefs should be broadcasted so that there is one value per year.
    threshold_quantile : float
        The quantile representation of the event in observations.
    dists : dict
        Dictionary of scipy.stats.rv_continous which are evaluated against the resampled datasets.
    ensemble : bool, default: True
        Is the provided data an ensemble.
    alpha : float
        Confidence level.
    n_resamples : int, defaul: 1000
        How many times should the data be resampled.
    log_sf : bool, default: True
         Did we compute the log sf?
    random_slope : bool, default: False
        Passed on to calc_prob_ratio.
    scale_dist : bool, default: False
        Passed to calc_prob_ratio.
    batch : int, optional
        Number of resamples to process in each call. Set to 1 if Statistic is not vectorized.
    client : dask.distributed.Client
        Use a dask client to map the tasks. Default: None

    Returns
    -------
    prob_ratio_ci : numpy.ndarray
        Quantiles [alpha, 0.25, 0.5, 0.75, 1 - alpha]
    theta_hat_b : numpy.ndarray
        The bootstrap distribution.
    """

    # When did we start?
    t0 = time.time()
    # Random number generator.
    rng = np.random.default_rng()
    # We know how many results we need.
    if not ensemble:
        # Generate all the resamples before the loop.
        # Generate integers in the range 0 to number of samples.
        # And we want to fill an array with shape n_resamples x length of data.
        # This is basically sampling with replacement.
        resample_indices = rng.integers(0, data.shape[0], (n_resamples, data.shape[0]))
        # Then we pick these indices from data and slopes
        data = data[..., resample_indices]
        reg_coefs = reg_coefs[..., resample_indices]

    # Initiate a partial _boot_helper.
    boot_helper_p = partial(
        _boot_helper,
        threshold_quantile=threshold_quantile,
        delta_temp=delta_temp,
        dists=dists,
        random_slope=random_slope,
        log_sf=log_sf,
        scale_dist=scale_dist,
    )
    # If we get a dask client, use it.
    if client:
        print("Submitting resampling tasks to client")
        # Map tasks to the client.
        theta_hat_b = client.map(
            boot_helper_p,
            data,
            reg_coefs,
        )
        # Gather the results
        theta_hat_b = client.gather(theta_hat_b)
    # If we don't have a client.
    else:
        print("Submitting resampling tasks to mp pool")
        # If no client is provided we simply use the standard multiprocessing pool.
        # Likley faster on a single machine.
        with Pool() as p:
            theta_hat_b = list(
                tqdm(
                    p.istarmap(
                        boot_helper_p,
                        zip(data, reg_coefs),
                    ),
                    total=n_resamples,
                )
            )

    t1 = time.time()
    runtime = time.strftime("%H:%M:%S", time.gmtime(t1 - t0))
    print(f"Time to compute: {runtime}")
    # We like arrays
    theta_hat_b = np.asarray(theta_hat_b)
    # Remove nans?
    # TODO it is probably better to deal with this at the statistics callable level instead.
    # e.g. Why did we get the nans in the first place.
    theta_hat_b = theta_hat_b[~np.isnan(theta_hat_b)]
    theta_hat_b = theta_hat_b[~np.isinf(theta_hat_b)]

    # Did we use log_sf?
    if log_sf:
        ci = np.exp(np.quantile(theta_hat_b, [alpha, 0.25, 0.5, 0.75, 1 - alpha]))
    else:
        ci = np.quantile(theta_hat_b, [alpha, 0.25, 0.5, 0.75, 1 - alpha])

    return ci, theta_hat_b


def prob_ratio_ds_ci(
    cube,
    index,
    dist,
    threshold,
    client,
    n_days=20,
    n_years=3,
    predictor=None,
    delta_temp=-1.0,
    quantile_shift=False,
    season=None,
    alpha=0.05,
    n_resamples=1000,
    rng=None,
):
    """Bootstrapping the confidence interval of the probability ratio of
    an event of the specified threshold using daily scaling. This first creates `n_resamples` random
    realisations of the cube. The realisations are then regressed against the `predictor` and a copy
    is shifted according to `delta_temp`. This leaves `n_resamples` pairs of cubes. On each cube the
    `index` is calculated. The index-cube pairs are then used to calculate the probability ratio of the event.
    The median of the probability ratio is bootstrapped from the resulting probability ratios (`n_resamples`).

    Arguments
    ---------
    cube : iris.cube.Cube
        Cube containing the data.
    index : climix.index.Index
        Prepared climix index. The probability ratio is based on the index series
        computed based on the cube.
    dist : scipy.stats.rv_contious
        Distribution used to represent the data.
    threshold : float
        Event threshold.
    client : dask.distributed.Client
        Use a dask client to distribute some tasks.
    n_days : int
        Number of days in the resampling window.
    n_years : int
        Number of years in the resampling window.
    predicor : numpy.ndarray
        Predictor used for the regression.
    delta_temp : float
        Temperature difference used to shift the cube data.
    quantile_shift : bool, default: False
        Wether to quantile shift the cube data, or median shift it.
    season : string
        Season abbreviation, e.g. "mjja". if seasonal data should be selected.
    alpha : float
        Confidence level.
    n_resamples : int, default: 1000
        How many times should the data be resampled.
    rng : numpy.random.default_rng
        Random number generator.

    Returns
    -------
    scipy.stats.BootstrapResult
    median : float
        The median of the bootstrap distribution.
    prob_ratios : ndarray
        The probability ratio distribution.
    """

    # Random number generator.
    if not rng:
        rng = np.random.default_rng()
    # When did we start?
    print("Resampling cubes.")
    t0 = time.time()
    # Get the data from the cube.
    data = cube.data
    # Get the daily window.
    windows, first_idx, last_idx = daily_resampling_windows(data, n_days, n_years)
    # Sample each daily window randomly, n_resamples times.
    resampled_windows = rng.choice(windows, axis=1, size=n_resamples).T
    # Select the resampled data.
    resampled_data = data[resampled_windows]

    # Get the predictor if none is given.
    if predictor is None:
        predictor = get_monthly_gmst(cube[first_idx:last_idx])

    # Create partial for preparing cubes.
    prepare_cubes_p = partial(
        prepare_resampled_cubes,
        orig_cube=cube,
        predictor=predictor,
        first_idx=first_idx,
        last_idx=last_idx,
        delta_temp=delta_temp,
        quantile_shift=quantile_shift,
        season=season,
    )
    # We then map the resampled data to the partial function.
    with Pool() as p:
        resampled_cubes = list(
            tqdm(p.imap(prepare_cubes_p, resampled_data), total=n_resamples)
        )
    # We like arrays.
    resampled_cubes = np.asarray(resampled_cubes)
    # Get the time.
    t1 = time.time()
    runtime = time.strftime("%H:%M:%S", time.gmtime(t1 - t0))
    print(f"Time to compute: {runtime}")

    print("Computing index cubes")
    t2 = time.time()
    # Now we can compute the index cubes.
    # Current clim.
    index_cubes = client.map(compute_index, resampled_cubes[:, 0], index=index)
    index_cubes = client.gather(index_cubes)
    # Shifted clim.
    shifted_index_cubes = client.map(compute_index, resampled_cubes[:, 1], index=index)
    shifted_index_cubes = client.gather(shifted_index_cubes)

    t3 = time.time()
    runtime = time.strftime("%H:%M:%S", time.gmtime(t3 - t2))
    print(f"Time to compute: {runtime}")

    # Now that we have all cubes, we can compute the probability ratio for each pair.
    # We know how many results we need.
    print("Computing prob. ratios.")
    t4 = time.time()
    prob_ratios = client.map(
        calc_prob_ratio_ds,
        index_cubes,
        shifted_index_cubes,
        dist=dist,
        threshold=threshold,
    )
    # Gather the results.
    prob_ratios = client.gather(prob_ratios)
    prob_ratios = np.asarray(prob_ratios)
    # Remove nans?
    prob_ratios = prob_ratios[~np.isnan(prob_ratios)]
    prob_ratios = prob_ratios[~np.isinf(prob_ratios)]
    t5 = time.time()
    runtime = time.strftime("%H:%M:%S", time.gmtime(t5 - t4))
    print(f"Time to compute: {runtime}")

    # Use scipy bootstrap to get the ci of the expected value of the prob ratio.
    ci = stats.bootstrap((prob_ratios,), np.median, confidence_level=1 - alpha)
    median = np.median(prob_ratios)
    # How long did it take?
    runtime = time.strftime("%H:%M:%S", time.gmtime(time.time() - t0))
    print(f"Time to compute: {runtime}")

    return ci, median, prob_ratios

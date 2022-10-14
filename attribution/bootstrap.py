import multiprocessing.pool as mpp
import time
from functools import partial
from multiprocessing import Pool

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from tqdm.autonotebook import tqdm

from attribution.funcs import calc_prob_ratio, calc_prob_ratio_ds
from attribution.utils import (
    compute_cube_regression,
    compute_index,
    daily_resampling_windows,
    get_gmst,
    get_monthly_gmst,
    prepare_resampled_cube,
    random_ts_from_windows,
)
from attribution.validation import select_distribution


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
    predictor,
    threshold_quantile,
    delta_temp,
    p_lim,
    dists,
    log_sf=True,
    scale_dist=False,
):
    """Bootstrap helper function."""
    dist = select_distribution(data, dists)
    # Fit
    fit = dist.fit(data)
    # Get the threshold through the inverse survival funciton.
    threshold = dist.isf(threshold_quantile, *fit)
    # Calculate regression slope and the significance of the regression.
    reg_coef, p_value = compute_cube_regression(data, predictor, broadcast_coef=False)

    # If the p-values is above the threshold, we set the reg_coef to 0.
    if p_value > p_lim:
        reg_coef = 0
    # Calculate the ratio.
    prob_ratio = calc_prob_ratio(
        data=data,
        reg_coef=reg_coef,
        threshold=threshold,
        temperature=delta_temp,
        log_sf=log_sf,
        # This is important for temperature.
        scale_dist=scale_dist,
        dist=dist,
    )
    return prob_ratio


def prob_ratio_ci(
    cube,
    threshold_quantile,
    delta_temp,
    dists,
    predictor=None,
    ensemble=False,
    window_size=5,
    alpha=0.05,
    p_lim=0.05,
    n_resamples=1000,
    log_sf=True,
    scale_dist=False,
    client=None,
):
    """Bootstrapping the confidence interval of the probability ratio. Resamples the timeseries by for each year in the series
    randomly selecting a new year from a sliding window n_resamples times. It then distributes the work of selecting a suitable
    distribution to represent the data, calculating the regression between the series and the predictor and
    calculating the probability ratio based on the "True" and shifted distribution.

    Arguments
    ---------
    cube : iris.cube.Cube
        Cube holding the data on which to calculate the probability ratio.
    threshold_quantile : float
        The quantile representation of the event in observations.
    delta_temp : float
        Temperature used in shifting/scaling the distribution.
    dists : dict
        Dictionary of scipy.stats.rv_continous which are evaluated against the resampled datasets.
    ensemble : bool, default: True
        Is the provided data an ensemble.
    alpha : float, default: 0.05
        Confidence level of the probability ratio.
    p_lim : float, default: 0.05
        P-value at which to set the regression coefficient between the
        predictor and the cube data to 0.
    n_resamples : int, defaul: 1000
        How many times should the data be resampled.
    log_sf : bool, default: True
         Did we compute the log sf?
    scale_dist : bool, default: False
        Passed to calc_prob_ratio.
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
    # Get the data.
    data = cube.data
    # Do we have a predictor?
    if predictor is None:
        predictor = get_gmst(cube)
    # We know how many results we need.
    if not ensemble:
        # Generate all the resamples before the loop.
        # This is essentially an random resample with replacement,
        # but kind of maintains the trend. We can only resample
        # each year from within a window.
        data_windows = sliding_window_view(data, window_size)
        # Select random data from the windows.
        data = random_ts_from_windows(data_windows, rng, n_resamples=n_resamples)
        # We have to shorten the predictor.
        predictor = predictor[: data_windows.shape[0]]

    # Initiate a partial _boot_helper.
    boot_helper_p = partial(
        _boot_helper,
        predictor=predictor,
        threshold_quantile=threshold_quantile,
        delta_temp=delta_temp,
        p_lim=p_lim,
        dists=dists,
        log_sf=log_sf,
        scale_dist=scale_dist,
    )
    # If we get a dask client, use it.
    if client:
        print("Submitting resampling tasks to client")
        # Map tasks to the client.
        theta_hat_b = client.map(boot_helper_p, data)
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
                    p.imap(boot_helper_p, data),
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
    dists,
    threshold_quantile,
    n_days=20,
    n_years=3,
    ensemble=False,
    predictor=None,
    n_hemisphere=True,
    delta_temp=-1.0,
    p_lim=0.05,
    quantile_shift=False,
    season=None,
    alpha=0.05,
    log_sf=True,
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
    dist : scipy.stats.rv_continous
        Dictionary holding scipy.stats.rv_contionous distributions. These are evaluated
        and used to represent the data.
    threshold_quantile : float
        Quantile of the event threshold in observations.
    n_days : int
        Number of days in the resampling window.
    n_years : int
        Number of years in the resampling window.
    ensemble : bool, default: True
        Is the provided data an ensemble.
    predicor : numpy.ndarray
        Predictor used for the regression.
    n_hemisphere : bool, defaul: True
        Use northern hemisphere mst data.
    delta_temp : float
        Temperature difference used to shift the cube data.
    p_lim : float, default: 0.05
        Significance level for regression coefficients.
    quantile_shift : bool, default: False
        Wether to quantile shift the cube data, or median shift it.
    season : string
        Season abbreviation, e.g. "mjja". if seasonal data should be selected.
    log_sf : bool, default: True
        Whether to calculate the log survival function or not.
    alpha : float
        Confidence level.
    n_resamples : int, default: 1000
        How many times should the data be resampled.
    rng : numpy.random.default_rng
        Random number generator.

    Returns
    -------
    prob_ratio_ci : numpy.ndarray
        Quantiles [alpha, 0.25, 0.5, 0.75, 1 - alpha]
    theta_hat_b : numpy.ndarray
        The bootstrap distribution.
    """

    # Random number generator.
    if not rng:
        rng = np.random.default_rng()
    # When did we start?
    print("Generating resampled cube")
    t0 = time.time()
    # Get the data from the cube.
    data = cube.data
    # Is data an ensemble?
    if not ensemble:
        # Get the daily windows.
        windows, first_idx, last_idx = daily_resampling_windows(data, n_days, n_years)
        # Select the resampled data.
        resampled_idx = random_ts_from_windows(windows, rng, n_resamples=n_resamples)
        # It is probably good to copy the data here.
        resampled_data = data[resampled_idx].data
        # Broadcast the realisation so we have 3 of each.
        resampled_data = np.broadcast_to(
            resampled_data.reshape(n_resamples, 1, -1),
            (n_resamples, 3, resampled_data.shape[-1]),
        ).copy()
    else:
        # If we have an ensemble.
        first_idx = None
        last_idx = None
        n_resamples = data.shape[0]
        # We use the data as the resampled data.
        resampled_data = data
        resampled_data = np.broadcast_to(
            resampled_data.reshape(n_resamples, 1, -1),
            (n_resamples, 3, resampled_data.shape[-1]),
        ).copy()
        cube = cube[0, :].copy()

    # Get the predictor if none is given.
    if predictor is None:
        predictor = get_monthly_gmst(
            cube[first_idx:last_idx], n_hemisphere=n_hemisphere
        )

    # Genereate the resampled cube.
    resampled_cube = prepare_resampled_cube(
        cube,
        resampled_data,
        predictor,
        first_idx,
        last_idx,
        delta_temp=delta_temp,
        p_lim=p_lim,
        quantile_shift=quantile_shift,
        season=season,
    )

    # Get the time.
    t1 = time.time()
    runtime = time.strftime("%H:%M:%S", time.gmtime(t1 - t0))
    print(f"Time to complete: {runtime}")

    print("Computing index cube")
    t2 = time.time()
    # Now we can compute the index on all realisations and regression variants in one go.
    index_cube = compute_index(resampled_cube, index=index)

    t3 = time.time()
    runtime = time.strftime("%H:%M:%S", time.gmtime(t3 - t2))
    print(f"Time to complete: {runtime}")
    # Compute the probability ratio for all realisations and regression combinations
    # e.g. orig to all reg. anc orig to sig. reg.
    # We know how many results we need.
    print("Calculating prob. ratios")
    t4 = time.time()
    # Partial function so we can distribute the realisations.
    calc_prob_ratio_p = partial(
        calc_prob_ratio_ds,
        dists=dists,
        threshold_quantile=threshold_quantile,
        log_sf=log_sf,
    )
    # Realise the data before mapping to pool.
    index_cube.data

    # Map the calculations to the processing pool.
    with Pool() as p:
        theta_hat_b_all = list(
            tqdm(
                p.imap(calc_prob_ratio_p, index_cube.slices_over("realization_index")),
                total=n_resamples,
            )
        )

    # We like arrays.
    theta_hat_b_all = np.asarray(theta_hat_b_all)
    # prob_ratios where all regressions have been used.
    theta_hat_b = theta_hat_b_all[:, 0]
    # Remove nans?
    theta_hat_b = theta_hat_b[~np.isnan(theta_hat_b)]
    theta_hat_b = theta_hat_b[~np.isinf(theta_hat_b)]
    # Ratios where only significant regressions have been used.
    theta_hat_b_sig = theta_hat_b_all[:, 1]
    theta_hat_b_sig = theta_hat_b_sig[~np.isnan(theta_hat_b_sig)]
    theta_hat_b_sig = theta_hat_b_sig[~np.isinf(theta_hat_b_sig)]
    t5 = time.time()
    runtime = time.strftime("%H:%M:%S", time.gmtime(t5 - t4))
    print(f"Time to complete: {runtime}")

    # Did we compute the log_sf?
    if log_sf:
        # Get the quantiles for both all and significant regressions.
        prob_ratio_ci = np.exp(
            np.quantile(theta_hat_b, [alpha, 0.25, 0.5, 0.75, 1 - alpha])
        )
        prob_ratio_ci_sig = np.exp(
            np.quantile(theta_hat_b_sig, [alpha, 0.25, 0.5, 0.75, 1 - alpha])
        )
    else:
        prob_ratio_ci = np.quantile(theta_hat_b, [alpha, 0.25, 0.5, 0.75, 1 - alpha])
        prob_ratio_ci_sig = np.quantile(
            theta_hat_b_sig, [alpha, 0.25, 0.5, 0.75, 1 - alpha]
        )
    # Stack it.
    prob_ratio_ci = np.vstack([prob_ratio_ci, prob_ratio_ci_sig])
    # How long did it take?
    runtime = time.strftime("%H:%M:%S", time.gmtime(time.time() - t0))
    print(f"Total runtime: {runtime}")

    return prob_ratio_ci, theta_hat_b_all

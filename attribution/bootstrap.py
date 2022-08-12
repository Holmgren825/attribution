from multiprocessing import Pool
import numpy as np
import time

# Privates...
from scipy.stats import _bootstrap as bs
from scipy.special import ndtr, ndtri


def jackknife_resample(sample, batch=None):
    """Modified from scipy. Jackknife resample the sample. Only one-sample stats for now.
    Yields the indices instead of the actual sample.

    """
    n = sample.shape[-1]
    batch_nominal = batch or n

    for k in range(0, n, batch_nominal):
        # col_start:col_end are the observations to remove
        batch_actual = min(batch_nominal, n - k)

        # jackknife - each row leaves out one observation
        j = np.ones((batch_actual, n), dtype=bool)
        np.fill_diagonal(j[:, k : k + batch_actual], False)
        i = np.arange(n)
        i = np.broadcast_to(i, (batch_actual, n))
        i = i[j].reshape((n - 1, batch_actual))

        yield i


def bca_interval(data, statistic, axis, alpha, theta_hat_b, batch, client):
    """Modified from scipy. Bias-corrected and accelerated interval, but doing the jackknife in paralell.

    Arguments
    ---------
    data : ndarray
        Data on which to calculate statistic.
    statistic : callable
        Function that computes a statistic. Has to take the argument data.
    axis : int
        Does not really do anytihng?
    alpha : float
        Confidence level.
    theta_hat_b : np.ndarray
        The bootstrap distribution of the statistic.
    batch : int, optional
        Number of resamples to process in each call. Set to 1 if Statistic is not vectorized.
    client : dask.distributed.Client
        Use a dask client to map the tasks. Default: None

    Returns
    -------
    alpha1, alpha2

    """
    # closely follows [2] "BCa Bootstrap CIs"
    sample = data[0]  # only works with 1 sample statistics right now
    slopes = data[1]

    # calculate z0_hat
    # We don't do any resampling here.
    theta_hat = np.asarray(statistic(sample, slopes, axis=axis))[..., None]
    percentile = bs._percentile_of_score(theta_hat_b, theta_hat, axis=-1)
    z0_hat = ndtri(percentile)

    # Compute the jackkknife resamples.
    # This is a generator.
    resamples = jackknife_resample(sample, batch)
    resamples = np.asarray(list(resamples))

    resampled_data = sample[..., resamples]
    resampled_slopes = slopes[..., resamples]
    # Store the results.
    theta_hat_i = np.zeros((batch, sample.shape[0]))
    if not client:
        with Pool() as p:
            print("Submitting jackknife tasks to mp pool")
            theta_hat_i = p.starmap(statistic, zip(resampled_data, resampled_slopes))
    else:
        print("Submitting jackknife tasks to client")
        resamples = list(resamples)
        theta_hat_i = client.map(statistic, resampled_data, resampled_slopes)
        theta_hat_i = client.gather(theta_hat_i)

    # theta_hat_i = np.concatenate(theta_hat_i, axis=-1)
    # calculate a_hat
    # Make sure we have an array.
    theta_hat_i = np.asarray(theta_hat_i).reshape(-1)
    # Get the mean.
    theta_hat_dot = theta_hat_i.mean(axis=-1, keepdims=True)
    num = ((theta_hat_dot - theta_hat_i) ** 3).sum(axis=-1)
    den = 6 * ((theta_hat_dot - theta_hat_i) ** 2).sum(axis=-1) ** (3 / 2)
    a_hat = num / den

    # calculate alpha_1, alpha_2
    z_alpha = ndtri(alpha)
    z_1alpha = -z_alpha
    num1 = z0_hat + z_alpha
    alpha_1 = ndtr(z0_hat + num1 / (1 - a_hat * num1))
    num2 = z0_hat + z_1alpha
    alpha_2 = ndtr(z0_hat + num2 / (1 - a_hat * num2))
    return alpha_1, alpha_2


def bootstrap_mp(
    data, statistic, alpha=0.05, n_resamples=9999, batch=None, client=None
):
    """Bootstrapping the confidence interval of statistic accodring to the BCa method.
    Essentially a copy of the internals of scipy.stats._bootstrap, but with some added paralellisation.

    Arguments
    ---------
    data : ndarray
        Data on which to calculate statistic.
    statistic : callable
        Function that computes a statistic. Has to take the argument data.
    alpha : float
        Confidence level.
    n_resamples : int
        How many times should the data be resampled. Default: 9999
    batch : int, optional
        Number of resamples to process in each call. Set to 1 if Statistic is not vectorized.
    client : dask.distributed.Client
        Use a dask client to map the tasks. Default: None

    Returns
    -------
    scipy.stats.BootstrapResult
    median : float
        The median of the bootstrap distribution.
    theta_hat_b : ndarray
        The bootstrap distribution.
    """

    # When did we start?
    t0 = time.time()
    # Random number generator.
    rng = np.random.default_rng()
    # We know how many results we need.
    theta_hat_b = np.zeros(n_resamples)
    # Generate all the resamples before the loop.
    # Generate integers in the range 0 to number of samples.
    # And we want to fill an array with shape n_resamples x length of data.
    # This is basically sampling with replacement.
    resample_indices = rng.integers(
        0, data[0].shape[0], (n_resamples, data[0].shape[0])
    )
    # Then we pick these indices from data and slopes
    resampled_data = data[0][..., resample_indices]
    resampled_slopes = data[1][..., resample_indices]
    # We can select the random slope here
    resampled_slopes = rng.choice(resampled_slopes, axis=1)
    # If we get a dask client, use it.
    if client:
        print("Submitting resampling tasks to client")
        # Map tasks to the client.
        theta_hat_b = client.map(statistic, resampled_data, resampled_slopes)
        # Gather the results
        theta_hat_b = client.gather(theta_hat_b)
    # If we don't have a client.
    else:
        print("Submitting resampling tasks to mp pool")
        # If no client is provided we simply use the standard multiprocessing pool.
        # Likley faster on a single machine.
        with Pool() as p:
            theta_hat_b = p.starmap(statistic, zip(resampled_data, resampled_slopes))

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

    print("Reached Jackknife")
    # Now we have the bootstrap distribution and then use the jackknife to get the correceted CI.
    interval = bca_interval(
        data,
        statistic,
        axis=-1,
        alpha=alpha,
        theta_hat_b=theta_hat_b,
        batch=batch,
        client=client,
    )
    print("Finished Jackknife")
    # Calculate the confidence interval
    ci_l = bs._percentile_along_axis(theta_hat_b, interval[0] * 100)
    ci_u = bs._percentile_along_axis(theta_hat_b, interval[1] * 100)
    # And the median
    median = bs._percentile_along_axis(theta_hat_b, 50)
    # How long did it take?
    t2 = time.time()
    runtime = time.strftime("%H:%M:%S", time.gmtime(t2 - t0))
    print(f"Time to compute: {runtime}")

    return (
        bs.BootstrapResult(
            confidence_interval=bs.ConfidenceInterval(ci_l, ci_u),
            standard_error=np.std(theta_hat_b, ddof=1, axis=-1),
        ),
        median,
        theta_hat_b,
    )

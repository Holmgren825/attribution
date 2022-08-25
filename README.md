# Extreme event attribution
Python package containing functions used for performing an extreme event attribution study.

## Modules

- preprocessing
  Contains functions to pre-process datasets for the study.
- validation
  Scores used to validate model data against observations.
- bootstrap
  A customised bootstrap function which computes the bca interval. Adapted from scipy to distribute tasks on a dask.distributed client.
- funcs
  Contains functions for distribution scaling/shifting and computing the probability ratio of an event.

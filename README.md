# Extreme event attribution
Python package containing functions used for performing an extreme event attribution study.
Work in progress.

## Usage
Clone and install with `pip install -e .`

### Dependencies
- iris
- dask
- geopandas
- cartopy
- scipy
- iris_utils: Small helper package available [here](https://github.com/Holmgren825/iris_utils).

## Modules

### preprocessing
Contains functions to pre-process datasets for the study.

### validation
Scores used to validate model data against observations.

### bootstrap
A customised bootstrap function which computes the bca interval. Adapted from scipy to distribute tasks on a dask.distributed client.

### funcs
Contains functions for distribution scaling/shifting and computing the probability ratio of an event.

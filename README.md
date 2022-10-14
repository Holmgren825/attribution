# Extreme event attribution
Python package containing functions used for performing an extreme event attribution study.
Work in progress.

## Usage
Clone and install with `pip install -e .`

### Dependencies
- cartopy
- [climix](https://git.smhi.se/climix/climix)
- dask
- geopandas
- iris
- iris_utils: Small helper package available [here](https://github.com/Holmgren825/iris_utils).
- scipy

## Modules

### bootstrap
Functions used to perform distributed bootstrap calculations of probability ratios.

### funcs
Contains functions for distribution scaling/shifting and computing the probability ratio of an event.

### plotting
Some functions to simplify plotting.

### preprocessing
Contains functions used in pre-processing datasets for the study. Currently supported datasets:
  1. GridClim
  2. E-OBS
  3. ERA5
  4. Euro-Cordex (Midas bias corrected)
  5. S-Lens

### utils
A collection of utility functions used by other modules.

### validation
A collection of metrics used to validate the models against gridded observations.

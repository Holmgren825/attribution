#!/usr/bin/env python
# coding: utf-8

import glob
import os
import warnings
from copy import deepcopy
from functools import partial
from multiprocessing import Pool

import cartopy.crs as ccrs
import iris
import iris.coord_categorisation
import iris.util
import iris_utils.utils
import numpy as np
from cf_units import Unit
from dask.distributed import Client
from iris.exceptions import CoordinateNotFoundError
from iris.time import PartialDateTime
from tqdm.autonotebook import trange

from attribution.config import init_config
from attribution.utils import get_country_shape


def extract_partial_date(cube, date0, date1):
    """To be used for cube extractions in parallel for cubes in a list.

    Arguments
    ---------
    cube : iris.cube.Cube
        Cube to perform the extraction.
    date0 : list(ints)
        List of partial date (year, month, day) e.g. [2018, 12, 5]
    date1 : list(ints)
        List of partial date (year, month, day) e.g. [2018, 12, 5]

    Returns:
    --------
    Extracted cube.
    """
    # Create partial datetimes
    # First date.
    date0 = list(date0.values())
    pdt0 = PartialDateTime(*date0)
    # Second date.
    date1 = list(date1.values())
    pdt1 = PartialDateTime(*date1)
    # Create the time constraint.
    time_constraint = iris.Constraint(
        time=lambda cell: pdt0 <= cell.point <= pdt1,
    )
    return cube.extract(time_constraint)


def region_selection(
    cube, roi_points, coord_names={"lat": "grid_latitude", "lon": "grid_longitude"}
):
    """Select data from a region of interest in the cube based on corner points.

    Arguments
    ---------
    cube : iris.cube.Cube
        Cube holding the data.
    roi_points : array_like(4)
        Points for longitude and latitude extent.
        [N, S, E, W] e.g [58, 55, 18, 11]
    coord_names : dict, default {"lat": "grid_latitude", "lon": "grid_longitude"}
        Spcify the name of latitude and longitude coordinate in the cube.

    Returns:
    --------
    Subset cube.
    """

    # As always, we like arrays.
    points = np.asarray(roi_points)

    # Target projection
    target_projection = cube.coord_system().as_cartopy_projection()
    # Transform them to the cube projection.
    transformed_points = target_projection.transform_points(
        ccrs.PlateCarree(), points[2:], points[:2]
    )
    # Create the constraint.
    region_constraint = iris.Constraint(
        coord_values={
            coord_names["lat"]: lambda v: transformed_points[:, 1].min()
            < v
            < transformed_points[:, 1].max(),
            coord_names["lon"]: lambda v: transformed_points[:, 0].min()
            < v
            < transformed_points[:, 0].max(),
        }
    )

    # Extract the roi
    cube = cube.extract(region_constraint)
    return cube


def load_gridclim(gridclim_path=None, variable=None, partial_dates=None):
    """Load the gridclim data.

    Agruments
    ---------
    griclim_path : string, optional
        Path to GridClim data. Will read from config.yml by default.
    variable : string, optional
        Which variable to read. Will read from config.yml by default.
    partial_dates : dict, optional
        Dictionary holding the parts of partial dates used to constrain the data.
        E.g.
        {
        "low": {"year": 1985, "month", 12},
        "high": {"year": 2012, "month": 5, "day": 20}
        }
        By default read from config.yml. If False no time extraction is done.

    Returns
    -------
    gc_cube : iris.Cube.cube
        Iris cube with GridClim data.
    """
    # Load the config.
    CFG = init_config()

    # Path to gridclim?
    # If we have a path to gridclim we assume it is complete with the variable.
    if gridclim_path is None:
        # Do we have the variable?
        if variable is None:
            variable = CFG["variable"]
        # Get gridclim path.
        gridclim_path = CFG["paths"]["data"]["gridclim"]
        # Join the path and variable.
        gridclim_path = os.path.join(gridclim_path, variable)

    # Do we have partial dates.
    if partial_dates is None:
        partial_dates = CFG["partial_dates"]
    # This gives a list of files in the base path matchig the wildcard.
    files = glob.glob(gridclim_path + "/*.nc")
    # Create a cube.
    iris.FUTURE.datum_support = True
    cube = iris.load(files)

    _ = iris.util.equalise_attributes(cube)

    # We concat on time.
    gc_cube = cube.concatenate_cube()

    # Only extract time if we have a time_range
    if partial_dates:
        # Extract time range.
        gc_cube = extract_partial_date(
            gc_cube, date0=partial_dates["low"], date1=partial_dates["high"]
        )

    return gc_cube


def get_filename(cube, ds_name, variable, CFG):
    basename = CFG["filenames"][ds_name]
    project_name = CFG["project_name"]
    # Get the timestamp
    t0 = cube.coord("time").cell(0).point.strftime("%Y%m%d")
    t1 = cube.coord("time").cell(-1).point.strftime("%Y%m%d")
    timestamp = t0 + "-" + t1

    # Join the parts
    filename = "_".join([variable, project_name, basename, timestamp])
    # Add format
    filename += ".nc"

    return filename


def prepare_gridclim_cube(
    path=None,
    filename=None,
    variable=None,
    project_path=None,
    shapefile=None,
    partial_dates=None,
    roi_points=None,
    return_cube=False,
):
    """Prepare an iris cube over a selected region and timespan with data
    from the GridClim dataset.

    Arguments
    ---------
    path : string, optional
        Path to a directory containing files for the cordex ensemble members.
        Parsed from config.yml by default.
    filename : string, optional
        Filename which to save the selected dataset to.
        Parsed from config.yml by default.
    variable : string, optional
        CF standard name of the variable.
        Parsed from config.yml by default.
    project_path : string, optional
        Where to save the prepared cube.
        Parsed from config.yml by default.
    shapefile : string, optional
        Path to shapefile
        Parsed from config.yml by default.
    partial_dates : dict, optional
        Dictionary holding the parts of partial dates used to constrain the data.
        E.g.
        {
        "low": {"year": 1985, "month", 12},
        "high": {"year": 2012, "month": 5, "day": 20}
        }
        By default read from config.yml. If False no time extraction is done.
    roi_points : array_like(4), optional
        Points for longitude and latitude extents: [N, S, E, W] = [58, 55, 18, 11].
        By default read from config.yml. If False no spatial extraction is done.
    return_cube : bool, default: False
        Whether to return the cube after saving it.
    """
    # Get the configuration
    CFG = init_config()

    if not shapefile:
        shapefile = CFG["paths"]["shapefile"]
    swe_mainland = get_country_shape(shapefile=shapefile)
    # What variable are we using?
    if not variable:
        variable = CFG["variable"]

    print("Loading GridClim")
    # Path to gridclim?
    if not path:
        path = CFG["paths"]["data"]["gridclim"]
    # Join the path and variable.
    path = os.path.join(path, variable)

    # Do we have partial dates.
    if partial_dates is None:
        partial_dates = CFG["partial_dates"]

    # Load the GridClim cube.
    cube = load_gridclim(path, partial_dates=partial_dates)

    # Any roi_poins?
    if roi_points is None:
        roi_points = list(CFG["roi_mask"].values())
    # If we have roi points by now, extract them.
    # If False, no selection is done.
    if roi_points:
        cube = region_selection(cube, roi_points)

    # Create a mask.
    # mask from shape cant handle the 4d cube so we have to do this manually for now.
    mask = iris_utils.utils.mask_from_shape(
        cube,
        swe_mainland,
        # Relies on CF convention.
        coord_names=("grid_latitude", "grid_longitude"),
    )

    cube = iris.util.mask_cube(cube, mask)

    print("Realising cube, see progression in dask UI")
    cube.data

    print("Saving cube")
    # Where to store the file
    if not project_path:
        project_path = CFG["paths"]["project_folder"]
    # The filename is made up of multiple components of CFG.
    if not filename:
        filename = get_filename(cube, "gridclim", variable, CFG)
    # Saving the prepared cubes.
    iris.save(cube, os.path.join(project_path, filename))
    print("Finished")
    if return_cube:
        return cube


def prepare_eobs_cube(
    path=None,
    filename=None,
    variable=None,
    project_path=None,
    shapefile=None,
    gridclim_path=None,
    partial_dates=None,
    roi_points=None,
    return_cube=False,
):
    """Prepare an iris cube over a selected region and timespan with data
    from the EOBS product.

    Arguments
    ---------
    path : string, optional
        Path to a directory containing files for the eobs product.
        Parsed from config.yml by default.
    filename : string, optional
        Filename which to save the processed dataset as.
        Parsed from config.yml by default.
    variable : string, optional
        CF standard name of the variable.
        Parsed from config.yml by default.
    project_path : string, optional
        Where to save the prepared cube.
        Parsed from config.yml by default.
    shapefile : string, optional
        Path to shapefile
        Parsed from config.yml by default.
    gridclim_path : string, optional
        Path to gridclim data.
        Parsed from config.yml by default.
    partial_dates : dict, optional
        Dictionary holding the parts of partial dates used to constrain the data.
        E.g.
        {
        "low": {"year": 1985, "month", 12},
        "high": {"year": 2012, "month": 5, "day": 20}
        }
        By default read from config.yml. If False no time extraction is done.
    roi_points : array_like(4), optional
        Points for longitude and latitude extents: [N, S, E, W] = [58, 55, 18, 11].
        By default read from config.yml. If False no spatial extraction is done.
    return_cube : bool, default: False
        Whether to return the cube after saving it.
    """

    # Get the configuration
    CFG = init_config()
    if not shapefile:
        shapefile = CFG["paths"]["shapefile"]
    swe_mainland = get_country_shape(shapefile=shapefile)
    # What variable are we using?
    if not variable:
        variable = CFG["variable"]

    print("Loading GridClim")
    # Path to gridclim?
    if not gridclim_path:
        gridclim_path = CFG["paths"]["data"]["gridclim"]
    # Join the path and variable.
    gridclim_path = os.path.join(gridclim_path, variable)

    # Do we have partial dates.
    if partial_dates is None:
        partial_dates = CFG["partial_dates"]

    # Load GridClim.
    gc_cube = load_gridclim(gridclim_path, partial_dates=partial_dates)

    # Load in the CORDEX ensemble.
    print("Loading EOBS")
    if not path:
        path = CFG["paths"]["data"]["eobs"]
    # Full path
    # eobs_base_path = os.path.join(eobs_base_path )
    # All Cordex files.
    files = glob.glob(path + f"/{variable}*.nc")

    cube = iris.load(files)

    # Remove attributes.
    _ = iris.util.equalise_attributes(cube)
    cube = cube.concatenate_cube()

    # We extract the data over the GridClim region. No need for all of Europe.
    ref_lats = gc_cube.coord("grid_latitude").points
    ref_lons = gc_cube.coord("grid_longitude").points
    constraint = iris.Constraint(
        grid_latitude=lambda v: ref_lats.min() <= v <= ref_lats.max(),
        grid_longitude=lambda v: ref_lons.min() <= v <= ref_lons.max(),
    )
    print("Extracting domain")
    cube = cube.extract(constraint)

    if partial_dates:
        print("Extracting timespan")
        cube = extract_partial_date(
            cube, date0=partial_dates["low"], date1=partial_dates["high"]
        )

    # Mask Sweden
    # Create a mask.
    # mask from shape cant handle the 4d cube so we have to do this manually for now.
    mask = iris_utils.utils.mask_from_shape(
        cube,
        swe_mainland,
        coord_names=("grid_latitude", "grid_longitude"),
    )

    cube = iris.util.mask_cube(cube, mask)

    # Any roi points?
    if roi_points is None:
        roi_points = list(CFG["roi_mask"].values())

    # If False we don't extract.
    if roi_points:
        cube = region_selection(cube, roi_points)

    print("Realising cube, see progression in dask UI")
    cube.data

    print("Saving cube")
    # Where to store the file
    if not project_path:
        project_path = CFG["paths"]["project_folder"]
    # The filename is made up of multiple components of CFG.
    if not filename:
        filename = get_filename(cube, "eobs", variable, CFG)
    # Saving the prepared cubes.
    iris.save(cube, os.path.join(project_path, filename))
    print("Finished")

    if return_cube:
        return cube


def prepare_era5_cube(
    path=None,
    filename=None,
    variable=None,
    project_path=None,
    shapefile=None,
    gridclim_path=None,
    partial_dates=None,
    roi_points=None,
    return_cube=False,
):
    """Prepare an iris cube over a selected region and timespan with data
    from the ERA5 dataset.

    Arguments:
    ----------
    path : string, optional
        Path to a directory containing files for the era5 product.
        Parsed from config.yml by default.
    filename : string, optional
        Filename which to save the selected dataset to.
        Parsed from config.yml by default.
    variable : string, optional
        CF standard name of the variable.
        Parsed from config.yml by default.
    project_path : string, optional
        Where to save the prepared cube.
        Parsed from config.yml by default.
    shapefile : string, optional
        Path to shapefile
        Parsed from config.yml by default.
    gridclim_path : string, optional
        Path to gridclim data.
        Parsed from config.yml by default.
    partial_dates : dict, optional
        Dictionary holding the parts of partial dates used to constrain the data.
        E.g.
        {
        "low": {"year": 1985, "month", 12},
        "high": {"year": 2012, "month": 5, "day": 20}
        }
        By default read from config.yml. If False no time extraction is done.
    roi_points : array_like(4), optional
        Points for longitude and latitude extents: [N, S, E, W] = [58, 55, 18, 11].
        By default read from config.yml. If False no spatial extraction is done.
    return_cube : bool, default: False
        Whether to return the cube after saving it.
    """

    # Get the configuration
    CFG = init_config()
    if not shapefile:
        shapefile = CFG["paths"]["shapefile"]
    swe_mainland = get_country_shape(shapefile=shapefile)
    # What variable are we using?
    if not variable:
        variable = CFG["variable"]

    print("Loading GridClim")
    # Path to gridclim?
    if not gridclim_path:
        gridclim_path = CFG["paths"]["data"]["gridclim"]
    # Join the path and variable.
    gridclim_path = os.path.join(gridclim_path, variable)

    # Do we have partial dates.
    if not partial_dates:
        partial_dates = CFG["partial_dates"]

    # Load GridClim.
    gc_cube = load_gridclim(gridclim_path, partial_dates=partial_dates)

    # Load in the ERA5.
    print("Loading ERA5")
    if not path:
        path = CFG["paths"]["data"]["era5"]
    # All ERA5 files.
    files = glob.glob(path + f"/{variable}*.nc")

    cube = iris.load(files)

    # Remove attributes.
    _ = iris.util.equalise_attributes(cube)
    cube = cube.concatenate_cube()
    # We extract the data over the GridClim region. No need for all of Europe.
    ref_lats = gc_cube.coord("grid_latitude").points
    ref_lons = gc_cube.coord("grid_longitude").points
    constraint = iris.Constraint(
        grid_latitude=lambda v: ref_lats.min() <= v <= ref_lats.max(),
        grid_longitude=lambda v: ref_lons.min() <= v <= ref_lons.max(),
    )
    print("Extracting domain")
    cube = cube.extract(constraint)

    # print("Extracting timespan")
    # cube = extract_partial_date(
    #     cube, date0=partial_dates["low"], date1=partial_dates["high"]
    # )

    # Mask Sweden
    # Create a mask.
    # mask from shape cant handle the 4d cube so we have to do this manually for now.
    mask = iris_utils.utils.mask_from_shape(
        cube,
        swe_mainland,
        coord_names=("grid_latitude", "grid_longitude"),
    )

    # Mask.
    cube = iris.util.mask_cube(cube, mask)

    # Any roi points?
    if roi_points is None:
        roi_points = list(CFG["roi_mask"].values())

    # If False we don't extract.
    if roi_points:
        cube = region_selection(cube, roi_points)

    cube = region_selection(cube, roi_points)

    # Realise data in the cube.
    print("Realising cube, see progression in dask UI")
    cube.data

    print("Saving cube")
    # Where to store the file
    if not project_path:
        project_path = CFG["paths"]["project_folder"]
    # The filename is made up of multiple components of CFG.
    if not filename:
        filename = get_filename(cube, "era5", variable, CFG)
    # Saving the prepared cubes.
    iris.save(cube, os.path.join(project_path, filename))
    print("Finished")

    if return_cube:
        return cube


def prepare_cordex_cube(
    path=None,
    filename=None,
    variable=None,
    project_path=None,
    shapefile=None,
    gridclim_path=None,
    partial_dates=None,
    roi_points=None,
    return_cube=False,
):
    """Prepare an iris cube over a selected region and timespan with data
    from the EURO-CORDEX ensemble.

    Arguments
    ---------
    path : string, optional
        Path to a directory containing files for the cordex ensemble members.
        Parsed from config.yml by default.
    filename : string
        Filename which to save the selected dataset to.
        Parsed from config.yml by default.
    variable : string, optional
        CF standard name of the variable.
        Parsed from config.yml by default.
    project_path : string, optional
        Where to save the prepared cube.
        Parsed from config.yml by default.
    shapefile : string, optional
        Path to shapefile
        Parsed from config.yml by default.
    gridclim_path : string, optional
        Path to gridclim data.
        Parsed from config.yml by default.
    partial_dates : dict, optional
        Dictionary holding the parts of partial dates used to constrain the data.
        E.g.
        {
        "low": {"year": 1985, "month", 12},
        "high": {"year": 2012, "month": 5, "day": 20}
        }
        By default read from config.yml. If False no time extraction is done.
    roi_points : array_like(4), optional
        Points for longitude and latitude extents: [N, S, E, W] = [58, 55, 18, 11].
        By default read from config.yml. If False no spatial extraction is done.
    return_cube : bool, default: False
        Whether to return the cube after saving it.
    """

    # Get the configuration
    CFG = init_config()
    if not shapefile:
        shapefile = CFG["paths"]["shapefile"]
    swe_mainland = get_country_shape(shapefile=shapefile)
    # What variable are we using?
    if not variable:
        variable = CFG["variable"]

    print("Loading GridClim")
    # Path to gridclim?
    if not gridclim_path:
        gridclim_path = CFG["paths"]["data"]["gridclim"]
    # Join the path and variable.
    gridclim_path = os.path.join(gridclim_path, variable)

    # Do we have partial dates.
    if partial_dates is None:
        partial_dates = CFG["partial_dates"]

    if partial_dates:
        # Load GridClim.
        gc_cube = load_gridclim(gridclim_path, partial_dates=partial_dates)

    # Load in the CORDEX ensemble.
    print("Loading the CORDEX ensemble")
    if not path:
        path = CFG["paths"]["data"]["cordex"]
    # Full path
    path = os.path.join(path, variable)
    # All Cordex files.
    files = glob.glob(path + "/*_rcp85*.nc")

    cube = iris.load(files)

    # HadGem_CLM is missing 1826 days after the timspan extraction below. So we pop it out.
    _ = cube.pop(32)

    # We use a "normal" mp-pool here.
    if partial_dates:
        print("Extracting timespan")
        func = partial(
            extract_partial_date,
            date0=partial_dates["low"],
            date1=partial_dates["high"],
        )
        # Map the extraction of each ensemble member to be done in parallell.
        with Pool() as p:
            cube = p.map(func, cube)

    # Create a CubeList from a list of cubes.
    cube = iris.cube.CubeList(cube)

    # After this we add a new auxiliary coordinate indicating the ensemble member.
    iris_utils.utils.attribute_to_aux(cube, new_coord_name="ensemble_id")

    # Remove attributes.
    _ = iris.util.equalise_attributes(cube)

    # We also need to remove the height coordinate since not all members have it.
    # This is only relevant for e.g. tasmax.
    for cube_p in cube:
        try:
            cube_p.remove_coord("height")
        except CoordinateNotFoundError:
            pass
    # Now we should be able to merge the cubes along the new coordinate.
    cube = iris_utils.utils.merge_aeq_cubes(cube)
    # Fix time coordinate
    # By now we should have all the correct data in the cube,
    # So we can simply replace the time coordinate to make sure they match,
    cube.remove_coord("time")
    cube.add_dim_coord(gc_cube.coord("time"), 1)

    # Check if grid points are almost equal
    lats = np.all(
        np.isclose(
            gc_cube.coord("grid_latitude").points,
            cube.coord("grid_latitude").points,
        )
    )

    # Check if grid points are almost equal
    longs = np.all(
        np.isclose(
            gc_cube.coord("grid_longitude").points,
            cube.coord("grid_longitude").points,
        )
    )

    # If these are both true we can replace the cordex coordinate points/bounds with the GridClim ones.
    if lats and longs:
        coords = ["grid_latitude", "grid_longitude", "latitude", "longitude"]
        # Loop over the coordinates and copy over points and bounds.
        for coord in coords:
            cube.coord(coord).points = deepcopy(gc_cube.coord(coord).points)
            # Bounds
            cube.coord(coord).bounds = deepcopy(gc_cube.coord(coord).bounds)

    else:
        raise ValueError(
            "Lats and longs not almost equal, not able to homogenise coordinates."
        )

    # If region of interest is None, parse it.
    if roi_points is None:
        roi_points = list(CFG["roi_mask"].values())

    if roi_points:
        # Select the region.
        cube = region_selection(cube, roi_points)

    # Realise data in the cube.
    # For some reason we have to do this before masking.
    print("Realising cube, see progression in dask UI")
    cube.data
    # Mask Sweden
    # Create a mask.
    # mask from shape cant handle the 4d cube so we have to do this manually for now.
    mask = iris_utils.utils.mask_from_shape(
        cube[0, :, :, :],
        swe_mainland,
        coord_names=("grid_latitude", "grid_longitude"),
    )

    # Broadcast along the fourth dimension (ensemble_id).
    mask = np.broadcast_to(mask, cube.shape)

    # Mask the cube.
    cube = iris.util.mask_cube(cube, mask)

    print("Saving cube")
    # Where to store the file
    if not project_path:
        project_path = CFG["paths"]["project_folder"]
    # The filename is made up of multiple components of CFG.
    if not filename:
        filename = get_filename(cube, "cordex", variable, CFG)
    # Saving the prepared cubes.
    iris.save(cube, os.path.join(project_path, filename))
    print("Finished")

    if return_cube:
        return cube


def load_slens_member(
    realisation_idx,
    path=None,
    variable=None,
    shapefile=None,
    partial_dates=None,
    roi_points=None,
):
    """Prepare an iris cube over a selected region and timespan with data
    from a single S-Lens ensemble member.

    Arguments
    ---------
    realisation_idx : int
        Which realisation to load.
    path : string, optional
        Path to a directory containing files for the S-Lens ensemble.
    filename : string, optional
        Filename which to save the selected dataset to.
    variable : string, optional
        CF standard name of the variable.
        Parsed from config.yml by default.
    project_path : string, optional
        Where to save the prepared cube.
        Parsed from config.yml by default.
    shapefile : string, optional
        Path to shapefile
        Parsed from config.yml by default.
    partial_dates : dict, optional
        Dictionary holding the parts of partial dates used to constrain the data.
        E.g.
        {
        "low": {"year": 1985, "month", 12},
        "high": {"year": 2012, "month": 5, "day": 20}
        }
        By default read from config.yml. If False no time extraction is done.
    roi_points : array_like(4), optional
        Points for longitude and latitude extents: [N, S, E, W] = [58, 55, 18, 11].
        By default read from config.yml. If False no spatial extraction is done.
    return_cube : bool, default: False
        Whether to return the cube after saving it.
    """
    # Get the configuration
    CFG = init_config()

    # Get the full name of the realisation folder.
    realisation_idx = f"r{realisation_idx}i1p1f1"

    if not shapefile:
        shapefile = CFG["paths"]["shapefile"]
        shapefile = get_country_shape(shapefile=shapefile)

    # What variable are we using?
    if not variable:
        variable = CFG["variable"]

    if partial_dates is None:
        partial_dates = CFG["partial_dates"]

    # Path to the data.
    if not path:
        path = CFG["paths"]["data"]["s-lens"]

    # TODO This is probably not corretct for s-lens.
    files = glob.glob(path + f"/{realisation_idx}/day/{variable}/*/*/*.nc")

    # Loading the S-Lens data throws a UserWarning
    # "Missing CF-netCDF measure variable 'areacella', referenced by netCDF variable 'tasmax'"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        cube = iris.load(files)

    # Equalise the attributes
    _ = iris.util.equalise_attributes(cube)
    # Concatenate the cube (along time).
    cube = cube.concatenate_cube()

    # Mask Sweden
    # Create a mask.
    mask = iris_utils.utils.mask_from_shape(
        cube,
        shapefile,
        coord_system=False,
        coord_names=("latitude", "longitude"),
    )

    # Broadcast along the fourth dimension (ensemble_id).
    mask = np.broadcast_to(mask, cube.shape)

    # Mask the cube.
    cube = iris.util.mask_cube(cube, mask)

    # If region of interest is None, parse it.
    if roi_points is None:
        roi_points = list(CFG["roi_mask"].values())

    if roi_points:
        region_constraint = iris.Constraint(
            coord_values={
                "latitude": lambda v: roi_points[1] < v < roi_points[0],
                "longitude": lambda v: roi_points[3] < v < roi_points[2],
            }
        )

        # Extract the roi
        cube = cube.extract(region_constraint)

    # Now we can return the cube.
    return cube


def prepare_slens_cube(
    path=None,
    filename=None,
    variable=None,
    project_path=None,
    shapefile=None,
    partial_dates=None,
    roi_points=None,
    return_cube=False,
):
    """Prepare an iris cube over a selected region and timespan with data
    from the S-Lens ensemble.

    Arguments
    ---------
    path : string, optional
        Path to a directory containing files for the S-Lens ensemble.
    filename : string, optional
        Filename which to save the selected dataset to.
    variable : string, optional
        CF standard name of the variable.
        Parsed from config.yml by default.
    project_path : string, optional
        Where to save the prepared cube.
        Parsed from config.yml by default.
    shapefile : string, optional
        Path to shapefile
        Parsed from config.yml by default.
    partial_dates : dict, optional
        Dictionary holding the parts of partial dates used to constrain the data.
        E.g.
        {
        "low": {"year": 1985, "month", 12},
        "high": {"year": 2012, "month": 5, "day": 20}
        }
        By default read from config.yml. If False no time extraction is done.
    roi_points : array_like(4), optional
        Points for longitude and latitude extents: [N, S, E, W] = [58, 55, 18, 11].
        By default read from config.yml. If False no spatial extraction is done.
    return_cube : bool, default: False
        Whether to return the cube after saving it.
    """
    # Get the configuration
    CFG = init_config()

    if not shapefile:
        shapefile = CFG["paths"]["shapefile"]
    shapefile = get_country_shape(shapefile=shapefile)

    # What variable are we using?
    if not variable:
        variable = CFG["variable"]

    if partial_dates is None:
        partial_dates = CFG["partial_dates"]

    print("Loading S-Lens ensemble members")
    # Path to the data.
    if not path:
        path = CFG["paths"]["data"]["s-lens"]

    # If region of interest is None, parse it.
    if roi_points is None:
        roi_points = list(CFG["roi_mask"].values())
    # Load every ensemble member as a single cube and merge in chunks of 10.
    # Merging the chunk reduces the memory use quite a bit.
    cubes_outer = []
    for i in trange(0, 50, 10, desc="Ens. chunk"):
        cubes_inner = []
        for j in trange(101 + i, 111 + i, leave=False, desc="Chunk member"):
            cubes_inner.append(
                load_slens_member(
                    j,
                    path=path,
                    variable=variable,
                    shapefile=shapefile,
                    roi_points=roi_points,
                )
            )

        cube = iris.cube.CubeList(cubes_inner)
        # Create a CubeList from a list of cubes.
        # After this we add a new auxiliary coordinate indicating the variant of the ensemble.
        iris_utils.utils.attribute_to_aux(
            cube,
            attribute_names="realization_index",
            new_coord_name="realization_index",
        )

        # Equalise cube attributes.
        _ = iris.util.equalise_attributes(cube)

        # Merge the cubes.
        cube = cube.merge_cube()
        cubes_outer.append(cube)

    print("Concatenating chunks")
    cube = iris.cube.CubeList(cubes_outer)
    cube = cube.concatenate_cube()

    del cubes_inner
    del cubes_outer
    print("Realising cube, see progression in dask UI")
    # Realising the cube data before saving.
    cube.data
    print("Saving cube")
    # Where to store the file
    if not project_path:
        project_path = CFG["paths"]["project_folder"]
    # The filename is made up of multiple components of CFG.
    if not filename:
        filename = get_filename(cube, "s-lens", variable, CFG)
    # Saving the prepared cubes.
    iris.save(cube, os.path.join(project_path, filename))
    print("Finished")

    if return_cube:
        return cube


def prepare_pthbv_cube(
    path=None,
    filename=None,
    variable=None,
    project_path=None,
    shapefile=None,
    partial_dates=None,
    roi_points=None,
    return_cube=False,
):
    """Prepare an iris cube over a selected region and timespan with data
    from the PTHBV dataset.

    Arguments
    ---------
    path : string, optional
        Path to a directory containing files for the PTHBV data.
        Parse from config.yml
    filename : string, optional
        Filename which to save the selected dataset to.
        Parse from config.yml
    variable : string, optional
        CF standard name of the variable.
        Parsed from config.yml by default.
    project_path : string, optional
        Where to save the prepared cube.
        Parsed from config.yml by default.
    shapefile : string, optional
        Path to shapefile
        Parsed from config.yml by default.
    partial_dates : dict, optional
        Dictionary holding the parts of partial dates used to constrain the data.
        E.g.
        {
        "low": {"year": 1985, "month", 12},
        "high": {"year": 2012, "month": 5, "day": 20}
        }
        By default read from config.yml. If False no time extraction is done.
    roi_points : array_like(4), optional
        Points for longitude and latitude extents: [N, S, E, W] = [58, 55, 18, 11].
        By default read from config.yml. If False no spatial extraction is done.
    return_cube : bool, default: False
        Whether to return the cube after saving it.
    """
    # Get the configuration
    CFG = init_config()

    if not shapefile:
        shapefile = CFG["paths"]["shapefile"]
    swe_mainland = get_country_shape(shapefile=shapefile)

    # What variable are we using?
    # Have to map the CF variable to the name used in pthbv
    if not variable:
        variable_cf = CFG["variable"]
    else:
        variable_cf = variable
    # Match the varible to pthbv names.
    if variable_cf == "pr":
        variable = "_p_"
    elif variable_cf == "tas":
        variable = "_t_"
    else:
        raise ValueError("Variable not available.")

    # Path to the data.
    if path is None:
        path = CFG["paths"]["data"]["pthbv"]

    # A lot of wildcards here since the data is in folders of years and months.
    # And archive/realtime
    # TODO Concatenating ~22000 files is very slow. This should be done in the raw data.
    files = glob.glob(path + f"*{variable}*.nc")
    print("Loading PTHBV")
    iris.FUTURE.datum_support = True
    cube = iris.load(files)
    # Equalise the attributes.
    _ = iris.util.equalise_attributes(cube)
    cube = cube.concatenate_cube()

    # Do we have partial dates?
    if partial_dates is None:
        partial_dates = CFG["partial_dates"]

    # We only extract time if we now have partial dates, either by config or user.
    # If partial_dates = False, we don't to the extraction.
    if partial_dates:
        cube = extract_partial_date(
            cube, date0=partial_dates["low"], date1=partial_dates["high"]
        )

    # Non default roi points?
    if roi_points is None:
        roi_points = list(CFG["roi_mask"].values())
    # Only extract if we by now have roi points.
    if roi_points:
        cube = region_selection(
            cube,
            roi_points,
            coord_names={
                "lat": "projection_y_coordinate",
                "lon": "projection_x_coordinate",
            },
        )

    # Create a mask.
    # mask from shape cant handle the 4d cube so we have to do this manually for now.
    mask = iris_utils.utils.mask_from_shape(
        cube,
        swe_mainland,
        # Relies on CF convention.
        coord_names=("projection_y_coordinate", "projection_x_coordinate"),
    )

    cube = iris.util.mask_cube(cube, mask)

    # Convert units of cube?
    if variable_cf == "pr":
        # PTHBV cube is in kg m-2 (for each day)
        # We want it to be kg m-2 s-1, so have to divide by seconds per day.
        cube.data = cube.core_data() / (24 * 60 * 60)
        # Then we also have to change all the attributes of the cube.
        # long_name, standarn_name, unit, variable. -> fetch these from GridClim.
        cube.standard_name = "precipitation_flux"
        cube.long_name = "Precipitation"
        cube.var_name = variable_cf
        cube.units = Unit("kg m-2 s-1")
    else:
        # Do nothing for temperature?
        print("Temperature data is not converted.")
        pass

    # Realising the cube data before saving.
    print("Realising cube, see progression in dask UI")
    cube.data

    print("Saving cube")
    # Where to store the file
    if not project_path:
        project_path = CFG["paths"]["project_folder"]
    # The filename is made up of multiple components of CFG.
    if not filename:
        filename = get_filename(cube, "pthbv", variable_cf, CFG)
    # Saving the prepared cubes.
    iris.save(cube, os.path.join(project_path, filename))
    print("Finished")

    if return_cube:
        return cube


def main():
    # We assume that we don't have a client if we run this from main.
    client = Client(n_workers=4)
    # Memory manager.
    client.amm.start()

    # TODO Here we can read from cfg which data sources we have and loop over the
    # corresponding prepare functions.


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# coding: utf-8

import glob
import os
from copy import deepcopy
from functools import partial
from multiprocessing import Pool

import cartopy.crs as ccrs
import dask
import iris
import iris.coord_categorisation
import iris.util
import iris_utils.utils
import numpy as np
from dask.distributed import Client
from iris.exceptions import CoordinateNotFoundError
from iris.time import PartialDateTime

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


def prepare_gridclim_cube(
    gridclim_path=None,
    gridclim_filename=None,
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
    gridclim_base_path : string, optional
        Path to a directory containing files for the cordex ensemble members.
        Parsed from config.yml by default.
    gridclim_filename : string, optional
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
    if not gridclim_path:
        gridclim_path = CFG["paths"]["data"]["gridclim"]
    # Join the path and variable.
    gridclim_path = os.path.join(gridclim_path, variable)

    # Do we have partial dates.
    if partial_dates is None:
        partial_dates = CFG["partial_dates"]

    # Load the GridClim cube.
    gc_cube = load_gridclim(gridclim_path, partial_dates=partial_dates)

    # Any roi_poins?
    if roi_points is None:
        roi_points = list(CFG["roi_mask"].values())
    # If we have roi points by now, extract them.
    # If False, no selection is done.
    if roi_points:
        gc_cube = region_selection(gc_cube, roi_points)

    # Create a mask.
    # mask from shape cant handle the 4d cube so we have to do this manually for now.
    mask = iris_utils.utils.mask_from_shape(
        gc_cube,
        swe_mainland,
        # Relies on CF convention.
        coord_names=("grid_latitude", "grid_longitude"),
    )

    gc_cube = iris.util.mask_cube(gc_cube, mask)

    print("Saving cube")
    # Where to store the file
    if not project_path:
        project_path = CFG["paths"]["project_folder"]
    # The filename is made up of multiple components of CFG.
    if not gridclim_filename:
        basename = CFG["filenames"]["gridclim"]
        project_name = CFG["project_name"]
        # Get the timestamp
        t0 = gc_cube.coord("time").cell(0).point.strftime("%Y%m%d")
        t1 = gc_cube.coord("time").cell(-1).point.strftime("%Y%m%d")
        timestamp = t0 + "-" + t1

        # Join the parts
        gridclim_filename = "_".join([variable, project_name, basename, timestamp])
        # Add format
        gridclim_filename += ".nc"

    # Saving the prepared cubes.
    with dask.config.set(scheduler="synchronous"):
        iris.save(gc_cube, os.path.join(project_path, gridclim_filename))
    print("Finished")
    if return_cube:
        return gc_cube


def prepare_eobs_cube(
    eobs_base_path=None,
    eobs_filename=None,
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
    eobs_base_path : string, optional
        Path to a directory containing files for the eobs product.
        Parsed from config.yml by default.
    eobs_filename : string, optional
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

    # Load GridClim.
    gc_cube = load_gridclim(gridclim_path, partial_dates=partial_dates)

    # Load in the CORDEX ensemble.
    print("Loading EOBS")
    if not eobs_base_path:
        eobs_base_path = CFG["paths"]["data"]["eobs"]
    # Full path
    # eobs_base_path = os.path.join(eobs_base_path )
    # All Cordex files.
    files = glob.glob(eobs_base_path + f"/{variable}*.nc")

    eobs_cube = iris.load(files)

    # Remove attributes.
    _ = iris.util.equalise_attributes(eobs_cube)
    eobs_cube = eobs_cube.concatenate_cube()

    # We extract the data over the GridClim region. No need for all of Europe.
    ref_lats = gc_cube.coord("grid_latitude").points
    ref_lons = gc_cube.coord("grid_longitude").points
    constraint = iris.Constraint(
        grid_latitude=lambda v: ref_lats.min() <= v <= ref_lats.max(),
        grid_longitude=lambda v: ref_lons.min() <= v <= ref_lons.max(),
    )
    print("Extracting domain")
    eobs_cube = eobs_cube.extract(constraint)

    if partial_dates:
        print("Extracting timespan")
        eobs_cube = extract_partial_date(
            eobs_cube, date0=partial_dates["low"], date1=partial_dates["high"]
        )

    # Mask Sweden
    # Create a mask.
    # mask from shape cant handle the 4d cube so we have to do this manually for now.
    mask = iris_utils.utils.mask_from_shape(
        eobs_cube,
        swe_mainland,
        coord_names=("grid_latitude", "grid_longitude"),
    )

    eobs_cube = iris.util.mask_cube(eobs_cube, mask)

    # Any roi points?
    if roi_points is None:
        roi_points = list(CFG["roi_mask"].values())

    # If False we don't extract.
    if roi_points:
        eobs_cube = region_selection(eobs_cube, roi_points)

    print("Saving cube")
    # Where to store the file
    if not project_path:
        project_path = CFG["paths"]["project_folder"]
    # The filename is made up of multiple components of CFG.
    if not eobs_filename:
        basename = CFG["filenames"]["eobs"]
        project_name = CFG["project_name"]
        # Get the timestamp
        t0 = eobs_cube.coord("time").cell(0).point.strftime("%Y%m%d")
        t1 = eobs_cube.coord("time").cell(-1).point.strftime("%Y%m%d")
        timestamp = t0 + "-" + t1

        # Join the parts
        filename = "_".join([variable, project_name, basename, timestamp])
        filename += ".nc"
    # Saving the prepared cubes.
    with dask.config.set(scheduler="synchronous"):
        iris.save(eobs_cube, os.path.join(project_path, filename))
    print("Finished")

    if return_cube:
        return eobs_cube


def prepare_era5_cube(
    base_path=None,
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
    base_path : string, optional
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
    if not base_path:
        base_path = CFG["paths"]["data"]["era5"]
    # All ERA5 files.
    files = glob.glob(base_path + f"/{variable}*.nc")

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

    print("Saving cube")
    # Where to store the file
    if not project_path:
        project_path = CFG["paths"]["project_folder"]
    # The filename is made up of multiple components of CFG.
    if not filename:
        basename = CFG["filenames"]["era5"]
        project_name = CFG["project_name"]
        # Get the timestamp
        t0 = cube.coord("time").cell(0).point.strftime("%Y%m%d")
        t1 = cube.coord("time").cell(-1).point.strftime("%Y%m%d")
        timestamp = t0 + "-" + t1

        # Join the parts
        filename = "_".join([variable, project_name, basename, timestamp])
        filename += ".nc"
    # Saving the prepared cubes.
    with dask.config.set(scheduler="synchronous"):
        iris.save(cube, os.path.join(project_path, filename))
    print("Finished")

    if return_cube:
        return cube


def prepare_cordex_cube(
    cordex_base_path=None,
    cordex_filename=None,
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
    cordex_base_path : string, optional
        Path to a directory containing files for the cordex ensemble members.
        Parsed from config.yml by default.
    cordex_filename : string
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
    if not cordex_base_path:
        cordex_base_path = CFG["paths"]["data"]["cordex"]
    # Full path
    cordex_base_path = os.path.join(cordex_base_path, variable)
    # All Cordex files.
    files = glob.glob(cordex_base_path + "/*_rcp85*.nc")

    cordex_cube = iris.load(files)

    # HadGem_CLM is missing 1826 days after the timspan extraction below. So we pop it out.
    _ = cordex_cube.pop(32)

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
            cordex_cube = p.map(func, cordex_cube)

    # Create a CubeList from a list of cubes.
    cordex_cube = iris.cube.CubeList(cordex_cube)

    # After this we add a new auxiliary coordinate indicating the ensemble member.
    iris_utils.utils.attribute_to_aux(cordex_cube, new_coord_name="ensemble_id")

    # Remove attributes.
    _ = iris.util.equalise_attributes(cordex_cube)

    # We also need to remove the height coordinate since not all members have it.
    # This is only relevant for e.g. tasmax.
    for cube in cordex_cube:
        try:
            cube.remove_coord("height")
        except CoordinateNotFoundError:
            pass
    # Now we should be able to merge the cubes along the new coordinate.
    cordex_cube = iris_utils.merge_aeq_cubes(cordex_cube)
    # Fix time coordinate
    # By now we should have all the correct data in the cube,
    # So we can simply replace the time coordinate to make sure they match,
    cordex_cube.remove_coord("time")
    cordex_cube.add_dim_coord(gc_cube.coord("time"), 1)

    # Mask Sweden
    # Create a mask.
    # mask from shape cant handle the 4d cube so we have to do this manually for now.
    mask = iris_utils.utils.mask_from_shape(
        cordex_cube[0, :, :, :],
        swe_mainland,
        coord_names=("grid_latitude", "grid_longitude"),
    )

    # Broadcast along the fourth dimension (ensemble_id).
    mask = np.broadcast_to(mask, cordex_cube.shape)

    # Mask the cube.
    cordex_cube = iris.util.mask_cube(cordex_cube, mask)

    # Check if grid points are almost equal
    lats = np.all(
        np.isclose(
            gc_cube.coord("grid_latitude").points,
            cordex_cube.coord("grid_latitude").points,
        )
    )

    # Check if grid points are almost equal
    longs = np.all(
        np.isclose(
            gc_cube.coord("grid_longitude").points,
            cordex_cube.coord("grid_longitude").points,
        )
    )

    # If these are both true we can replace the cordex coordinate points/bounds with the GridClim ones.
    if lats and longs:
        coords = ["grid_latitude", "grid_longitude", "latitude", "longitude"]
        # Loop over the coordinates and copy over points and bounds.
        for coord in coords:
            cordex_cube.coord(coord).points = deepcopy(gc_cube.coord(coord).points)
            # Bounds
            cordex_cube.coord(coord).bounds = deepcopy(gc_cube.coord(coord).bounds)

    else:
        raise ValueError(
            "Lats and longs not almost equal, not able to homogenise coordinates."
        )

    # If region of interest is None, parse it.
    if roi_points is None:
        roi_points = list(CFG["roi_mask"].values())

    if roi_points:
        # Select the region.
        cordex_cube = region_selection(cordex_cube, roi_points)

    print("Saving cube")
    # Where to store the file
    if not project_path:
        project_path = CFG["paths"]["project_folder"]
    # The filename is made up of multiple components of CFG.
    if not cordex_filename:
        basename = CFG["filenames"]["cordex"]
        project_name = CFG["project_name"]
        # Get the timestamp
        t0 = cordex_cube.coord("time").cell(0).point.strftime("%Y%m%d")
        t1 = cordex_cube.coord("time").cell(-1).point.strftime("%Y%m%d")
        timestamp = t0 + "-" + t1

        # Join the parts
        cordex_filename = "_".join([variable, project_name, basename, timestamp])
        cordex_filename += ".nc"
    # Saving the prepared cubes.
    with dask.config.set(scheduler="synchronous"):
        iris.save(cordex_cube, os.path.join(project_path, cordex_filename))
    print("Finished")

    if return_cube:
        return cordex_cube


def prepare_slens(
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
    from the GridClim product.

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
    swe_mainland = get_country_shape(shapefile=shapefile)

    # What variable are we using?
    if not variable:
        variable = CFG["variable"]

    # Path to the data.
    if not path:
        path = CFG["paths"]["data"]["slens"]


def prepare_pthbv(
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

    print("Saving cube")
    # Where to store the file
    if not project_path:
        project_path = CFG["paths"]["project_folder"]
    # The filename is made up of multiple components of CFG.
    if not filename:
        basename = CFG["filenames"]["pthbv"]
        project_name = CFG["project_name"]
        # Get the timestamp
        t0 = cube.coord("time").cell(0).point.strftime("%Y%m%d")
        t1 = cube.coord("time").cell(-1).point.strftime("%Y%m%d")
        timestamp = t0 + "-" + t1

        # Join the parts
        filename = "_".join([variable_cf, project_name, basename, timestamp])
        # Add format
        filename += ".nc"
    # Saving the prepared cubes.
    with dask.config.set(scheduler="synchronous"):
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

#!/usr/bin/env python
# coding: utf-8

import glob
import os
from copy import deepcopy
from functools import partial
from multiprocessing import Pool

import cartopy.crs as ccrs
import dask
import dask.distributed
import geopandas as gpd
import iris
import iris.coord_categorisation
import iris.util
import iris_utils
import numpy as np
from dask.distributed import Client
from iris.exceptions import CoordinateNotFoundError
from iris.time import PartialDateTime

from attribution.config import init_config


def extract_partial_date(cube, date0, date1):
    """To be used for cube extractions in parallel for cubes in a list.

    Arguments:
    ----------
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


def region_selection(cube, roi_points):
    """Select data from a region of interest in the cube based on corner points.

    Arguments:
    ----------
    cube : iris.cube.Cube
        Cube holding the data.
    roi_points : array_like(4)
        Points for longitude and latitude extent.
        [N, S, E, W] e.g [58, 55, 18, 11]

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
        grid_latitude=lambda v: transformed_points[:, 1].min()
        < v
        < transformed_points[:, 1].max(),
        grid_longitude=lambda v: transformed_points[:, 0].min()
        < v
        < transformed_points[:, 0].max(),
    )

    # Extract the roi
    cube = cube.extract(region_constraint)
    return cube


def load_gridclim(gridclim_path=None, variable=None, time_range=None):
    """Load the gridclim data.

    Agruments:
    ----------
    griclim_path : string, optional
        Path to GridClim data. Will read from config.yml by default.
    variable : string, optional
        Which variable to read. Will read from config.yml by default.
    time_range : array_like, optional
        Sequence of years describing the lower and upper range of the timespan.
        Will read from config.yml by default.

    Returns:
    --------
    gc_cube : iris.Cube.cube
        Iris cube with GridClim data.
    """

    # Path to gridclim?
    if gridclim_path is None:
        # Load the config.
        CFG = init_config()
        # Get gridclim path.
        gridclim_path = CFG["paths"]["data"]["gridclim"]
    # Do we have the variable?
    if variable is None:
        variable = CFG["variable"]
    # Join the path and variable.
    gridclim_path = os.path.join(gridclim_path, variable)

    # Do we have partial dates.
    if not time_range:
        time_range = [
            CFG["partial_dates"]["low"]["year"],
            CFG["partial_dates"]["high"]["year"],
        ]
    # This gives a list of files in the base path matchig the wildcard.
    files = glob.glob(gridclim_path + "/*.nc")
    # Create a cube.
    iris.FUTURE.datum_support = True
    cube = iris.load(files)

    _ = iris.util.equalise_attributes(cube)

    # We concat on time.
    gc_cube = cube.concatenate_cube()

    # Create a time constraint
    time_constraint = iris.Constraint(
        time=lambda cell: time_range[0] <= cell.point.year <= time_range[1]
    )
    # Extract time range.
    gc_cube = gc_cube.extract(time_constraint)

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
    from the GridClim product.

    Arguments:
    ----------
    gridclim_base_path : string
        Path to a directory containing files for the cordex ensemble members.
        Default: None. Parse from config.yml
    gridclim_filename : string
        Filename which to save the selected dataset to.
        Default: None. Parse from config.yml
    variable : string
        CF standard name of the variable.
    project_path : string
        Where to save the prepared cube.
    shapefile : string
        Path to shapefile
    partial_dates : dict
        Dictionary holding the parts of partial dates used to constrain the data.
        E.g. {
        "low": {"year": 1985, "month", 12},
        "high": {"year": 2012, "month": 5, "day": 20}
        }
    roi_points : array_like(4)
        Points for longitude and latitude extent. Optional.
        [N, S, E, W] e.g [58, 55, 18, 11]
    return_cube : False
        Whether to return the cube after saving it. Default: False

    """
    # Get the configuration
    CFG = init_config()

    # This file contains shapes of most countries in the world.
    # https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-0-boundary-lines/
    if not shapefile:
        shapefile = CFG["paths"]["shapefile"]

    gdf = gpd.read_file(shapefile)

    # Select Sweden.
    # TODO  This should be in config somehow maybe?
    swe_shapes = gdf[gdf.SOVEREIGNT == "Sweden"].geometry
    swe_mainland = swe_shapes.iloc[0].geoms[0]
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
        time_range = [
            CFG["partial_dates"]["low"]["year"],
            CFG["partial_dates"]["high"]["year"],
        ]
        partial_dates = CFG["partial_dates"]

    # Load the GridClim cube.
    gc_cube = load_gridclim(gridclim_path, time_range)

    # Extract roi
    if not roi_points:
        roi_points = list(CFG["roi_mask"].values())
    gc_cube = region_selection(gc_cube, roi_points)

    # Create a mask.
    # mask from shape cant handle the 4d cube so we have to do this manually for now.
    mask = iris_utils.mask_from_shape(
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

    Arguments:
    ----------
    eobs_base_path : string
        Path to a directory containing files for the eobs product.
        Default: None. Parse from config.yml
    eobs_filename : string
        Filename which to save the selected dataset to.
        Default: None. Parse from config.yml
    variable : string
        CF standard name of the variable.
    project_path : string
        Where to save the prepared cube.
    shapefile : string
        Path to shapefile
    gridclim_path : string
        Path to gridclim data.
    partial_dates : dict
        Dictionary holding the parts of partial dates used to constrain the data.
        E.g. {
        "low": {"year": 1985, "month", 12},
        "high": {"year": 2012, "month": 5, "day": 20}
        }
    roi_points : array_like(4)
        Points for longitude and latitude extent. Optional.
        [N, S, E, W] e.g [58, 55, 18, 11]
    return_cube : bool
        Wheter to return the cube after saving it. Default: False.

    """

    # Get the configuration
    CFG = init_config()
    # This file contains shapes of most countries in the world.
    # https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-0-boundary-lines/
    if not shapefile:
        shapefile = CFG["paths"]["shapefile"]

    gdf = gpd.read_file(shapefile)

    # Select Sweden.
    # TODO  This should be in config somehow maybe?
    swe_shapes = gdf[gdf.SOVEREIGNT == "Sweden"].geometry
    swe_mainland = swe_shapes.iloc[0].geoms[0]
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
        time_range = [
            CFG["partial_dates"]["low"]["year"],
            CFG["partial_dates"]["high"]["year"],
        ]
        partial_dates = CFG["partial_dates"]

    # Load GridClim.
    gc_cube = load_gridclim(gridclim_path, time_range)

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

    print("Extracting timespan")
    eobs_cube = extract_partial_date(
        eobs_cube, date0=partial_dates["low"], date1=partial_dates["high"]
    )

    # Mask Sweden
    # Create a mask.
    # mask from shape cant handle the 4d cube so we have to do this manually for now.
    mask = iris_utils.mask_from_shape(
        eobs_cube,
        swe_mainland,
        coord_names=("grid_latitude", "grid_longitude"),
    )

    # TODO replace with iris.utils when it is lazy.
    eobs_cube = iris.util.mask_cube(eobs_cube, mask)

    # Select roi
    if not roi_points:
        roi_points = list(CFG["roi_mask"].values())

    eobs_cube = region_selection(eobs_cube, roi_points)
    # gc_cube = region_selection(gc_cube, roi_points)

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

    Arguments:
    ----------
    cordex_base_path : string
        Path to a directory containing files for the cordex ensemble members.
        Default: None. Parse from config.yml
    cordex_filename : string
        Filename which to save the selected dataset to.
        Default: None. Parse from config.yml
    variable : string
        CF standard name of the variable.
    project_path : string
        Where to save the prepared cube.
    shapefile : string
        Path to shapefile
    gridclim_path : string
        Path to gridclim data.
    partial_dates : dict
        Dictionary holding the parts of partial dates used to constrain the data.
        E.g. {
        "low": {"year": 1985, "month", 12},
        "high": {"year": 2012, "month": 5, "day": 20}
        }
    roi_points : array_like(4)
        Points for longitude and latitude extent. Optional.
        [N, S, E, W] e.g [58, 55, 18, 11]
    return_cube : bool
        Wheter to return the cube after saving it. Default: False.

    """

    # Get the configuration
    CFG = init_config()
    # This file contains shapes of most countries in the world.
    # https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-0-boundary-lines/
    if not shapefile:
        shapefile = CFG["paths"]["shapefile"]

    gdf = gpd.read_file(shapefile)

    # Select Sweden.
    # TODO  This should be in config somehow maybe?
    swe_shapes = gdf[gdf.SOVEREIGNT == "Sweden"].geometry
    swe_mainland = swe_shapes.iloc[0].geoms[0]
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
        time_range = [
            CFG["partial_dates"]["low"]["year"],
            CFG["partial_dates"]["high"]["year"],
        ]
        partial_dates = CFG["partial_dates"]

    # Load GridClim.
    gc_cube = load_gridclim(gridclim_path, time_range)

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
    iris_utils.attribute_to_aux(cordex_cube, new_coord_name="ensemble_id")

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
    mask = iris_utils.mask_from_shape(
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
        coords = ["grid_latitide", "grid_longitude", "latitude", "longitude"]
        # Loop over the coordinates and copy over points and bounds.
        for coord in coords:
            cordex_cube.coord(coord).points = deepcopy(gc_cube.coord(coord).points)
            # Bounds
            cordex_cube.coord(coord).bounds = deepcopy(gc_cube.coord(coord).bounds)

    else:
        raise ValueError(
            "Lats and longs not almost equal, not able to homogenise coordinates."
        )

    # If region of interest is not provided, parse it.
    if not roi_points:
        roi_points = list(CFG["roi_mask"].values())

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


def main():
    # We assume that we don't have a client if we run this from main.
    client = Client(n_workers=4)
    # Memory manager.
    client.amm.start()

    # TODO Here we can read from cfg which data sources we have and loop over the
    # corresponding prepare functions.


if __name__ == "__main__":
    main()

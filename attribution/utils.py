from climix.metadata import load_metadata
import iris
import iris.analysis
import iris.analysis.cartography
import iris.coord_categorisation
from iris.exceptions import CoordinateNotFoundError


def select_season(cube, season_abbr, season_name="season"):
    """Select data from a cube for a specific seasonself.

    Arguments:
    ----------
    cube : iris.Cube.cube
        Cube hodling the data.
    season_abbr : string
        A string of month abbreviations specifying the season of interest.
        For instance "mjja", "djf".
    season_name : string
        Name of the custom season.

    Returns:
    --------
    season_cube
        Cube with the data selected.
    """
    # First we need a monthly coordinate.
    try:
        iris.coord_categorisation.add_month(cube, "time")
    # If there already is a month coordinate, we pass.
    except ValueError:
        print("Cooridnate month alredy exist.")
    # Then we add the membership
    try:
        iris.coord_categorisation.add_season_membership(
            cube, "time", name=season_name, season=season_abbr
        )
    except ValueError:
        print("Season membership already exist.")

    # Create the constriant
    season_constraint = iris.Constraint(coord_values={season_name: True})

    season_cube = cube.extract(season_constraint)

    return season_cube


def compute_index(cube, index_name, client, spatial_average=False):
    """Compute a climate index based on the cube using Climix.

    Arguments:
    ----------
    cube : iris.Cube.cube

    index_name : string
        CF name of the index to compute.
    client : dask.distributed.Client
    spatial_average : bool, Default: False
        Whether to return a spatially averaged cube or not.

    Returns:
    --------
    index_cube

    """
    # Prepare the catalog
    index_catalog = load_metadata()

    # Select the index
    index = index_catalog.prepare_indices([index_name])[0]

    # Prepare the cube.
    # Can't have a "year" coordinate in climix
    try:
        cube.remove_coord("year")
    # If there is none, do nothing.
    except CoordinateNotFoundError:
        pass

    # Compute the index.
    index_cube = index([cube], client)

    # Do we want to compute the spatial average
    if spatial_average:
        index_cube = compute_spatial_average(index_cube)

    return index_cube


def compute_spatial_average(cube):
    """Compute the spatial average of a cube.

    Arguments:
    ----------
    cube : iris.cube.Cube

    Returns:
    --------
    Spatially averaged cube.
    """

    # Have to guess the bounds of the cube.
    try:
        cube.coord("grid_latitude").guess_bounds()
    # If we already have bounds, do nothing.
    except ValueError:
        pass
    try:
        cube.coord("grid_longitude").guess_bounds()
    except ValueError:
        pass

    # Have to remove latitude and longitude if there
    try:
        cube.remove_coord("latitude")
    # If we already have bounds, do nothing.
    except ValueError:
        pass
    try:
        cube.remove_coord("longitude")
    # If we already have bounds, do nothing.
    except ValueError:
        pass

    # Compute weights.
    area_weights = iris.analysis.cartography.area_weights(cube)

    # Collapse the dimensions.
    averaged_cube = cube.collapsed(
        ["grid_longitude", "grid_latitude"], iris.analysis.MEAN, weights=area_weights
    )

    return averaged_cube

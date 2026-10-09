import csv

from configuration import load_config_file


def write_orientations(
    path,
    rng,
    n_points=None,
    formations=None,
    extent=None,
    margin=None,
    azimuth_range=None,
    dip_range=None,
    polarity_values=None,
):
    settings = load_config_file("gempy/default.yaml")
    n_points = settings["orientation_points"] if n_points is None else n_points
    formations = settings["formations"] if formations is None else formations
    extent = settings["extent"] if extent is None else extent
    margin = settings["point_margin"] if margin is None else margin
    azimuth_range = settings["azimuth_range"] if azimuth_range is None else azimuth_range
    dip_range = settings["dip_range"] if dip_range is None else dip_range
    polarity_values = settings["polarity_values"] if polarity_values is None else polarity_values
    x = rng.uniform(extent[0] + margin, extent[1] - margin, n_points)
    y = rng.uniform(extent[2] + margin, extent[3] - margin, n_points)
    z = rng.uniform(extent[4] + margin, extent[5] - margin, n_points)

    azimuth = rng.uniform(*azimuth_range, n_points)
    dip = rng.uniform(*dip_range, n_points)

    polarity = rng.choice(polarity_values, n_points)
    formation = rng.choice(formations, n_points)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(
            ("X", "Y", "Z", "Azimuth", "Dip", "Polarity", "Formation")
        )
        writer.writerows(
            zip(x, y, z, azimuth, dip, polarity, formation)
        )


def write_surface_points(
    path,
    rng,
    n_points=None,
    formations=None,
    extent=None,
    margin=None,
):
    settings = load_config_file("gempy/default.yaml")
    n_points = settings["surface_points"] if n_points is None else n_points
    formations = settings["formations"] if formations is None else formations
    extent = settings["extent"] if extent is None else extent
    margin = settings["point_margin"] if margin is None else margin
    ids = range(n_points)

    x = rng.uniform(extent[0] + margin, extent[1] - margin, n_points)
    y = rng.uniform(extent[2] + margin, extent[3] - margin, n_points)
    z = rng.uniform(extent[4] + margin, extent[5] - margin, n_points)

    formation = rng.choice(formations, n_points)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(
            ("Id", "X", "Y", "Z", "Formation")
        )
        writer.writerows(
            zip(ids, x, y, z, formation)
        )

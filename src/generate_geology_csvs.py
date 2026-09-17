import csv

FORMATIONS = ["s1", "s2", "s3", "s4"]


def write_orientations(path, rng, n_points=3):
    x = rng.uniform(1, 99, n_points)
    y = rng.uniform(1, 99, n_points)
    z = rng.uniform(1, 9, n_points)

    azimuth = rng.uniform(0, 360, n_points)
    dip = rng.uniform(0, 90, n_points)

    polarity = rng.choice([-1, 1], n_points)
    formation = rng.choice(FORMATIONS, n_points)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ("X", "Y", "Z", "Azimuth", "Dip", "Polarity", "Formation")
        )
        writer.writerows(
            zip(x, y, z, azimuth, dip, polarity, formation)
        )


def write_surface_points(path, rng, n_points=100):
    ids = range(n_points)

    x = rng.uniform(1, 99, n_points)
    y = rng.uniform(1, 99, n_points)
    z = rng.uniform(1, 9, n_points)

    formation = rng.choice(FORMATIONS, n_points)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ("Id", "X", "Y", "Z", "Formation")
        )
        writer.writerows(
            zip(ids, x, y, z, formation)
        )

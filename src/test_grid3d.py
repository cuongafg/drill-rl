import numpy as np

from grid3d import Grid3D


def test_origin_and_depth_bounds():
    grid = Grid3D(nx=101, ny=101, nz=11)
    assert (grid.x_coords[0], grid.y_coords[0], grid.z_coords[0]) == (0.0, 0.0, 0.0)
    assert grid.z_coords.min() == -10.0
    assert grid.z_coords.max() == 0.0


def test_coordinate_conversion_round_trip():
    grid = Grid3D(nx=101, ny=101, nz=11)
    x = grid.x_coords[25]
    y = grid.y_coords[40]
    z = grid.z_coords[5]
    world = (x, y, z)
    idx = grid.world_to_index(*world)
    back = grid.index_to_world(*idx)
    assert idx == (25, 40, 5)
    assert back[0] == x
    assert back[1] == y
    assert back[2] == z


def test_generated_volume_shape():
    grid = Grid3D(nx=51, ny=61, nz=21)
    volume = grid.as_volume(dtype=np.float32)
    assert volume.shape == (51, 61, 21)


def test_index_bounds():
    grid = Grid3D(nx=5, ny=7, nz=9)
    assert grid.index_to_world(0, 0, 0) == (0.0, 0.0, 0.0)
    assert grid.index_to_world(4, 6, 8) == (100.0, 100.0, -10.0)

import numpy as np

from property_field import PropertyField3D


def test_field_dimensions_and_ranges():
    field = PropertyField3D(nx=31, ny=29, nz=21, seed=42)
    values = field.generate(seed=42)

    assert values.shape == (31, 29, 21)
    assert values.min() >= 0.0
    assert values.max() <= 300.0
    assert values[-1, -1, -1] > 200.0
    assert values[:, :, 0].mean() < 20.0


def test_coordinate_conversion_round_trip():
    field = PropertyField3D(nx=11, ny=13, nz=5, seed=7)
    x = field.grid.x_coords[5]
    y = field.grid.y_coords[8]
    z = field.grid.z_coords[2]
    idx = field.grid.world_to_index(x, y, z)
    world = field.grid.index_to_world(*idx)

    assert idx == (5, 8, 2)
    assert world[0] == x
    assert world[1] == y
    assert world[2] == z


def test_geology_changes_across_depth_and_bedrock_is_undulating():
    field = PropertyField3D(nx=41, ny=37, nz=21, seed=14)
    values = field.generate()

    assert field.bedrock_depth is not None
    assert float(field.bedrock_depth.std()) > 0.2
    assert not np.allclose(values[:, :, 0], values[:, :, 10])
    assert not np.allclose(values[:, :, 10], values[:, :, -1])
    assert float(values[:, :, 0].std()) > 0.2


def test_borehole_sampling_extracts_profile():
    field = PropertyField3D(nx=21, ny=21, nz=11, seed=11)
    field.generate(seed=11)
    x = float(field.grid.x_coords[4])
    y = float(field.grid.y_coords[6])
    profile = field.sample_borehole(x, y)

    assert profile.shape[0] == field.grid.nz
    assert profile.shape[1] == 2
    assert np.all(np.diff(profile[:, 0]) < 0.0)
    assert profile[0, 0] == 0.0
    assert profile[-1, 0] == -10.0
    assert np.allclose(profile[:, 1], field.field[4, 6, :])


def test_reproducibility():
    field_a = PropertyField3D(nx=17, ny=17, nz=7, seed=5)
    field_b = PropertyField3D(nx=17, ny=17, nz=7, seed=5)

    a = field_a.generate(seed=5)
    b = field_b.generate(seed=5)

    assert np.allclose(a, b)
    assert not np.allclose(a, field_a.generate(seed=6))


def test_save_load_round_trip(tmp_path):
    field = PropertyField3D(nx=9, ny=11, nz=5, property_range=(0.0, 100.0), seed=99)
    field.generate(seed=99)

    path = tmp_path / "property_field_roundtrip.npz"
    field.save(path)
    restored = PropertyField3D.load(path)

    assert restored.field.shape == field.field.shape
    assert np.allclose(restored.field, field.field)
    assert np.allclose(restored.bedrock_depth, field.bedrock_depth)
    assert restored.grid.z_coords[0] == 0.0
    assert restored.grid.z_coords[-1] == -10.0


def test_visualization_writes_3d_ground_truth_summary(tmp_path):
    field = PropertyField3D(nx=15, ny=13, nz=11, seed=33)
    field.generate()
    output_path = tmp_path / "property_field_geology.png"

    figure = field.visualize_3d(output_path)

    assert figure is not None
    assert output_path.is_file()

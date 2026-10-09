import numpy as np
from property_field import PropertyField3D


def test_different_seeds_generate_different_soil_fields():
    field_one = PropertyField3D(nx=17, ny=17, nz=7)
    field_two = PropertyField3D(nx=17, ny=17, nz=7)

    first = field_one.generate(seed=1)
    second = field_two.generate(seed=2)

    assert not np.array_equal(first, second)

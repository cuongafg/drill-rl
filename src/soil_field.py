import numpy as np

def generate_soil_field(seed=None):
    np.random.seed(seed)

    field = np.random.randint(
        1,
        6,
        size=(10, 100)
    )

    return field

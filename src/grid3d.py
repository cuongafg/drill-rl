import numpy as np

from configuration import load_config_file


class Grid3D:
    """Physical 3D Cartesian grid aligned with the geological site."""

    def __init__(
        self,
        nx=None,
        ny=None,
        nz=None,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
        z_min=None,
        z_max=None,
    ):
        defaults = load_config_file("grid/site.yaml")
        values = {
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "z_min": z_min,
            "z_max": z_max,
        }
        for name, value in values.items():
            if value is None:
                values[name] = defaults[name]

        self.nx = int(values["nx"])
        self.ny = int(values["ny"])
        self.nz = int(values["nz"])

        if self.nx < 1 or self.ny < 1 or self.nz < 1:
            raise ValueError("Grid dimensions must be positive integers.")

        self.x_coords = np.linspace(values["x_min"], values["x_max"], self.nx)
        self.y_coords = np.linspace(values["y_min"], values["y_max"], self.ny)
        self.z_coords = np.linspace(values["z_min"], values["z_max"], self.nz)

        self.x_min = float(self.x_coords[0])
        self.x_max = float(self.x_coords[-1])
        self.y_min = float(self.y_coords[0])
        self.y_max = float(self.y_coords[-1])
        self.z_min = float(self.z_coords[0])
        self.z_max = float(self.z_coords[-1])

    def index_to_world(self, ix, iy, iz):
        ix = int(ix)
        iy = int(iy)
        iz = int(iz)

        ix = max(0, min(self.nx - 1, ix))
        iy = max(0, min(self.ny - 1, iy))
        iz = max(0, min(self.nz - 1, iz))

        return (
            float(self.x_coords[ix]),
            float(self.y_coords[iy]),
            float(self.z_coords[iz]),
        )

    def world_to_index(self, x, y, z):
        x = float(x)
        y = float(y)
        z = float(z)

        if self.nx == 1:
            ix = 0
        else:
            dx = self.x_coords[1] - self.x_coords[0]
            ix = int(np.clip(np.rint((x - self.x_min) / dx), 0, self.nx - 1))

        if self.ny == 1:
            iy = 0
        else:
            dy = self.y_coords[1] - self.y_coords[0]
            iy = int(np.clip(np.rint((y - self.y_min) / dy), 0, self.ny - 1))

        if self.nz == 1:
            iz = 0
        else:
            dz = self.z_coords[1] - self.z_coords[0]
            iz = int(np.clip(np.rint((z - self.z_min) / dz), 0, self.nz - 1))

        return ix, iy, iz

    def as_volume(self, value=0.0, dtype=np.float32):
        return np.full((self.nx, self.ny, self.nz), value, dtype=dtype)

    def __repr__(self):
        return (
            f"Grid3D(nx={self.nx}, ny={self.ny}, nz={self.nz}, "
            f"x=[{self.x_coords[0]}, {self.x_coords[-1]}], "
            f"y=[{self.y_coords[0]}, {self.y_coords[-1]}], "
            f"z=[{self.z_coords[0]}, {self.z_coords[-1]}])"
        )

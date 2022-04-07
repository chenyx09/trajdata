import numpy as np


class MapPatch:
    def __init__(
        self, data: np.ndarray, rot_angle: float, crop_size: int, resolution: float
    ) -> None:
        self.data = data
        self.rot_angle = rot_angle
        self.crop_size = crop_size
        self.resolution = resolution

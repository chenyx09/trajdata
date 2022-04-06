from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor


class MapMetadata:
    def __init__(
        self,
        name: str,
        shape: Tuple[int, int],
        layers: List[str],
        world_to_img: np.ndarray,
    ) -> None:
        self.name: str = name
        self.shape: Tuple[int, int] = shape
        self.layers: List[str] = layers
        self.world_to_img: np.ndarray = world_to_img


class Map:
    def __init__(
        self,
        metadata: MapMetadata,
        data: np.ndarray,
    ) -> None:
        assert data.shape == metadata.shape
        self.metadata: MapMetadata = metadata
        self.data: np.ndarray = data

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @staticmethod
    def to_img(
        map_arr: Tensor, idx_groups: Tuple[List[int], List[int], List[int]]
    ) -> Tensor:
        return torch.stack(
            [
                torch.amax(map_arr[idx_groups[0]], dim=0),
                torch.amax(map_arr[idx_groups[1]], dim=0),
                torch.amax(map_arr[idx_groups[2]], dim=0),
            ],
            dim=-1,
        ).numpy()

from pathlib import Path
from typing import Dict

import numpy as np

from avdata.caching import SceneCache
from avdata.data_structures import SceneMetadata


class SimulationCache(SceneCache):
    def reset(self) -> None:
        raise NotImplementedError()

    def transform_data(self, **kwargs) -> None:
        raise NotImplementedError()

    def new_pos(pos_dict: Dict[str, np.ndarray]) -> None:
        raise NotImplementedError()

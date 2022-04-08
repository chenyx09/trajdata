from typing import Dict

import numpy as np

from avdata.caching.scene_cache import SceneCache


class SimulationCache(SceneCache):
    def reset(self) -> None:
        raise NotImplementedError()

    def transform_data(self, **kwargs) -> None:
        raise NotImplementedError()

    def append_state(pos_dict: Dict[str, np.ndarray]) -> None:
        raise NotImplementedError()

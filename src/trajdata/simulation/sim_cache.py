from typing import Dict, List, Optional, Tuple

import numpy as np

from trajdata.caching.scene_cache import SceneCache
from trajdata.simulation.sim_metrics import SimMetric
from trajdata.simulation.sim_stats import SimStatistic


class SimulationCache(SceneCache):
    def reset(self) -> None:
        raise NotImplementedError()

    def transform_data(self, **kwargs) -> None:
        raise NotImplementedError()

    def append_state(self, xyh_dict: Dict[str, np.ndarray]) -> None:
        raise NotImplementedError()

    def add_agents(self,agent_data:list) -> None:
        raise NotImplementedError

    def save_sim_scene(self) -> None:
        raise NotImplementedError()

    def calculate_metrics(
        self, metrics: List[SimMetric], ts_range: Optional[Tuple[int, int]] = None
    ) -> Dict[str, Dict[str, float]]:
        raise NotImplementedError()

    def calculate_stats(
        self, stats: List[SimStatistic], ts_range: Optional[Tuple[int, int]] = None
    ) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        raise NotImplementedError()

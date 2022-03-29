from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from avdata.data_structures.agent import AgentMetadata
from avdata.data_structures.scene_metadata import SceneMetadata


class SceneCache:
    def __init__(
        self, cache_path: Path, scene_info: SceneMetadata, scene_ts: int
    ) -> None:
        """
        Creates and prepares the cache for online data loading.
        """
        self.path = cache_path
        self.scene_info = scene_info
        self.dt = scene_info.dt
        self.scene_ts = scene_ts

        # Ensuring the scene cache folder exists
        self.scene_dir: Path = (
            self.path / self.scene_info.env_name / self.scene_info.name
        )
        self.scene_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def save_agent_data(
        agent_data: pd.DataFrame,
        cache_path: Path,
        scene_info: SceneMetadata,
    ) -> None:
        raise NotImplementedError()

    def get_value(self, agent_id: str, scene_ts: int, attribute: str) -> float:
        """
        Get a single attribute value for an agent at a timestep.
        """
        raise NotImplementedError()

    def get_state(self, agent_id: str, scene_ts: int) -> np.ndarray:
        """
        Get an agent's state at a specific timestep.
        """
        raise NotImplementedError()

    def transform_data(self, **kwargs) -> None:
        """
        Transform the data before accessing it later, e.g., to make the mean zero or rotate the scene around an agent.
        This can either be done in this function call or just stored for later lazy application.
        """
        raise NotImplementedError()

    def get_agent_history(
        self,
        agent_info: AgentMetadata,
        scene_ts: int,
        history_sec: Tuple[Optional[float], Optional[float]],
    ) -> pd.DataFrame:
        raise NotImplementedError()

    def get_agent_future(
        self,
        agent_info: AgentMetadata,
        scene_ts: int,
        future_sec: Tuple[Optional[float], Optional[float]],
    ) -> pd.DataFrame:
        raise NotImplementedError()

    def get_positions_at(
        self, scene_ts: int, agents: List[AgentMetadata]
    ) -> np.ndarray:
        raise NotImplementedError()

    def get_agents_history(
        self,
        scene_ts: int,
        agents: List[AgentMetadata],
        history_sec: Tuple[Optional[float], Optional[float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()

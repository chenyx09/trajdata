from collections import defaultdict
from copy import deepcopy
from math import floor
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from avdata.caching.df_cache import DataFrameCache
from avdata.data_structures.agent import AgentMetadata
from avdata.data_structures.scene_metadata import SceneMetadata
from avdata.simulation.sim_cache import SimulationCache


class SimulationDataFrameCache(DataFrameCache, SimulationCache):
    def __init__(
        self, cache_path: Path, scene_info: SceneMetadata, scene_ts: int
    ) -> None:
        super().__init__(cache_path, scene_info, scene_ts)
        agent_names: List[str] = [agent.name for agent in scene_info.agents]
        in_index: np.ndarray = self.scene_data_df.index.isin(agent_names, level=0)
        self.persistent_data_df: pd.DataFrame = self.scene_data_df.iloc[in_index].copy()

        self.scene_data_df = self.persistent_data_df.copy()
        self.index_dict: Dict[Tuple[str, int], int] = {
            val: idx for idx, val in enumerate(self.scene_data_df.index)
        }

    def reset(self) -> None:
        self.scene_data_df = self.persistent_data_df.copy()
        self.index_dict: Dict[Tuple[str, int], int] = {
            val: idx for idx, val in enumerate(self.scene_data_df.index)
        }

    def get_agent_future(
        self,
        agent_info: AgentMetadata,
        scene_ts: int,
        future_sec: Tuple[Optional[float], Optional[float]],
    ) -> pd.DataFrame:
        if scene_ts >= agent_info.last_timestep:
            # Returning an empty DataFrame with the correct
            # columns.
            return self.scene_data_df.iloc[0:0]

        return super().get_agent_future(agent_info, scene_ts, future_sec)

    def append_state(self, xyh_dict: Dict[str, np.ndarray]) -> None:
        self.scene_ts += 1

        sim_dict: Dict[str, List[Union[str, float, int]]] = defaultdict(list)
        for agent, state in xyh_dict.items():
            prev_state = self.get_state(agent, self.scene_ts - 1)

            sim_dict["agent_id"].append(agent)
            sim_dict["scene_ts"].append(self.scene_ts)

            sim_dict["x"].append(state[0])
            sim_dict["y"].append(state[1])

            vx: float = (state[0] - prev_state[0]) / self.scene_info.dt
            vy: float = (state[1] - prev_state[1]) / self.scene_info.dt
            sim_dict["vx"].append(vx)
            sim_dict["vy"].append(vy)

            ax: float = (vx - prev_state[2]) / self.scene_info.dt
            ay: float = (vy - prev_state[3]) / self.scene_info.dt
            sim_dict["ax"].append(ax)
            sim_dict["ay"].append(ay)

            sim_dict["heading"].append(state[2])

        sim_step_df = pd.DataFrame(sim_dict)
        sim_step_df.set_index(["agent_id", "scene_ts"], inplace=True)
        if self.scene_ts < self.scene_info.length_timesteps:
            self.persistent_data_df.drop(index=self.scene_ts, level=1, inplace=True)
            
        self.persistent_data_df = pd.concat([self.persistent_data_df, sim_step_df])
        self.persistent_data_df.sort_index(inplace=True)
        self.reset()

    def save_sim_scene(self, sim_scene_info: SceneMetadata) -> None:
        history_idxs = (
            self.persistent_data_df.index.get_level_values("scene_ts") <= self.scene_ts
        )
        DataFrameCache.save_agent_data(
            self.persistent_data_df[history_idxs], self.path, sim_scene_info
        )

from collections import defaultdict
from copy import deepcopy
from math import floor
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from avdata.augmentation.augmentation import Augmentation

from avdata.caching.df_cache import DataFrameCache
from avdata.data_structures.agent import AgentMetadata
from avdata.data_structures.scene_metadata import SceneMetadata
from avdata.simulation.sim_cache import SimulationCache


class SimulationDataFrameCache(DataFrameCache, SimulationCache):
    def __init__(
        self, cache_path: Path, scene_info: SceneMetadata, scene_ts: int, augmentations: Optional[List[Augmentation]] = None,
    ) -> None:
        super().__init__(cache_path, scene_info, scene_ts, augmentations)

        agent_names: List[str] = [agent.name for agent in scene_info.agents]
        in_index: np.ndarray = self.scene_data_df.index.isin(agent_names, level=0)
        self.scene_data_df: pd.DataFrame = self.scene_data_df.iloc[in_index].copy()
        self.index_dict: Dict[Tuple[str, int], int] = {
            val: idx for idx, val in enumerate(self.scene_data_df.index)
        }

        # Important to first prune self.scene_data_df before interpolation (since it
        # will use the agents list from the scene_info object which was modified earlier
        # in the SimulationScene init.
        if scene_info.env_metadata.dt != scene_info.dt:
            self.interpolate_data(scene_info.dt)

        self.persistent_data_df: pd.DataFrame = self.scene_data_df.copy()

    def reset(self) -> None:
        self.scene_data_df = self.persistent_data_df.copy()
        self.index_dict: Dict[Tuple[str, int], int] = {
            val: idx for idx, val in enumerate(self.scene_data_df.index)
        }
        self._get_and_reorder_col_idxs()

    def get_agent_future(
        self,
        agent_info: AgentMetadata,
        scene_ts: int,
        future_sec: Tuple[Optional[float], Optional[float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if scene_ts >= agent_info.last_timestep:
            # Returning an empty DataFrame with the correct
            # columns. 3 = Extent size.
            return np.zeros((0, self.state_dim)), np.zeros((0, 3))

        return super().get_agent_future(agent_info, scene_ts, future_sec)

    def get_agents_future(
        self,
        scene_ts: int,
        agents: List[AgentMetadata],
        future_sec: Tuple[Optional[float], Optional[float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        last_timesteps = np.array(
            [agent.last_timestep for agent in agents], dtype=np.long
        )

        if np.all(np.greater(scene_ts, last_timesteps)):
            return (
                [np.zeros((0, self.state_dim)) for agent in agents],
                [np.zeros((0, 3)) for agent in agents],  # 3 = Extent size.
                np.zeros_like(last_timesteps),
            )

        return super().get_agents_future(scene_ts, agents, future_sec)

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

            if self.extent_cols:
                sim_dict["length"].append(
                    self.get_value(agent, self.scene_ts - 1, "length")
                )
                sim_dict["width"].append(
                    self.get_value(agent, self.scene_ts - 1, "width")
                )
                sim_dict["height"].append(
                    self.get_value(agent, self.scene_ts - 1, "height")
                )

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

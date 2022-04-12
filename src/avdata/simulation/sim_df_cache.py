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
        self.og_index_dict = deepcopy(self.index_dict)
        self.og_data_df = self.scene_data_df.copy()
        
        history_idxs = self.scene_data_df.index.get_level_values("scene_ts") <= scene_ts
        self.persistent_data_df: pd.DataFrame = self.scene_data_df[history_idxs].copy()

        self.scene_data_df = self.persistent_data_df.copy()
        self.index_dict: Dict[Tuple[str, int], int] = {
            val: idx for idx, val in enumerate(self.scene_data_df.index)
        }
        
        self._data_transf_mean: np.ndarray = np.zeros_like(self.scene_data_df.iloc[0].to_numpy())
        self._data_transf_rot: np.ndarray = np.eye(2)

    def reset(self) -> None:
        self.scene_data_df = self.persistent_data_df.copy()
        self.index_dict: Dict[Tuple[str, int], int] = {
            val: idx for idx, val in enumerate(self.scene_data_df.index)
        }

    def transform_data(self, **kwargs) -> None:
        """
        Only difference to the original df_cache is that we don't touch the heading column.
        """
        if "shift_mean_to" in kwargs:
            # This standardizes the scene to be relative to the state passed in
            self._data_transf_mean = kwargs["shift_mean_to"]
        
        if "rotate_by" in kwargs:
            # This rotates the scene so that the predicted agent's current heading aligns with the x-axis
            agent_heading: float = kwargs["rotate_by"]
            self._data_transf_rot = np.array(
                [
                    [np.cos(agent_heading), -np.sin(agent_heading)],
                    [np.sin(agent_heading), np.cos(agent_heading)],
                ]
            )
            
        if "sincos_heading" in kwargs:
            self._data_transf_sincos = True
        
        super().transform_data(**kwargs)

    def get_agent_future(
        self,
        agent_info: AgentMetadata,
        scene_ts: int,
        future_sec: Tuple[Optional[float], Optional[float]],
    ) -> pd.DataFrame:
        if scene_ts >= agent_info.last_timestep:
            # Returning an empty DataFrame with the correct
            # number of columns.
            return self.scene_data_df.iloc[0 : 0]
        
        first_index_incl: int = self.og_index_dict[(agent_info.name, scene_ts + 1)]
        last_index_incl: int
        if future_sec[1] is not None:
            max_future = floor(future_sec[1] / self.dt)
            last_index_incl = self.og_index_dict[
                (agent_info.name, min(scene_ts + max_future, agent_info.last_timestep))
            ]
        else:
            last_index_incl = self.og_index_dict[
                (agent_info.name, agent_info.last_timestep)
            ]

        future_df: pd.DataFrame = self.og_data_df.iloc[first_index_incl : last_index_incl + 1].copy()
        
        future_df -= self._data_transf_mean
        
        future_df.iloc[:, self.pos_cols] = (
            future_df.iloc[:, self.pos_cols].to_numpy() @ self._data_transf_rot
        )
        future_df.iloc[:, self.vel_cols] = (
            future_df.iloc[:, self.vel_cols].to_numpy() @ self._data_transf_rot
        )
        future_df.iloc[:, self.acc_cols] = (
            future_df.iloc[:, self.acc_cols].to_numpy() @ self._data_transf_rot
        )
        
        if self._data_transf_sincos:
            future_df["sin_heading"] = np.sin(future_df["heading"])
            future_df["cos_heading"] = np.cos(future_df["heading"])
            future_df.drop(columns=["heading"], inplace=True)
        
        return future_df

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
        self.persistent_data_df = pd.concat([self.persistent_data_df, sim_step_df])
        self.persistent_data_df.sort_index(inplace=True)
        self.reset()

    def save_sim_scene(self, sim_scene_info: SceneMetadata) -> None:
        DataFrameCache.save_agent_data(
            self.persistent_data_df, self.path, sim_scene_info
        )

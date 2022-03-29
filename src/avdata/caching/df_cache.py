import pickle
from math import floor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from avdata.caching.scene_cache import SceneCache
from avdata.data_structures.agent import AgentMetadata
from avdata.data_structures.scene_metadata import SceneMetadata
from avdata.utils import arr_utils


class DataFrameCache(SceneCache):
    def __init__(
        self, cache_path: Path, scene_info: SceneMetadata, scene_ts: int
    ) -> None:
        """
        Data cache primarily based on pandas DataFrames,
        with Feather for fast agent data serialization
        and pickle for miscellaneous supporting objects.
        """
        super().__init__(cache_path, scene_info, scene_ts)

        self.agent_data_path: Path = self.scene_dir / "agent_data.feather"

        self._load_agent_data()
        self._compute_col_idxs()

    def _compute_col_idxs(self) -> None:
        self.column_dict: Dict[str, int] = {
            val: idx for idx, val in enumerate(self.scene_data_df.columns)
        }

        self.pos_cols = [self.column_dict["x"], self.column_dict["y"]]
        self.vel_cols = [self.column_dict["vx"], self.column_dict["vy"]]
        self.acc_cols = [self.column_dict["ax"], self.column_dict["ay"]]

    def _load_agent_data(self) -> pd.DataFrame:
        self.scene_data_df: pd.DataFrame = pd.read_feather(
            self.agent_data_path, use_threads=False
        ).set_index(["agent_id", "scene_ts"])

        with open(self.scene_dir / "scene_index.pkl", "rb") as f:
            self.index_dict: Dict[Tuple[str, int], int] = pickle.load(f)

    @staticmethod
    def save_agent_data(
        agent_data: pd.DataFrame,
        cache_path: Path,
        scene_info: SceneMetadata,
    ) -> None:
        scene_cache_dir: Path = cache_path / scene_info.env_name / scene_info.name
        scene_cache_dir.mkdir(parents=True, exist_ok=True)

        index_dict: Dict[Tuple[str, int], int] = {
            val: idx for idx, val in enumerate(agent_data.index)
        }
        with open(scene_cache_dir / "scene_index.pkl", "wb") as f:
            pickle.dump(index_dict, f)

        agent_data.reset_index().to_feather(scene_cache_dir / "agent_data.feather")

    def get_value(self, agent_id: str, scene_ts: int, attribute: str) -> float:
        return self.scene_data_df.iat[
            self.index_dict[(agent_id, scene_ts)], self.column_dict[attribute]
        ]

    def get_state(self, agent_id: str, scene_ts: int) -> np.ndarray:
        return self.scene_data_df.iloc[self.index_dict[(agent_id, scene_ts)]]

    def transform_data(self, **kwargs) -> None:
        if "shift_mean_to" in kwargs:
            # This standardizes the scene to be relative to the agent being predicted
            self.scene_data_df -= kwargs["shift_mean_to"]

        if "rotate_by" in kwargs:
            # This rotates the scene so that the predicted agent's current heading aligns with the x-axis
            agent_heading: float = kwargs["rotate_by"]
            self.rot_matrix: np.ndarray = np.array(
                [
                    [np.cos(agent_heading), -np.sin(agent_heading)],
                    [np.sin(agent_heading), np.cos(agent_heading)],
                ]
            )
            self.scene_data_df.iloc[:, self.pos_cols] = (
                self.scene_data_df.iloc[:, self.pos_cols].to_numpy() @ self.rot_matrix
            )
            self.scene_data_df.iloc[:, self.vel_cols] = (
                self.scene_data_df.iloc[:, self.vel_cols].to_numpy() @ self.rot_matrix
            )
            self.scene_data_df.iloc[:, self.acc_cols] = (
                self.scene_data_df.iloc[:, self.acc_cols].to_numpy() @ self.rot_matrix
            )

        if "sincos_heading" in kwargs:
            self.scene_data_df["sin_heading"] = np.sin(self.scene_data_df["heading"])
            self.scene_data_df["cos_heading"] = np.cos(self.scene_data_df["heading"])
            self.scene_data_df.drop(columns=["heading"], inplace=True)
            self._compute_col_idxs()

    def get_agent_history(
        self,
        agent_info: AgentMetadata,
        scene_ts: int,
        history_sec: Tuple[Optional[float], Optional[float]],
    ) -> pd.DataFrame:
        # We don't have to check the mins here because our data_index filtering in dataset.py already
        # took care of it.
        first_index_incl: int
        last_index_incl: int = self.index_dict[(agent_info.name, scene_ts)]
        if history_sec[1] is not None:
            max_history: int = floor(history_sec[1] / self.dt)
            first_index_incl = self.index_dict[
                (
                    agent_info.name,
                    max(scene_ts - max_history, agent_info.first_timestep),
                )
            ]
        else:
            first_index_incl = self.index_dict[
                (agent_info.name, agent_info.first_timestep)
            ]

        return self.scene_data_df.iloc[first_index_incl : last_index_incl + 1]

    def get_agent_future(
        self,
        agent_info: AgentMetadata,
        scene_ts: int,
        future_sec: Tuple[Optional[float], Optional[float]],
    ) -> pd.DataFrame:
        # We don't have to check the mins here because our data_index filtering in dataset.py already
        # took care of it.
        first_index_incl: int = self.index_dict[(agent_info.name, scene_ts + 1)]
        last_index_incl: int
        if future_sec[1] is not None:
            max_future = floor(future_sec[1] / self.dt)
            last_index_incl = self.index_dict[
                (agent_info.name, min(scene_ts + max_future, agent_info.last_timestep))
            ]
        else:
            last_index_incl = self.index_dict[
                (agent_info.name, agent_info.last_timestep)
            ]

        return self.scene_data_df.iloc[first_index_incl : last_index_incl + 1]

    def get_positions_at(
        self, scene_ts: int, agents: List[AgentMetadata]
    ) -> np.ndarray:
        rows = [self.index_dict[(agent.name, scene_ts)] for agent in agents]
        return self.scene_data_df.iloc[rows, self.pos_cols].to_numpy()

    def get_agents_history(
        self,
        scene_ts: int,
        agents: List[AgentMetadata],
        history_sec: Tuple[Optional[float], Optional[float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        first_timesteps = np.array(
            [agent.first_timestep for agent in agents], dtype=np.long
        )
        if history_sec[1] is not None:
            max_history: int = floor(history_sec[1] / self.dt)
            first_timesteps = np.maximum(scene_ts - max_history, first_timesteps)

        first_index_incl: np.ndarray = np.array(
            [
                self.index_dict[(agent.name, first_timesteps[idx])]
                for idx, agent in enumerate(agents)
            ],
            dtype=np.long,
        )
        last_index_incl: np.ndarray = np.array(
            [self.index_dict[(agent.name, scene_ts)] for agent in agents], dtype=np.long
        )

        concat_idxs = arr_utils.vrange(first_index_incl, last_index_incl + 1)
        return (
            self.scene_data_df.iloc[concat_idxs, :].to_numpy(),
            last_index_incl - first_index_incl + 1,
        )

from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from avdata.caching.df_cache import DataFrameCache
from avdata.data_structures.scene_metadata import SceneMetadata
from avdata.simulation.sim_cache import SimulationCache


class SimulationDataFrameCache(DataFrameCache, SimulationCache):
    def __init__(
        self, cache_path: Path, scene_info: SceneMetadata, scene_ts: int
    ) -> None:
        super().__init__(cache_path, scene_info, scene_ts)
        history_idxs = self.scene_data_df.index.get_level_values("scene_ts") <= scene_ts
        self.original_data_df: pd.DataFrame = self.scene_data_df[history_idxs].copy()

        self.scene_data_df = self.original_data_df.copy()
        self.index_dict: Dict[Tuple[str, int], int] = defaultdict(int)
        self.index_dict.update(
            {val: idx for idx, val in enumerate(self.scene_data_df.index)}
        )

    def reset(self) -> None:
        self.scene_data_df = self.original_data_df.copy()

    def transform_data(self, **kwargs) -> None:
        """
        Only difference to the original df_cache is that we don't touch the heading column.
        """

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

    def append_state(pos_dict: Dict[str, np.ndarray]) -> None:
        pass

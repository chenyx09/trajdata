import pickle
from math import ceil, floor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import dill
import kornia
import numpy as np
import pandas as pd
import torch
import zarr

from avdata.caching.scene_cache import SceneCache
from avdata.data_structures.agent import AgentMetadata
from avdata.data_structures.map import Map, MapMetadata
from avdata.data_structures.map_patch import MapPatch
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

    # AGENT STATE DATA
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
        return self.scene_data_df.iloc[self.index_dict[(agent_id, scene_ts)]].to_numpy()

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

    # MAPS
    @staticmethod
    def is_map_cached(cache_path: Path, env_name: str, map_name: str) -> bool:
        maps_path: Path = cache_path / env_name / "maps"
        metadata_file: Path = maps_path / f"{map_name}_metadata.dill"
        map_file: Path = maps_path / f"{map_name}.zarr"
        return maps_path.is_dir() and metadata_file.is_file() and map_file.is_file()

    @staticmethod
    def cache_map(cache_path: Path, map_obj: Map, env_name: str) -> None:
        maps_path: Path = cache_path / env_name / "maps"
        maps_path.mkdir(parents=True, exist_ok=True)

        metadata_file: Path = maps_path / f"{map_obj.metadata.name}_metadata.dill"
        with open(metadata_file, "wb") as f:
            dill.dump(map_obj.metadata, f)

        map_file: Path = maps_path / f"{map_obj.metadata.name}.zarr"
        zarr.save(map_file, map_obj.data)

    @staticmethod
    def cache_map_layers(
        cache_path: Path,
        map_info: MapMetadata,
        layer_fn: Callable[[str], np.ndarray],
        env_name: str,
    ) -> None:
        maps_path: Path = cache_path / env_name / "maps"
        maps_path.mkdir(parents=True, exist_ok=True)

        map_file: Path = maps_path / f"{map_info.name}.zarr"
        disk_data = zarr.open_array(map_file, mode="w", shape=map_info.shape)
        for idx, layer_name in enumerate(map_info.layers):
            disk_data[idx] = layer_fn(layer_name)

        metadata_file: Path = maps_path / f"{map_info.name}_metadata.dill"
        with open(metadata_file, "wb") as f:
            dill.dump(map_info, f)

    def pad_map_patch(
        self,
        patch: np.ndarray,
        # top, bot, left, right
        patch_sides: Tuple[int, int, int, int],
        patch_size: int,
        map_dims: Tuple[int, int],
    ) -> np.ndarray:
        if patch.shape[-2:] == (patch_size, patch_size):
            return patch

        top, bot, left, right = patch_sides
        height, width = map_dims

        pad_top, pad_bot, pad_left, pad_right = 0, 0, 0, 0
        if top < 0:
            pad_top = 0 - top
        if bot >= height:
            pad_bot = bot - height
        if left < 0:
            pad_left = 0 - left
        if right >= width:
            pad_right = right - width

        return np.pad(patch, [(0, 0), (pad_top, pad_bot), (pad_left, pad_right)])

    def load_map_patch(
        self,
        world_x: float,
        world_y: float,
        desired_patch_size: int,
        resolution: int,
        rot_pad_factor: float = 1.0,
    ) -> Tuple[np.ndarray, MapMetadata]:
        maps_path: Path = self.path / self.scene_info.env_name / "maps"

        metadata_file: Path = maps_path / f"{self.scene_info.location}_metadata.dill"
        with open(metadata_file, "rb") as f:
            map_info: MapMetadata = dill.load(f)

        map_coords: np.ndarray = map_info.resolution * np.array([world_x, world_y])
        map_x, map_y = round(map_coords[0].item()), round(map_coords[1].item())

        data_patch_size: int = ceil(
            desired_patch_size * map_info.resolution / resolution
        )
        data_with_rot_pad_size: int = ceil(rot_pad_factor * data_patch_size)

        map_file: Path = maps_path / f"{map_info.name}.zarr"
        disk_data = zarr.open_array(map_file, mode="r")

        top: int = map_y - data_with_rot_pad_size // 2
        bot: int = map_y + data_with_rot_pad_size // 2
        left: int = map_x - data_with_rot_pad_size // 2
        right: int = map_x + data_with_rot_pad_size // 2

        data_patch: np.ndarray = self.pad_map_patch(
            disk_data[
                ...,
                max(top, 0) : min(bot, disk_data.shape[1]),
                max(left, 0) : min(right, disk_data.shape[2]),
            ],
            (top, bot, left, right),
            data_with_rot_pad_size,
            disk_data.shape[-2:],
        )

        if desired_patch_size == data_patch_size:
            return data_patch
        else:
            rescaled_patch: np.ndarray = (
                kornia.geometry.rescale(
                    torch.from_numpy(data_patch).unsqueeze(0),
                    desired_patch_size / data_patch_size,
                    # Default align_corners value, just putting it to remove warnings
                    align_corners=False,
                    antialias=True,
                )
                .squeeze(0)
                .numpy()
            )

            return rescaled_patch

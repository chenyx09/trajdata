from collections import defaultdict
from math import ceil, floor
from multiprocessing import Manager
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import dill
import numpy as np

# Lyft Level 5
from l5kit.data import ChunkedDataset
from nuscenes.map_expansion.map_api import NuScenesMap

# NuScenes
from nuscenes.nuscenes import NuScenes
from torch.utils.data import Dataset
from tqdm import tqdm

from unified_dataset.data_structures import (
    AgentBatchElement,
    AgentMetadata,
    AgentType,
    EnvMetadata,
    SceneBatchElement,
    SceneMetadata,
    SceneTime,
    agent_collate_fn,
    scene_collate_fn,
)
from unified_dataset.utils import env_utils, lyft_utils, nusc_utils, string_utils


class UnifiedDataset(Dataset):
    def __init__(
        self,
        datasets: Optional[List[str]] = None,
        centric: str = "agent",
        history_sec: Tuple[Optional[float], Optional[float]] = (
            None,
            None,
        ),  # Both inclusive
        future_sec: Tuple[Optional[float], Optional[float]] = (
            None,
            None,
        ),  # Both inclusive
        agent_interaction_distances: Dict[
            Tuple[AgentType, AgentType], float
        ] = defaultdict(lambda: np.inf),
        incl_robot_future: bool = False,
        incl_map: bool = False,
        only_types: Optional[List[AgentType]] = None,
        no_types: Optional[List[AgentType]] = None,
        data_dirs: Dict[str, str] = {
            "nusc": "~/datasets/nuScenes",
            "nusc_mini": "~/datasets/nuScenes",
            "lyft_sample": "~/datasets/lyft/scenes/sample.zarr",
        },
        cache_location: str = "~/.unified_data_cache",
        rebuild_cache: bool = False,
    ) -> None:
        self.centric = centric

        if self.centric == "agent":
            self.collate_fn = agent_collate_fn
        elif self.centric == "scene":
            self.collate_fn = scene_collate_fn

        self.rebuild_cache = rebuild_cache
        self.cache_dir = Path(cache_location).expanduser().resolve()
        if not self.cache_dir.is_dir():
            self.cache_dir.mkdir()

        self.history_sec = history_sec
        self.future_sec = future_sec
        self.agent_interaction_distances = agent_interaction_distances
        self.incl_robot_future = incl_robot_future
        self.incl_map = incl_map
        self.only_types = None if only_types is None else set(only_types)
        self.no_types = None if no_types is None else set(no_types)

        self.envs_dict: Dict[str, EnvMetadata] = env_utils.get_env_metadata(data_dirs)

        self.all_components = list()
        for env in self.envs_dict.values():
            self.all_components += env.components

        matching_datasets: List[Tuple[str, ...]] = self.get_matching_datasets(datasets)
        print(
            "Loading data for matched datasets:",
            string_utils.pretty_string_tuples(matching_datasets),
            flush=True,
        )

        if any("nusc" in dataset_tuple for dataset_tuple in matching_datasets):
            print("Loading nuScenes dataset...", flush=True)
            self.nusc_obj: NuScenes = NuScenes(
                version="v1.0-trainval", dataroot=self.envs_dict["nusc"].data_dir
            )

        if any("nusc_mini" in dataset_tuple for dataset_tuple in matching_datasets):
            print("Loading nuScenes mini dataset...", flush=True)
            self.nusc_mini_obj: NuScenes = NuScenes(
                version="v1.0-mini", dataroot=self.envs_dict["nusc_mini"].data_dir
            )

        if any("lyft_sample" in dataset_tuple for dataset_tuple in matching_datasets):
            print("Loading lyft sample dataset...", flush=True)
            self.lyft_sample_obj: ChunkedDataset = ChunkedDataset(
                str(self.envs_dict["lyft_sample"].data_dir)
            ).open()

        self.scene_index: List[SceneMetadata] = self.create_scene_metadata(
            matching_datasets
        )
        print(self.scene_index)

        self.data_index = list()
        if self.centric == "scene":
            for scene_info in self.scene_index:
                for ts in range(scene_info.length_timesteps):
                    # This is where we remove scene timesteps that would have no remaining agents after filtering.
                    if self.no_types is not None and all(
                        agent_info.type in self.no_types
                        for agent_info in scene_info.agent_presence[ts]
                    ):
                        continue
                    elif self.only_types is not None and all(
                        agent_info.type not in self.only_types
                        for agent_info in scene_info.agent_presence[ts]
                    ):
                        continue

                    if all(
                        not self.satisfies_times(agent_info, ts, scene_info)
                        for agent_info in scene_info.agent_presence[ts]
                    ):
                        # Ignore this datum if no agent in the scene satisfies our time requirements.
                        continue

                    self.data_index.append((scene_info.env_name, scene_info.name, ts))

        elif self.centric == "agent":
            for scene_info in self.scene_index:
                for ts in range(scene_info.length_timesteps):
                    for agent_info in scene_info.agent_presence[ts]:
                        # Ignore this datum if the agent does not satisfy our time requirements.
                        if not self.satisfies_times(agent_info, ts, scene_info):
                            continue

                        # This is where we remove agents that do not match our filters.
                        if (
                            self.no_types is not None
                            and agent_info.type in self.no_types
                        ):
                            continue
                        elif (
                            self.only_types is not None
                            and agent_info.type not in self.only_types
                        ):
                            continue

                        self.data_index.append(
                            (scene_info.env_name, scene_info.name, ts, agent_info.name)
                        )

        manager = Manager()
        self.data_index = manager.list(self.data_index)
        self.data_len: int = len(self.data_index)

    def get_matching_datasets(
        self, queries: Optional[List[str]]
    ) -> List[Tuple[str, ...]]:
        if queries is None:
            return self.all_components

        dataset_tuples = [set(data.split("-")) for data in queries]

        matching_datasets = list()
        for dataset_tuple in dataset_tuples:
            for dataset_component in self.all_components:
                if dataset_tuple.issubset(dataset_component):
                    matching_datasets.append(dataset_component)

        return matching_datasets

    def create_scene_metadata(
        self, datasets: List[Tuple[str, ...]]
    ) -> List[SceneMetadata]:
        scenes_list: List[SceneMetadata] = list()
        for dataset_tuple in tqdm(datasets, desc="Loading Scene Metadata"):
            if "nusc_mini" in dataset_tuple:
                scenes_list += nusc_utils.get_matching_scenes(
                    self.nusc_mini_obj, self.envs_dict["nusc_mini"], dataset_tuple
                )

            if "lyft_sample" in dataset_tuple:
                scenes_list += lyft_utils.get_matching_scenes(
                    self.lyft_sample_obj, self.envs_dict["lyft_sample"], dataset_tuple
                )

        self.calculate_agent_presence(scenes_list)
        return scenes_list

    def calculate_agent_presence(self, scenes: List[SceneMetadata]) -> None:
        scene_info: SceneMetadata
        for scene_info in tqdm(scenes, desc="Calculating Agent Presence"):
            cache_scene_dir: Path = (
                self.cache_dir / scene_info.env_name / scene_info.name
            )
            if not cache_scene_dir.is_dir():
                cache_scene_dir.mkdir(parents=True)

            scene_file: Path = cache_scene_dir / "scene_metadata.dill"
            if scene_file.is_file() and not self.rebuild_cache:
                with open(scene_file, "rb") as f:
                    cached_scene_info: SceneMetadata = dill.load(f)

                scene_info.update_agent_presence(cached_scene_info.agent_presence)
                continue

            if scene_info.env_name == "nusc_mini":
                agent_presence = nusc_utils.calc_agent_presence(
                    scene_info=scene_info,
                    nusc_obj=self.nusc_mini_obj,
                    cache_scene_dir=cache_scene_dir,
                    rebuild_cache=self.rebuild_cache,
                )
            elif scene_info.env_name == "lyft_sample":
                agent_presence = lyft_utils.calc_agent_presence(
                    scene_info=scene_info,
                    lyft_obj=self.lyft_sample_obj,
                    cache_scene_dir=cache_scene_dir,
                    rebuild_cache=self.rebuild_cache,
                )

            scene_info.update_agent_presence(agent_presence)

            with open(scene_file, "wb") as f:
                dill.dump(scene_info, f)

    def satisfies_times(
        self, agent_info: AgentMetadata, ts: int, scene_info: SceneMetadata
    ) -> bool:
        dt = scene_info.dt

        if self.history_sec[0] is not None:
            min_history = ceil(self.history_sec[0] / dt)
            agent_history_satisfies = ts - agent_info.first_timestep >= min_history
        else:
            agent_history_satisfies = True

        if self.future_sec[0] is not None:
            min_future = ceil(self.future_sec[0] / dt)
            agent_future_satisfies = agent_info.last_timestep - ts >= min_future
        else:
            agent_future_satisfies = True

        return agent_history_satisfies and agent_future_satisfies

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, idx: int) -> AgentBatchElement:
        if self.centric == "scene":
            env_name, scene_name, ts = self.data_index[idx]
        elif self.centric == "agent":
            env_name, scene_name, ts, agent_id = self.data_index[idx]

        scene_cache_dir: Path = self.cache_dir / env_name / scene_name
        scene_file: Path = scene_cache_dir / "scene_metadata.dill"
        with open(scene_file, "rb") as f:
            scene_info: SceneMetadata = dill.load(f)

        scene_time: SceneTime = SceneTime.from_cache(
            scene_info,
            ts,
            scene_cache_dir,
            only_types=self.only_types,
            no_types=self.no_types,
        )

        if self.centric == "scene":
            return SceneBatchElement(scene_time, self.history_sec, self.future_sec)
        elif self.centric == "agent":
            return AgentBatchElement(
                idx,
                scene_time,
                agent_id,
                self.history_sec,
                self.future_sec,
                self.agent_interaction_distances,
                self.incl_robot_future,
                self.incl_map,
            )

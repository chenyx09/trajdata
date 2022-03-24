from collections import defaultdict
from itertools import chain
from multiprocessing import Manager
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Type

import dill
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from avdata import filtering
from avdata.caching import BaseCache, SQLiteCache
from avdata.data_structures import (
    AgentBatchElement,
    AgentType,
    SceneBatchElement,
    SceneMetadata,
    SceneTime,
    SceneTimeAgent,
    agent_collate_fn,
    scene_collate_fn,
)
from avdata.dataset_specific import RawDataset
from avdata.utils import env_utils, string_utils


class UnifiedDataset(Dataset):
    # @profile
    def __init__(
        self,
        datasets: List[str],
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
        standardize_rotation: bool = True,
        data_dirs: Dict[str, str] = {
            # "nusc": "~/datasets/nuScenes",
            "nusc_mini": "~/datasets/nuScenes",
            "lyft_sample": "~/datasets/lyft/scenes/sample.zarr",
        },
        cache_type: str = "sqlite",
        cache_location: str = "~/.unified_data_cache",
        rebuild_cache: bool = False,
    ) -> None:
        self.centric = centric

        if self.centric == "agent":
            self.collate_fn = agent_collate_fn
        elif self.centric == "scene":
            self.collate_fn = scene_collate_fn

        if cache_type == "sqlite":
            cache_class = SQLiteCache

        self.rebuild_cache = rebuild_cache
        self.cache: Type[BaseCache] = cache_class(cache_location)

        self.history_sec = history_sec
        self.future_sec = future_sec
        self.agent_interaction_distances = agent_interaction_distances
        self.incl_robot_future = incl_robot_future
        self.incl_map = incl_map
        self.only_types = None if only_types is None else set(only_types)
        self.no_types = None if no_types is None else set(no_types)
        self.standardize_rotation = standardize_rotation

        self.envs: List[RawDataset] = env_utils.get_raw_datasets(data_dirs)

        matching_datasets: List[Tuple[str, ...]] = self.get_matching_scene_tags(
            datasets
        )
        print(
            "Loading data for matched scene tags:",
            string_utils.pretty_string_tuples(matching_datasets),
            flush=True,
        )

        for env in self.envs:
            if (
                self.rebuild_cache or not (self.cache.path / env.name).is_dir()
            ) and any(env.name in dataset_tuple for dataset_tuple in matching_datasets):
                # Loading dataset objects in case we don't have
                # their data already cached.
                env.load_dataset_obj()

        self.scene_index: List[SceneMetadata] = self.preprocess_scene_metadata(
            matching_datasets
        )
        print(self.scene_index)

        self.data_index = list()
        if self.centric == "scene":
            for scene_info in self.scene_index:
                for ts in range(scene_info.length_timesteps):
                    # This is where we remove scene timesteps that would have no remaining agents after filtering.
                    if filtering.all_agents_excluded_types(
                        no_types, scene_info.agent_presence[ts]
                    ):
                        continue
                    elif filtering.no_agent_included_types(
                        only_types, scene_info.agent_presence[ts]
                    ):
                        continue

                    if filtering.no_agent_satisfies_time(
                        ts,
                        scene_info.dt,
                        self.history_sec,
                        self.future_sec,
                        scene_info.agent_presence[ts],
                    ):
                        # Ignore this datum if no agent in the scene satisfies our time requirements.
                        continue

                    self.data_index.append((scene_info.env_name, scene_info.name, ts))

        elif self.centric == "agent":
            for scene_info in self.scene_index:
                for ts in range(scene_info.length_timesteps):
                    for agent_info in scene_info.agent_presence[ts]:
                        # Ignore this datum if the agent does not satisfy
                        # our minimum timestep requirements.
                        if not filtering.satisfies_times(
                            agent_info,
                            ts,
                            scene_info.dt,
                            self.history_sec,
                            self.future_sec,
                        ):
                            continue

                        # This is where we remove agents that do not match
                        # our AgentType inclusion/exclusion filters.
                        if filtering.exclude_types(self.no_types, agent_info.type):
                            continue
                        elif filtering.not_included_types(
                            self.only_types, agent_info.type
                        ):
                            continue

                        # Don't want to predict the ego if we're going to be giving the model its future!
                        if filtering.robot_future(
                            self.incl_robot_future, agent_info.name
                        ):
                            continue

                        self.data_index.append(
                            (scene_info.env_name, scene_info.name, ts, agent_info.name)
                        )

        manager = Manager()
        self.data_index = manager.list(self.data_index)
        self.data_len: int = len(self.data_index)

    def get_matching_scene_tags(self, queries: List[str]) -> List[Tuple[str, ...]]:
        # if queries is None:
        #     return list(chain.from_iterable(env.components for env in self.envs))

        query_tuples = [set(data.split("-")) for data in queries]

        matching_scene_tags = list()
        for dataset_tuple in query_tuples:
            for env in self.envs:
                matching_scene_tags += env.get_matching_scene_tags(dataset_tuple)

        return matching_scene_tags

    def preprocess_scene_metadata(
        self, scene_tags: List[Tuple[str, ...]]
    ) -> List[SceneMetadata]:
        scenes_list: List[SceneMetadata] = list()
        for scene_tag in tqdm(scene_tags, desc="Loading Scene Metadata"):
            for env in self.envs:
                if env.name in scene_tag:
                    scenes_list += env.get_matching_scenes(
                        scene_tag, self.cache, self.rebuild_cache
                    )

        self.calculate_agent_presence(scenes_list)
        return scenes_list

    def calculate_agent_presence(self, scenes: List[SceneMetadata]) -> None:
        scene_info: SceneMetadata
        for scene_info in tqdm(scenes, desc="Calculating Agent Presence"):
            cache_scene_dir: Path = (
                self.cache.path / scene_info.env_name / scene_info.name
            )
            if not cache_scene_dir.is_dir():
                cache_scene_dir.mkdir(parents=True)

            scene_file: Path = cache_scene_dir / "scene_metadata.dill"
            if scene_file.is_file() and not self.rebuild_cache:
                with open(scene_file, "rb") as f:
                    cached_scene_info: SceneMetadata = dill.load(f)

                scene_info.update_agent_presence(cached_scene_info.agent_presence)
                continue

            for env in self.envs:
                if scene_info.env_name == env.name:
                    agent_presence = env.get_and_cache_agent_presence(
                        scene_info, cache_scene_dir, self.rebuild_cache
                    )
                    scene_info.update_agent_presence(agent_presence)
                    break
            else:
                raise ValueError(
                    f"Scene {str(scene_info)} had no corresponding environemnt!"
                )

            with open(scene_file, "wb") as f:
                dill.dump(scene_info, f)

    def __len__(self) -> int:
        return self.data_len

    # @profile
    def __getitem__(self, idx: int) -> AgentBatchElement:
        if self.centric == "scene":
            env_name, scene_name, ts = self.data_index[idx]
        elif self.centric == "agent":
            env_name, scene_name, ts, agent_id = self.data_index[idx]

        scene_cache_dir: Path = self.cache.path / env_name / scene_name
        scene_file: Path = scene_cache_dir / "scene_metadata.dill"
        with open(scene_file, "rb") as f:
            scene_info: SceneMetadata = dill.load(f)

        if self.centric == "scene":
            scene_time: SceneTime = SceneTime.from_cache(
                scene_info,
                ts,
                scene_cache_dir,
                only_types=self.only_types,
                no_types=self.no_types,
            )

            return SceneBatchElement(scene_time, self.history_sec, self.future_sec)
        elif self.centric == "agent":
            scene_time_agent: SceneTimeAgent = SceneTimeAgent.from_cache(
                scene_info,
                ts,
                agent_id,
                scene_cache_dir,
                only_types=self.only_types,
                no_types=self.no_types,
                incl_robot_future=self.incl_robot_future,
            )

            return AgentBatchElement(
                scene_cache_dir,
                idx,
                scene_time_agent,
                self.history_sec,
                self.future_sec,
                self.agent_interaction_distances,
                self.incl_robot_future,
                self.incl_map,
                self.standardize_rotation,
            )

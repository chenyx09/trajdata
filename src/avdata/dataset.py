from collections import defaultdict

# from itertools import chain
from multiprocessing import Manager
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from avdata import filtering
from avdata.caching import DataFrameCache, EnvCache, SceneCache
from avdata.data_structures import (
    AgentBatchElement,
    AgentType,
    SceneBatchElement,
    SceneMetadata,
    SceneTag,
    SceneTime,
    SceneTimeAgent,
    agent_collate_fn,
    scene_collate_fn,
)
from avdata.data_structures.agent import AgentMetadata
from avdata.dataset_specific import RawDataset
from avdata.utils import env_utils, string_utils


class UnifiedDataset(Dataset):
    # @profile
    def __init__(
        self,
        desired_data: List[str],
        scene_description_contains: Optional[List[str]] = None,
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
        standardize_data: bool = True,
        data_dirs: Dict[str, str] = {
            # "nusc": "~/datasets/nuScenes",
            "nusc_mini": "~/datasets/nuScenes",
            "lyft_sample": "~/datasets/lyft/scenes/sample.zarr",
        },
        cache_type: str = "dataframe",
        cache_location: str = "~/.unified_data_cache",
        rebuild_cache: bool = False,
    ) -> None:
        self.centric = centric

        if self.centric == "agent":
            self.collate_fn = agent_collate_fn
        elif self.centric == "scene":
            self.collate_fn = scene_collate_fn

        if cache_type == "dataframe":
            self.cache_class = DataFrameCache

        self.rebuild_cache = rebuild_cache
        self.env_cache: EnvCache = EnvCache(cache_location)

        self.history_sec = history_sec
        self.future_sec = future_sec
        self.agent_interaction_distances = agent_interaction_distances
        self.incl_robot_future = incl_robot_future
        self.incl_map = incl_map
        self.only_types = None if only_types is None else set(only_types)
        self.no_types = None if no_types is None else set(no_types)
        self.standardize_data = standardize_data

        # Ensuring scene description queries are all lowercase
        if scene_description_contains is not None:
            scene_description_contains = [s.lower() for s in scene_description_contains]

        self.envs: List[RawDataset] = env_utils.get_raw_datasets(data_dirs)

        matching_datasets: List[SceneTag] = self.get_matching_scene_tags(desired_data)
        print(
            "Loading data for matched scene tags:",
            string_utils.pretty_string_tags(matching_datasets),
            flush=True,
        )

        for env in self.envs:
            if (
                self.rebuild_cache or not self.env_cache.env_is_cached(env.name)
            ) and any(env.name in dataset_tuple for dataset_tuple in matching_datasets):
                # Loading dataset objects in case we don't have
                # their data already cached.
                env.load_dataset_obj()

        self.scene_index: List[SceneMetadata] = self.preprocess_scene_metadata(
            matching_datasets, scene_description_contains
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
                filtered_agents: List[AgentMetadata] = filtering.agent_types(
                    scene_info.agents, self.no_types, self.only_types
                )

                for agent_info in filtered_agents:
                    # Don't want to predict the ego if we're going to be giving the model its future!
                    if incl_robot_future and agent_info.name == "ego":
                        continue

                    valid_ts: List[int] = filtering.get_valid_ts(
                        agent_info, scene_info.dt, self.history_sec, self.future_sec
                    )
                    self.data_index += [
                        (scene_info.env_name, scene_info.name, ts, agent_info.name)
                        for ts in valid_ts
                    ]

        manager = Manager()
        self.data_index = manager.list(self.data_index)
        self.data_len: int = len(self.data_index)

    def get_matching_scene_tags(self, queries: List[str]) -> List[SceneTag]:
        # if queries is None:
        #     return list(chain.from_iterable(env.components for env in self.envs))

        query_tuples = [set(data.split("-")) for data in queries]

        matching_scene_tags: List[SceneTag] = list()
        for query_tuple in query_tuples:
            for env in self.envs:
                matching_scene_tags += env.get_matching_scene_tags(query_tuple)

        return matching_scene_tags

    def preprocess_scene_metadata(
        self,
        scene_tags: List[SceneTag],
        scene_description_contains: Optional[List[str]],
    ) -> List[SceneMetadata]:
        scenes_list: List[SceneMetadata] = list()
        for scene_tag in tqdm(scene_tags, desc="Loading Scene Metadata"):
            for env in self.envs:
                if env.name in scene_tag:
                    scenes_list += env.get_matching_scenes(
                        scene_tag,
                        scene_description_contains,
                        self.env_cache,
                        self.rebuild_cache,
                    )

        self.calculate_agent_data(scenes_list)
        return scenes_list

    def calculate_agent_data(self, scenes: List[SceneMetadata]) -> None:
        scene_info: SceneMetadata
        for scene_info in tqdm(scenes, desc="Calculating Agent Data"):
            if (
                self.env_cache.scene_is_cached(scene_info.env_name, scene_info.name)
                and not self.rebuild_cache
            ):
                cached_scene_info: SceneMetadata = self.env_cache.load_scene_metadata(
                    scene_info.env_name, scene_info.name
                )

                scene_info.update_agent_info(
                    cached_scene_info.agents, cached_scene_info.agent_presence
                )
                continue

            for env in self.envs:
                if scene_info.env_name == env.name:
                    agent_list, agent_presence = env.get_agent_info(
                        scene_info, self.env_cache.path, self.cache_class
                    )
                    scene_info.update_agent_info(agent_list, agent_presence)
                    break
            else:
                raise ValueError(
                    f"Scene {str(scene_info)} had no corresponding environemnt!"
                )

            self.env_cache.save_scene_metadata(scene_info)

    def __len__(self) -> int:
        return self.data_len

    # @profile
    def __getitem__(self, idx: int) -> AgentBatchElement:
        if self.centric == "scene":
            env_name, scene_name, ts = self.data_index[idx]
        elif self.centric == "agent":
            env_name, scene_name, ts, agent_id = self.data_index[idx]

        scene_info: SceneMetadata = self.env_cache.load_scene_metadata(
            env_name, scene_name
        )
        scene_cache: Type[SceneCache] = self.cache_class(
            self.env_cache.path, scene_info, ts
        )

        if self.centric == "scene":
            scene_time: SceneTime = SceneTime.from_cache(
                scene_info,
                ts,
                scene_cache,
                only_types=self.only_types,
                no_types=self.no_types,
            )

            return SceneBatchElement(scene_time, self.history_sec, self.future_sec)
        elif self.centric == "agent":
            scene_time_agent: SceneTimeAgent = SceneTimeAgent.from_cache(
                scene_info,
                ts,
                agent_id,
                scene_cache,
                only_types=self.only_types,
                no_types=self.no_types,
                incl_robot_future=self.incl_robot_future,
            )

            return AgentBatchElement(
                scene_cache,
                idx,
                scene_time_agent,
                self.history_sec,
                self.future_sec,
                self.agent_interaction_distances,
                self.incl_robot_future,
                self.incl_map,
                self.standardize_data,
            )

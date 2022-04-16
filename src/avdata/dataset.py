from collections import defaultdict
from functools import partial

# from itertools import chain
from multiprocessing import Manager
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from pathos.pools import ProcessPool
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
        desired_dt: Optional[float] = None,
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
        map_params: Optional[Dict[str, int]] = None,
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
        rebuild_maps: bool = False,
        num_workers: int = 0,
        verbose: bool = False,
    ) -> None:
        """Instantiates a PyTorch Dataset object which aggregates data
        from multiple autonomous vehicle (AV) datasets.

        Args:
            desired_data (List[str]): Names of AV datasets, training splits, or locations from which to load data.
            scene_description_contains (Optional[List[str]], optional): Strings to search for within scene descriptions (for datasets which provide scene descriptions). Defaults to None.
            centric (str, optional): Batch data format, one of {"agent", "scene"}, matching the type of model you want to train. Defaults to "agent".
            history_sec (Tuple[Optional[float], Optional[float]], optional): _description_. Defaults to ( None, None, ).
            incl_robot_future (bool, optional): _description_. Defaults to False.
            incl_map (bool, optional): _description_. Defaults to False.
            map_params (Optional[Dict[str, int]], optional): _description_. Defaults to None.
            only_types (Optional[List[AgentType]], optional): _description_. Defaults to None.
            no_types (Optional[List[AgentType]], optional): _description_. Defaults to None.
            standardize_data (bool, optional): _description_. Defaults to True.
            data_dirs (_type_, optional): _description_. Defaults to { "nusc_mini": "~/datasets/nuScenes", "lyft_sample": "~/datasets/lyft/scenes/sample.zarr", }.
            cache_type (str, optional): _description_. Defaults to "dataframe".
            cache_location (str, optional): _description_. Defaults to "~/.unified_data_cache".
            rebuild_cache (bool, optional): _description_. Defaults to False.
            rebuild_maps (bool, optional): _description_. Defaults to False.
            num_workers (int, optional): _description_. Defaults to 0.
            verbose (bool, optional): _description_. Defaults to False.
        """
        self.centric: str = centric
        self.desired_dt: float = desired_dt

        if cache_type == "dataframe":
            self.cache_class = DataFrameCache

        self.rebuild_cache: bool = rebuild_cache
        self.cache_path: Path = Path(cache_location).expanduser().resolve()
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.env_cache: EnvCache = EnvCache(self.cache_path)

        if incl_map:
            assert (
                map_params is not None
            ), r"Path size information, i.e., {'px_per_m': ..., 'map_size_px': ...}, must be provided if incl_map=True"
            assert (
                map_params["map_size_px"] % 2 == 0
            ), "Patch parameter 'map_size_px' must be divisible by 2"

        self.history_sec = history_sec
        self.future_sec = future_sec
        self.agent_interaction_distances = agent_interaction_distances
        self.incl_robot_future = incl_robot_future
        self.incl_map = incl_map
        self.map_params = map_params
        self.only_types = None if only_types is None else set(only_types)
        self.no_types = None if no_types is None else set(no_types)
        self.standardize_data = standardize_data
        self.verbose = verbose

        # Ensuring scene description queries are all lowercase
        if scene_description_contains is not None:
            scene_description_contains = [s.lower() for s in scene_description_contains]

        self.envs: List[RawDataset] = env_utils.get_raw_datasets(data_dirs)
        self.envs_dict: Dict[str, RawDataset] = {env.name: env for env in self.envs}

        matching_datasets: List[SceneTag] = self.get_matching_scene_tags(desired_data)
        if self.verbose:
            print(
                "Loading data for matched scene tags:",
                string_utils.pretty_string_tags(matching_datasets),
                flush=True,
            )

        for env in self.envs:
            if (
                self.rebuild_cache
                or rebuild_maps
                or not self.env_cache.env_is_cached(env.name)
            ) and any(env.name in dataset_tuple for dataset_tuple in matching_datasets):
                if self.desired_dt is not None and self.desired_dt != env.metadata.dt:
                    raise ValueError(
                        f"{env.name} has yet to be cached, and setting a desired dt of {self.desired_dt}s "
                        f"(which differs from its original dt of {env.metadata.dt}s) is not allowed."
                    )

                # Loading dataset objects in case we don't have
                # their data already cached.
                env.load_dataset_obj()

                if rebuild_maps or not self.cache_class.are_maps_cached(
                    self.cache_path, env.name
                ):
                    env.cache_maps(self.cache_path, self.cache_class)

        self.scene_index: List[SceneMetadata] = self.preprocess_scene_metadata(
            matching_datasets, scene_description_contains, num_workers
        )
        if self.verbose:
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

    def get_collate_fn(
        self, centric: str = "agent", return_dict: bool = False
    ) -> Callable:
        if centric == "agent":
            collate_fn = partial(agent_collate_fn, return_dict=return_dict)
        elif centric == "scene":
            collate_fn = partial(scene_collate_fn, return_dict=return_dict)

        return collate_fn

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
        num_workers: int,
    ) -> List[SceneMetadata]:
        scenes_list: List[SceneMetadata] = list()
        for scene_tag in tqdm(
            scene_tags, desc="Loading Scene Metadata", disable=not self.verbose
        ):
            for env in self.envs:
                if env.name in scene_tag:
                    scenes_list += env.get_matching_scenes(
                        scene_tag,
                        scene_description_contains,
                        self.env_cache,
                        self.rebuild_cache,
                    )

        all_cached: bool = not self.rebuild_cache and all(
            self.env_cache.scene_is_cached(scene_info.env_name, scene_info.name)
            for scene_info in scenes_list
        )

        serial_scenes: List[SceneMetadata]
        parallel_scenes: List[SceneMetadata]
        if num_workers > 1 and not all_cached:
            serial_scenes = [
                scene_info
                for scene_info in scenes_list
                if not self.envs_dict[scene_info.env_name].parallelizable
            ]
            parallel_scenes = [
                scene_info
                for scene_info in scenes_list
                if self.envs_dict[scene_info.env_name].parallelizable
            ]
        else:
            serial_scenes = scenes_list
            parallel_scenes = list()

        filled_scenes_list: List[SceneMetadata] = list()
        if serial_scenes:
            # Scenes for which it's faster to process them serially. See
            # the longer comment below for a more thorough explanation.
            scene_info: SceneMetadata
            for scene_info in tqdm(
                serial_scenes,
                desc="Calculating Agent Data (Serially)",
                disable=not self.verbose,
            ):
                corresponding_env = self.envs_dict[scene_info.env_name]
                filled_scenes_list.append(
                    self.get_agent_data(scene_info, corresponding_env)
                )

            # No more need for the original dataset objects and freeing up
            # this memory allows the parallel processing below to run very fast.
            for env in self.envs:
                if not env.parallelizable:
                    env.del_dataset_obj()

        if parallel_scenes:
            # Scenes for which it's faster to process them in parallel
            # Note this really only applies to scenes whose raw datasets
            # are "parallelizable" AKA take up a small amount of memory
            # and effectively act as a window into the data on disk.
            # E.g., NuScenes objects load a lot of data into RAM, so
            # they are not parallelizable and should be processed
            # serially (thankfully it is quite fast to do so).
            with ProcessPool(num_workers) as pool:
                filled_scenes_list += list(
                    tqdm(
                        pool.uimap(
                            self.get_agent_data,
                            parallel_scenes,
                            list(
                                self.envs_dict[scene_info.env_name]
                                for scene_info in parallel_scenes
                            ),
                        ),
                        total=len(parallel_scenes),
                        desc=f"Calculating Agent Data ({num_workers} CPUs)",
                        disable=not self.verbose,
                    )
                )

            # No more need for the original dataset objects.
            for env in self.envs:
                if env.parallelizable:
                    env.del_dataset_obj()

        return filled_scenes_list

    def get_agent_data(
        self,
        scene_info: SceneMetadata,
        raw_dataset: RawDataset,
    ) -> SceneMetadata:
        if not self.rebuild_cache and self.env_cache.scene_is_cached(
            scene_info.env_name, scene_info.name
        ):
            cached_scene_info: SceneMetadata = self.env_cache.load_scene_metadata(
                scene_info.env_name, scene_info.name
            )

            scene_info.update_agent_info(
                cached_scene_info.agents,
                cached_scene_info.agent_presence,
            )

        else:
            agent_list, agent_presence = raw_dataset.get_agent_info(
                scene_info, self.env_cache.path, self.cache_class
            )

            scene_info.update_agent_info(agent_list, agent_presence)
            self.env_cache.save_scene_metadata(scene_info)

        self.enforce_desired_dt(scene_info)

        return scene_info

    def enforce_desired_dt(self, scene_info: SceneMetadata) -> None:
        # TODO(bivanovic): Eventually also implement subsample_scene_dt
        if self.desired_dt is not None and scene_info.dt != self.desired_dt:
            self.interpolate_scene_dt(scene_info)

    def interpolate_scene_dt(self, scene_info: SceneMetadata) -> None:
        dt_ratio: float = scene_info.dt / self.desired_dt
        if not dt_ratio.is_integer():
            raise ValueError(
                f"{scene_info.dt} is not divisible by {self.desired_dt} for {str(scene_info)}"
            )

        dt_factor: int = int(dt_ratio)

        # E.g., the scene is currently at dt = 0.5s (2 Hz),
        # but we want desired_dt = 0.1s (10 Hz).
        scene_info.length_timesteps = (scene_info.length_timesteps - 1) * dt_factor + 1
        agent_presence: List[List[AgentMetadata]] = [
            [] for _ in range(scene_info.length_timesteps)
        ]
        for agent in scene_info.agents:
            agent.first_timestep *= dt_factor
            agent.last_timestep *= dt_factor

            for scene_ts in range(agent.first_timestep, agent.last_timestep + 1):
                agent_presence[scene_ts].append(agent)

        scene_info.update_agent_info(scene_info.agents, agent_presence)
        scene_info.dt = self.desired_dt
        # Note we do not touch scene_info.env_metadata.dt, this will serve as our
        # source of the "original" data dt information.

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
        self.enforce_desired_dt(scene_info)

        scene_cache: SceneCache = self.cache_class(self.cache_path, scene_info, ts)
        if (
            self.desired_dt is not None
            and scene_info.env_metadata.dt != self.desired_dt
        ):
            scene_cache.interpolate_data(self.desired_dt)

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
                self.map_params,
                self.standardize_data,
            )

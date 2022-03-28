from typing import Dict, List, Optional, Tuple, Type

import pandas as pd
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

from avdata.caching import BaseCache
from avdata.data_structures.agent import Agent, AgentMetadata, AgentType, FixedSize
from avdata.data_structures.environment import EnvMetadata
from avdata.data_structures.scene import SceneMetadata
from avdata.data_structures.scene_tag import SceneTag
from avdata.dataset_specific.nusc import nusc_utils
from avdata.dataset_specific.raw_dataset import RawDataset
from avdata.dataset_specific.scene_records import NuscSceneRecord


class NuscDataset(RawDataset):
    def compute_metadata(self, env_name: str, data_dir: str) -> EnvMetadata:
        all_scene_splits: Dict[str, List[str]] = create_splits_scenes()
        if env_name == "nusc":
            nusc_scene_splits: Dict[str, List[str]] = {
                k: all_scene_splits[k] for k in ["train", "val", "test"]
            }

            # nuScenes possibilities are the Cartesian product of these
            dataset_parts: List[Tuple[str, ...]] = [
                ("train", "val", "test"),
                ("boston", "singapore"),
            ]
        elif env_name == "nusc_mini":
            nusc_scene_splits: Dict[str, List[str]] = {
                k: all_scene_splits[k] for k in ["mini_train", "mini_val"]
            }

            # nuScenes possibilities are the Cartesian product of these
            dataset_parts: List[Tuple[str, ...]] = [
                ("mini_train", "mini_val"),
                ("boston", "singapore"),
            ]

        # Inverting the dict from above, associating every scene with its data split.
        nusc_scene_split_map: Dict[str, str] = {
            v_elem: k for k, v in nusc_scene_splits.items() for v_elem in v
        }

        return EnvMetadata(
            name=env_name,
            data_dir=data_dir,
            dt=nusc_utils.NUSC_DT,
            parts=dataset_parts,
            scene_split_map=nusc_scene_split_map,
        )

    def load_dataset_obj(self) -> None:
        print(f"Loading {self.name} dataset...", flush=True)

        if self.name == "nusc_mini":
            version_str = "v1.0-mini"
        elif self.name == "nusc":
            version_str = "v1.0-trainval"

        self.dataset_obj = NuScenes(
            version=version_str, dataroot=self.metadata.data_dir
        )

    def _get_matching_scenes_from_obj(
        self,
        scene_tag: SceneTag,
        scene_desc_matches: Optional[List[str]],
        cache: Type[BaseCache],
    ) -> List[SceneMetadata]:
        all_scenes_list: List[NuscSceneRecord] = list()

        scenes_list: List[SceneMetadata] = list()
        for scene_record in self.dataset_obj.scene:
            scene_name: str = scene_record["name"]
            scene_desc: str = scene_record["description"].lower()
            scene_location: str = self.dataset_obj.get(
                "log", scene_record["log_token"]
            )["location"]
            scene_split: str = self.metadata.scene_split_map[scene_name]
            scene_length: int = scene_record["nbr_samples"]

            # Saving all scene records for later caching.
            all_scenes_list.append(
                NuscSceneRecord(scene_name, scene_location, scene_length, scene_desc)
            )

            if scene_location.split("-")[0] in scene_tag and scene_split in scene_tag:
                if scene_desc_matches is not None and not any(
                    desc_query in scene_desc for desc_query in scene_desc_matches
                ):
                    continue

                scene_metadata = SceneMetadata(
                    self.metadata,
                    scene_name,
                    scene_location,
                    scene_split,
                    scene_length,
                    scene_record,
                    scene_desc,
                )
                scenes_list.append(scene_metadata)

        self.cache_all_scenes_list(cache, all_scenes_list)
        return scenes_list

    def _get_matching_scenes_from_cache(
        self,
        scene_tag: SceneTag,
        scene_desc_matches: Optional[List[str]],
        cache: Type[BaseCache],
    ) -> List[SceneMetadata]:
        all_scenes_list: List[NuscSceneRecord] = cache.load_env_scenes_list(self.name)

        scenes_list: List[SceneMetadata] = list()
        for scene_record in all_scenes_list:
            scene_name, scene_location, scene_length, scene_desc = scene_record
            scene_split: str = self.metadata.scene_split_map[scene_name]

            if scene_location.split("-")[0] in scene_tag and scene_split in scene_tag:
                if scene_desc_matches is not None and not any(
                    desc_query in scene_desc for desc_query in scene_desc_matches
                ):
                    continue

                scene_metadata = SceneMetadata(
                    self.metadata,
                    scene_name,
                    scene_location,
                    scene_split,
                    scene_length,
                    None,  # This isn't used if everything is already cached.
                    scene_desc,
                )
                scenes_list.append(scene_metadata)

        return scenes_list

    def get_and_cache_agent_presence(
        self, scene_info: SceneMetadata, cache: Type[BaseCache]
    ) -> List[List[AgentMetadata]]:
        agent_presence: List[List[AgentMetadata]] = [
            [
                AgentMetadata(
                    name="ego",
                    agent_type=AgentType.VEHICLE,
                    first_timestep=0,
                    last_timestep=scene_info.length_timesteps - 1,
                    fixed_size=FixedSize(length=4.084, width=1.730, height=1.562),
                )
            ]
            for _ in range(scene_info.length_timesteps)
        ]

        agent_data_list: List[pd.DataFrame] = list()
        existing_agents: Dict[str, AgentMetadata] = dict()
        for frame_idx, frame_info in enumerate(
            nusc_utils.frame_iterator(self.dataset_obj, scene_info)
        ):
            for agent_info in nusc_utils.agent_iterator(self.dataset_obj, frame_info):
                if agent_info["instance_token"] in existing_agents:
                    agent_presence[frame_idx].append(
                        existing_agents[agent_info["instance_token"]]
                    )
                    continue

                if not agent_info["next"]:
                    # There are some agents with only a single detection to them, we don't care about these.
                    continue

                agent: Agent = nusc_utils.agg_agent_data(
                    self.dataset_obj, agent_info, frame_idx
                )

                agent_presence[frame_idx].append(agent.metadata)
                existing_agents[agent.name] = agent.metadata

                agent_data_list.append(agent.data)

        ego_agent: Agent = nusc_utils.agg_ego_data(self.dataset_obj, scene_info)
        agent_data_list.append(ego_agent.data)

        cache.save_agent_data(pd.concat(agent_data_list), scene_info)

        return agent_presence

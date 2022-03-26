from pathlib import Path
from typing import List, Optional, Type

import dill
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes

from avdata.caching import BaseCache
from avdata.data_structures.agent import AgentMetadata
from avdata.data_structures.environment import EnvMetadata
from avdata.data_structures.scene import SceneMetadata
from avdata.data_structures.scene_tag import SceneTag
from avdata.dataset_specific.raw_dataset import RawDataset
from avdata.dataset_specific.scene_records import NuscSceneRecord
from avdata.utils import nusc_utils


class NuscDataset(RawDataset):
    def __init__(self, metadata: EnvMetadata) -> None:
        super().__init__(metadata)

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
        self, scene_info: SceneMetadata, cache_scene_dir: Path, rebuild_cache: bool
    ) -> List[List[AgentMetadata]]:
        agent_presence: List[List[AgentMetadata]] = nusc_utils.calc_agent_presence(
            scene_info=scene_info,
            nusc_obj=self.dataset_obj,
            cache_scene_dir=cache_scene_dir,
            rebuild_cache=rebuild_cache,
        )
        return agent_presence

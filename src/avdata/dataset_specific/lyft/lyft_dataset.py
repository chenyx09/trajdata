from pathlib import Path
from typing import List, Tuple, Type

import dill
from l5kit.data import ChunkedDataset

from avdata.caching import BaseCache
from avdata.data_structures import AgentMetadata, EnvMetadata, SceneMetadata
from avdata.data_structures.environment import EnvMetadata
from avdata.dataset_specific.raw_dataset import RawDataset
from avdata.dataset_specific.scene_records import LyftSceneRecord
from avdata.utils import lyft_utils


class LyftDataset(RawDataset):
    def __init__(self, metadata: EnvMetadata) -> None:
        super().__init__(metadata)

    def load_dataset_obj(self) -> None:
        print(f"Loading {self.name} dataset...", flush=True)

        self.dataset_obj = ChunkedDataset(str(self.metadata.data_dir)).open()

    def _get_matching_scenes_from_obj(
        self,
        dataset_tuple: Tuple[
            str, ...
        ],  # TODO(bivanovic): Add in the scene_tag query handling below
        cache: Type[BaseCache],
    ) -> List[SceneMetadata]:
        all_scenes_list: List[LyftSceneRecord] = list()

        scenes_list: List[SceneMetadata] = list()
        all_scene_frames = self.dataset_obj.scenes["frame_index_interval"]
        for idx in range(all_scene_frames.shape[0]):
            scene_name: str = f"scene-{idx:04d}"
            scene_split: str = self.metadata.scene_split_map[scene_name]
            scene_length: int = (
                all_scene_frames[idx, 1] - all_scene_frames[idx, 0]
            ).item()  # Doing .item() otherwise it'll be a numpy.int64

            # Saving all scene records for later caching.
            all_scenes_list.append(LyftSceneRecord(scene_name, scene_length))

            if scene_split in dataset_tuple:
                scene_metadata = SceneMetadata(
                    self.metadata,
                    scene_name,
                    "palo_alto",
                    scene_split,
                    scene_length,
                    all_scene_frames[idx],
                )
                scenes_list.append(scene_metadata)

        self.cache_all_scenes_list(cache, all_scenes_list)
        return scenes_list

    def _get_matching_scenes_from_cache(
        self,
        dataset_tuple: Tuple[
            str, ...
        ],  # TODO(bivanovic): Add in the scene_tag query handling below
        cache: Type[BaseCache],
    ) -> List[SceneMetadata]:
        env_cache_dir: Path = cache.path / self.name
        with open(env_cache_dir / "scenes_list.dill", "rb") as f:
            all_scenes_list: List[LyftSceneRecord] = dill.load(f)

        scenes_list: List[SceneMetadata] = list()
        for scene_record in all_scenes_list:
            scene_name, scene_length = scene_record
            scene_split: str = self.metadata.scene_split_map[scene_name]

            if scene_split in dataset_tuple:
                scene_metadata = SceneMetadata(
                    self.metadata,
                    scene_name,
                    "palo_alto",
                    scene_split,
                    scene_length,
                    None,  # This isn't used if everything is already cached.
                )
                scenes_list.append(scene_metadata)

        return scenes_list

    def get_and_cache_agent_presence(
        self, scene_info: SceneMetadata, cache_scene_dir: Path, rebuild_cache: bool
    ) -> List[List[AgentMetadata]]:
        agent_presence = lyft_utils.calc_agent_presence(
            scene_info=scene_info,
            lyft_obj=self.dataset_obj,
            cache_scene_dir=cache_scene_dir,
            rebuild_cache=rebuild_cache,
        )
        return agent_presence

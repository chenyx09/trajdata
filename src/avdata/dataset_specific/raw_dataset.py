from pathlib import Path
from typing import List, NamedTuple, Optional, Set, Tuple, Type

from avdata.caching import EnvCache, SceneCache
from avdata.data_structures import AgentMetadata, EnvMetadata, SceneMetadata, SceneTag


class RawDataset:
    def __init__(self, env_name: str, data_dir: str, parallelizable: bool) -> None:
        metadata = self.compute_metadata(env_name, data_dir)

        self.metadata = metadata
        self.name = metadata.name
        self.scene_tags = metadata.scene_tags
        self.dataset_obj = None
        self.parallelizable = parallelizable

    def compute_metadata(self, env_name: str, data_dir: str) -> EnvMetadata:
        raise NotImplementedError()

    def get_matching_scene_tags(self, query: Set[str]) -> List[SceneTag]:
        return [scene_tag for scene_tag in self.scene_tags if scene_tag.contains(query)]

    def load_dataset_obj(self, verbose: bool = False) -> None:
        raise NotImplementedError()

    def del_dataset_obj(self) -> None:
        self.dataset_obj = None

    def _get_matching_scenes_from_cache(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[SceneMetadata]:
        raise NotImplementedError()

    def _get_matching_scenes_from_obj(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[SceneMetadata]:
        raise NotImplementedError()

    def cache_all_scenes_list(
        self, env_cache: EnvCache, all_scenes_list: List[NamedTuple]
    ) -> None:
        env_cache.save_env_scenes_list(self.name, all_scenes_list)

    def get_matching_scenes(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
        rebuild_cache: bool,
    ) -> List[SceneMetadata]:
        if self.dataset_obj is None and not rebuild_cache:
            return self._get_matching_scenes_from_cache(
                scene_tag, scene_desc_contains, env_cache
            )
        else:
            return self._get_matching_scenes_from_obj(
                scene_tag, scene_desc_contains, env_cache
            )

    def get_agent_info(
        self, scene_info: SceneMetadata, cache_path: Path, cache_class: Type[SceneCache]
    ) -> Tuple[List[AgentMetadata], List[List[AgentMetadata]]]:
        raise NotImplementedError()

    def cache_maps(
        self, cache_path: Path, map_cache_class: Type[SceneCache], resolution: int = 2
    ) -> None:
        """
        resolution is in pixels per meter.
        """
        raise NotImplementedError()

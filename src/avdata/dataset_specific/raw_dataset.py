from typing import List, NamedTuple, Optional, Set, Type

from avdata.caching import SceneCache, EnvCache
from avdata.data_structures import AgentMetadata, EnvMetadata, SceneMetadata, SceneTag


class RawDataset:
    def __init__(self, env_name: str, data_dir: str) -> None:
        metadata = self.compute_metadata(env_name, data_dir)

        self.metadata = metadata
        self.name = metadata.name
        self.scene_tags = metadata.scene_tags
        self.dataset_obj = None

    def compute_metadata(self, env_name: str, data_dir: str) -> EnvMetadata:
        raise NotImplementedError()

    def get_matching_scene_tags(self, query: Set[str]) -> List[SceneTag]:
        return [scene_tag for scene_tag in self.scene_tags if scene_tag.contains(query)]

    def load_dataset_obj(self) -> None:
        raise NotImplementedError()

    def _get_matching_scenes_from_cache(
        self,
        scene_tag: SceneTag,
        scene_desc_matches: Optional[List[str]],
        cache: Type[SceneCache],
    ) -> List[SceneMetadata]:
        raise NotImplementedError()

    def _get_matching_scenes_from_obj(
        self,
        scene_tag: SceneTag,
        scene_desc_matches: Optional[List[str]],
        cache: Type[SceneCache],
    ) -> List[SceneMetadata]:
        raise NotImplementedError()

    def cache_all_scenes_list(
        self, cache: EnvCache, all_scenes_list: List[Type[NamedTuple]]
    ) -> None:
        cache.save_env_scenes_list(self.name, all_scenes_list)

    def get_matching_scenes(
        self,
        scene_tag: SceneTag,
        scene_desc_matches: Optional[List[str]],
        cache: Type[SceneCache],
        rebuild_cache: bool,
    ) -> List[SceneMetadata]:
        if self.dataset_obj is None and not rebuild_cache:
            return self._get_matching_scenes_from_cache(
                scene_tag, scene_desc_matches, cache
            )
        else:
            return self._get_matching_scenes_from_obj(
                scene_tag, scene_desc_matches, cache
            )

    def get_and_cache_agent_presence(
        self, scene_info: SceneMetadata, cache: Type[SceneCache]
    ) -> List[List[AgentMetadata]]:
        raise NotImplementedError()

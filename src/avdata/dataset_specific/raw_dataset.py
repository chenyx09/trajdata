from pathlib import Path
from typing import List, NamedTuple, Set, Tuple, Type

import dill

from avdata.caching import BaseCache
from avdata.data_structures import AgentMetadata, EnvMetadata, SceneMetadata


class RawDataset:
    def __init__(self, metadata: EnvMetadata) -> None:
        self.metadata = metadata
        self.name = (
            metadata.name
        )  # TODO(bivanovic): This being here means I don't have to have a Dict in dataset.py
        self.components = metadata.components

        self.dataset_obj = None

    def get_matching_scene_tags(self, query: Set[str]) -> List[str]:
        return [component for component in self.components if query.issubset(component)]

    def load_dataset_obj(self) -> None:
        raise NotImplementedError()

    def _get_matching_scenes_from_cache(
        self, dataset_tuple: Tuple[str, ...], cache: Type[BaseCache]
    ) -> List[SceneMetadata]:
        raise NotImplementedError()

    def _get_matching_scenes_from_obj(
        self,
        dataset_tuple: Tuple[str, ...],
        cache: Type[BaseCache],
    ) -> List[SceneMetadata]:
        raise NotImplementedError()

    def cache_all_scenes_list(
        self, cache: BaseCache, all_scenes_list: List[Type[NamedTuple]]
    ) -> None:
        env_cache_dir: Path = cache.path / self.name
        env_cache_dir.mkdir(parents=True, exist_ok=True)
        with open(env_cache_dir / "scenes_list.dill", "wb") as f:
            dill.dump(all_scenes_list, f)

    def get_matching_scenes(
        self, scene_tag: Tuple[str, ...], cache: BaseCache, rebuild_cache: bool
    ) -> List[SceneMetadata]:
        if self.dataset_obj is None and not rebuild_cache:
            return self._get_matching_scenes_from_cache(scene_tag, cache)
        else:
            return self._get_matching_scenes_from_obj(scene_tag, cache)

    def get_and_cache_agent_presence(
        self, scene_info: SceneMetadata, cache_scene_dir: Path, rebuild_cache: bool
    ) -> List[List[AgentMetadata]]:
        raise NotImplementedError()

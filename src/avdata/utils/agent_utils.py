from typing import Type

from avdata.caching import EnvCache, SceneCache
from avdata.data_structures import SceneMetadata
from avdata.dataset_specific import RawDataset


def get_agent_data(
    scene_info: SceneMetadata,
    raw_dataset: RawDataset,
    env_cache: EnvCache,
    rebuild_cache: bool,
    cache_class: Type[SceneCache],
) -> SceneMetadata:
    if not rebuild_cache and env_cache.scene_is_cached(
        scene_info.env_name, scene_info.name
    ):
        cached_scene_info: SceneMetadata = env_cache.load_scene_metadata(
            scene_info.env_name, scene_info.name
        )

        scene_info.update_agent_info(
            cached_scene_info.agents,
            cached_scene_info.agent_presence,
        )

    else:
        agent_list, agent_presence = raw_dataset.get_agent_info(
            scene_info, env_cache.path, cache_class
        )

        scene_info.update_agent_info(agent_list, agent_presence)
        env_cache.save_scene_metadata(scene_info)

    return scene_info

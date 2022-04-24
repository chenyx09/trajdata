from multiprocessing import Manager
from typing import Dict, List, Type

from torch.utils.data import Dataset

from avdata.caching import EnvCache, SceneCache
from avdata.data_structures import SceneMetadata
from avdata.dataset_specific import RawDataset


def scene_metadata_collate_fn(
    filled_scenes: List[SceneMetadata],
) -> List[SceneMetadata]:
    return filled_scenes


class ParallelDatasetPreprocessor(Dataset):
    def __init__(
        self,
        scenes_list: List[SceneMetadata],
        envs_dict: Dict[str, RawDataset],
        env_cache: EnvCache,
        cache_class: Type[SceneCache],
    ) -> None:
        self.env_cache = env_cache
        self.cache_class = cache_class

        manager = Manager()
        self.scenes_list = manager.list(scenes_list)
        self.envs_dict = manager.dict(envs_dict)
        self.data_len: int = len(self.scenes_list)

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, idx: int) -> None:
        scene_info = self.scenes_list[idx]
        raw_dataset = self.envs_dict[scene_info.env_name]

        # Leaving verbose False here so that we don't spam
        # stdout with loading messages.
        raw_dataset.load_dataset_obj(verbose=False)

        agent_list, agent_presence = raw_dataset.get_agent_info(
            scene_info, self.env_cache.path, self.cache_class
        )

        raw_dataset.del_dataset_obj()

        scene_info.update_agent_info(agent_list, agent_presence)
        self.env_cache.save_scene_metadata(scene_info)

        return scene_info

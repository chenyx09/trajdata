from pathlib import Path
from typing import Dict, List, Type

import numpy as np
from torch.utils.data import Dataset

from avdata.caching import EnvCache, SceneCache
from avdata.data_structures import SceneMetadata
from avdata.parallel.temp_cache import TemporaryCache
from avdata.utils.env_utils import get_raw_dataset


def scene_metadata_collate_fn(
    filled_scenes: List[SceneMetadata],
) -> List[SceneMetadata]:
    return filled_scenes


class ParallelDatasetPreprocessor(Dataset):
    def __init__(
        self,
        scene_info_paths: List[str],
        envs_dir_dict: Dict[str, str],
        env_cache_path: str,
        cache_class: Type[SceneCache],
    ) -> None:
        self.env_cache_path = np.array(env_cache_path).astype(np.string_)
        self.cache_class = cache_class

        self.scene_info_paths = np.array(scene_info_paths).astype(np.string_)

        self.env_names_arr = np.array(list(envs_dir_dict.keys())).astype(np.string_)
        self.data_dir_arr = np.array(list(envs_dir_dict.values())).astype(np.string_)

        self.data_len: int = len(scene_info_paths)

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, idx: int) -> None:
        env_cache_path: Path = Path(str(self.env_cache_path, encoding="utf-8"))
        env_cache: EnvCache = EnvCache(env_cache_path)

        scene_info_path: str = str(self.scene_info_paths[idx], encoding="utf-8")
        scene_info: SceneMetadata = TemporaryCache.load(scene_info_path)

        env_idx: int = np.argmax(
            self.env_names_arr == np.array(scene_info.env_name).astype(np.string_)
        ).item()
        raw_dataset = get_raw_dataset(
            scene_info.env_name, str(self.data_dir_arr[env_idx], encoding="utf-8")
        )

        # Leaving verbose False here so that we don't spam
        # stdout with loading messages.
        raw_dataset.load_dataset_obj(verbose=False)

        agent_list, agent_presence = raw_dataset.get_agent_info(
            scene_info, env_cache.path, self.cache_class
        )

        raw_dataset.del_dataset_obj()

        scene_info.update_agent_info(agent_list, agent_presence)
        env_cache.save_scene_metadata(scene_info)

        # Not calling self.enforce_desired_dt(scene_info) here
        # (as we do in UnifiedDataset.get_agent_data) because
        # this parallel dataloading will only occur if we are caching
        # new data (and we do not allow changing of dt during the
        # first cache creation).
        # TODO(bivanovic): Update this if the above changes in the future.

        return (
            env_cache.path
            / scene_info.env_name
            / scene_info.name
            / "scene_metadata.dill"
        )

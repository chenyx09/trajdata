from multiprocessing import Queue
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
from torch.utils.data import Dataset

from avdata.caching import EnvCache, SceneCache
from avdata.data_structures import Scene, SceneMetadata
from avdata.parallel.temp_cache import TemporaryCache
from avdata.utils import agent_utils, scene_utils
from avdata.utils.env_utils import get_raw_dataset


def scene_paths_collate_fn(filled_scenes: List) -> List:
    return filled_scenes


class ParallelDatasetPreprocessor(Dataset):
    def __init__(
        self,
        scene_info_q: Queue,
        num_scenes: int,
        envs_dir_dict: Dict[str, str],
        env_cache_path: str,
        temp_cache_path: str,
        desired_dt: Optional[float],
        cache_class: Type[SceneCache],
        rebuild_cache: bool,
    ) -> None:
        self.env_cache_path = np.array(env_cache_path).astype(np.string_)
        self.temp_cache_path = np.array(temp_cache_path).astype(np.string_)
        self.desired_dt = desired_dt
        self.cache_class = cache_class
        self.rebuild_cache = rebuild_cache

        self.scene_info_q = scene_info_q

        self.env_names_arr = np.array(list(envs_dir_dict.keys())).astype(np.string_)
        self.data_dir_arr = np.array(list(envs_dir_dict.values())).astype(np.string_)

        self.data_len: int = num_scenes

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, idx: int) -> Tuple[Path, Path]:
        env_cache_path: Path = Path(str(self.env_cache_path, encoding="utf-8"))
        env_cache: EnvCache = EnvCache(env_cache_path)

        scene_info: SceneMetadata = pickle.loads(self.scene_info_q.get())

        env_idx: int = np.argmax(
            self.env_names_arr == np.array(scene_info.env_name).astype(np.string_)
        ).item()
        raw_dataset = get_raw_dataset(
            scene_info.env_name, str(self.data_dir_arr[env_idx], encoding="utf-8")
        )

        # Leaving verbose False here so that we don't spam
        # stdout with loading messages.
        raw_dataset.load_dataset_obj(verbose=False)
        scene: Scene = agent_utils.get_agent_data(
            scene_info, raw_dataset, env_cache, self.rebuild_cache, self.cache_class
        )
        raw_dataset.del_dataset_obj()

        orig_scene_path: Path = EnvCache.scene_metadata_path(
            env_cache.path, scene.env_name, scene.name
        )
        if scene_utils.enforce_desired_dt(scene, self.desired_dt):
            temp_cache_path: str = str(self.temp_cache_path, encoding="utf-8")
            return (
                str(orig_scene_path),
                str(TemporaryCache(temp_cache_path).cache(scene)),
            )
        else:
            return (str(orig_scene_path), None)

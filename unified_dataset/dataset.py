import dill
from math import ceil
from tqdm import tqdm
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from torch.utils.data import Dataset
from multiprocessing import Manager

from unified_dataset.data_structures import UnifiedBatchElement, SceneMetadata, SceneTimeMetadata, SceneTimeNodeMetadata, EnvMetadata, Agent
from unified_dataset.utils import string_utils, nusc_utils, lyft_utils, env_utils

# NuScenes
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap

# Lyft Level 5
from l5kit.data import ChunkedDataset


class UnifiedDataset(Dataset):
    def __init__(self, 
                 datasets: Optional[List[str]] = None, 
                 centric: str = "agent",
                 history_sec_between: Tuple[int, int] = (0.5, 1),
                 future_sec_between: Tuple[int, int] = (0.5, 3),
                 data_dirs: Dict[str, str] = {'nusc': '~/datasets/nuScenes',
                                              'nusc_mini': '~/datasets/nuScenes',
                                              'lyft_sample': '~/datasets/lyft/scenes/sample.zarr'},
                 cache_location: str = '~/.unified_data_cache',
                 rebuild_cache: bool = False) -> None:
        
        self.rebuild_cache = rebuild_cache
        self.cache_dir = Path(cache_location).expanduser().resolve()
        if not self.cache_dir.is_dir():
            self.cache_dir.mkdir()

        self.history_sec_between = history_sec_between
        self.future_sec_between = future_sec_between

        self.envs_dict: Dict[str, EnvMetadata] = env_utils.get_env_metadata(data_dirs)

        self.all_components = list()
        for env in self.envs_dict.values():
            self.all_components += env.components
        
        matching_datasets: List[Tuple[str, ...]] = self.get_matching_datasets(datasets)
        print('Loading data for matched datasets:', string_utils.pretty_string_tuples(matching_datasets), flush=True)

        if any('nusc' in dataset_tuple for dataset_tuple in matching_datasets):
            print('Loading nuScenes dataset...', flush=True)
            self.nusc_obj: NuScenes = NuScenes(version='v1.0-trainval', dataroot=self.envs_dict['nusc'].data_dir)

        if any('nusc_mini' in dataset_tuple for dataset_tuple in matching_datasets):
            print('Loading nuScenes mini dataset...', flush=True)
            self.nusc_mini_obj: NuScenes = NuScenes(version='v1.0-mini', dataroot=self.envs_dict['nusc_mini'].data_dir)

        if any('lyft_sample' in dataset_tuple for dataset_tuple in matching_datasets):
            print('Loading lyft sample dataset...', flush=True)
            self.lyft_sample_obj: ChunkedDataset = ChunkedDataset(str(self.envs_dict['lyft_sample'].data_dir)).open()

        self.scene_index: List[SceneMetadata] = self.create_scene_metadata(matching_datasets)
        print(self.scene_index)
        
        self.data_index = list()
        if centric == "scene":
            for scene_info in self.scene_index:
                self.data_index += [(scene_info.env_name, scene_info.name, ts) for ts in range(scene_info.length_timesteps)]
        elif centric == "agent":
            for scene_info in self.scene_index:
                for ts in range(scene_info.length_timesteps):
                    self.data_index += [(scene_info.env_name, scene_info.name, ts, agent_id) for agent_id in scene_info.agent_presence[ts]]

        manager = Manager()
        self.data_index = manager.list(self.data_index)
        self.data_len: int = len(self.data_index)

    def get_matching_datasets(self, queries: Optional[List[str]]) -> List[Tuple[str, ...]]:
        if queries is None:
            return self.all_components

        dataset_tuples = [set(data.split('-')) for data in queries]

        matching_datasets = list()
        for dataset_tuple in dataset_tuples:
            for dataset_component in self.all_components:
                if dataset_tuple.issubset(dataset_component):
                    matching_datasets.append(dataset_component)

        return matching_datasets

    def create_scene_metadata(self, datasets: List[Tuple[str, ...]]) -> List[SceneMetadata]:
        scenes_list: List[SceneMetadata] = list()
        for dataset_tuple in tqdm(datasets, desc='Loading Scene Metadata'):
            if 'nusc_mini' in dataset_tuple:
                scenes_list += nusc_utils.get_matching_scenes(self.nusc_mini_obj, self.envs_dict['nusc_mini'], dataset_tuple)

            if 'lyft_sample' in dataset_tuple:
                scenes_list += lyft_utils.get_matching_scenes(self.lyft_sample_obj, self.envs_dict['lyft_sample'], dataset_tuple)

        self.calculate_agent_presence(scenes_list)
        return scenes_list

    def calculate_agent_presence(self, scenes: List[SceneMetadata]) -> None:
        scene_info: SceneMetadata
        for scene_info in tqdm(scenes, desc='Calculating Agent Presence'):
            cache_scene_dir: Path = self.cache_dir / scene_info.env_name / scene_info.name
            if not cache_scene_dir.is_dir():
                cache_scene_dir.mkdir(parents=True)

            scene_file: Path = cache_scene_dir / "scene_metadata.dill"
            if scene_file.is_file() and not self.rebuild_cache:
                with open(scene_file, 'rb') as f:
                    cached_scene_info: SceneMetadata = dill.load(f)
                
                scene_info.update_agent_presence(cached_scene_info.agent_presence)
                continue

            if scene_info.env_name == 'nusc_mini':
                agent_presence = nusc_utils.calc_agent_presence(scene_info=scene_info,
                                                                nusc_obj=self.nusc_mini_obj,
                                                                cache_scene_dir=cache_scene_dir,
                                                                rebuild_cache=self.rebuild_cache)
            elif scene_info.env_name == 'lyft_sample':
                agent_presence = lyft_utils.calc_agent_presence(scene_info=scene_info,
                                                                lyft_obj=self.lyft_sample_obj,
                                                                cache_scene_dir=cache_scene_dir,
                                                                rebuild_cache=self.rebuild_cache)

            scene_info.update_agent_presence(agent_presence)
            
            with open(scene_file, 'wb') as f:
                dill.dump(scene_info, f)

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, idx) -> UnifiedBatchElement:
        # print(self.data_index[idx])
        return UnifiedBatchElement(idx)

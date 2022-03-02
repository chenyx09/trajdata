import dill
from math import ceil
from tqdm import tqdm
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from torch.utils.data import Dataset
from multiprocessing import Manager

from unified_dataset.data_structures import UnifiedBatchElement, SceneMetadata, SceneTimeMetadata, SceneTimeNodeMetadata, EnvMetadata, Agent
from unified_dataset.utils import string_utils, nusc_utils, lyft_utils

# NuScenes
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.utils.splits import create_splits_scenes

# Lyft Level 5
from l5kit.data import ChunkedDataset


cache_location = '/home/bivanovic/.unified_data_cache'

nusc_env = EnvMetadata(
    name='nusc',
    data_dir='/home/bivanovic/datasets/nuScenes',
    dt=0.5,
    parts=[ # nuScenes possibilities are the Cartesian product of these
        ('train', 'val', 'test'),
        ('boston', 'singapore')
    ]
)

nusc_mini_env = EnvMetadata(
    name='nusc_mini',
    data_dir='/home/bivanovic/datasets/nuScenes',
    dt=0.5,
    parts=[ # nuScenes mini possibilities are the Cartesian product of these
        ('train', 'val'),
        ('boston', 'singapore')
    ]
)

lyft_sample_env = EnvMetadata(
    name='lyft_sample',
    data_dir='/home/bivanovic/datasets/lyft/scenes/sample.zarr',
    dt=0.1,
    parts=[ # Lyft Level 5 Sample dataset possibilities are the Cartesian product of these
        ('palo_alto', )
    ]
)

all_components = list()
for env in [nusc_env, nusc_mini_env, lyft_sample_env]:
    all_components += env.components

class UnifiedDataset(Dataset):
    def __init__(self, 
                 datasets: Optional[List[str]] = None, 
                 centric: str = "agent",
                 history_sec_between: Tuple[int, int] = (0.5, 1),
                 future_sec_between: Tuple[int, int] = (0.5, 3),
                 rebuild_cache: bool = False) -> None:
        
        self.rebuild_cache = rebuild_cache
        self.cache_dir = Path(cache_location)
        if not self.cache_dir.is_dir():
            self.cache_dir.mkdir()

        self.history_sec_between = history_sec_between
        self.future_sec_between = future_sec_between
        
        matching_datasets: List[Tuple[str]] = self.get_matching_datasets(datasets)
        print('Loading data for matched datasets:', string_utils.pretty_string_tuples(matching_datasets), flush=True)

        if any('nusc' in dataset_tuple for dataset_tuple in matching_datasets):
            print('Loading nuScenes dataset...', flush=True)
            all_scene_splits: Dict[str, List[str]] = create_splits_scenes()
            nusc_scene_splits: Dict[str, List[str]] = {k: all_scene_splits[k] for k in ['train', 'val', 'test']}

            # Inverting the dict from above, associating every scene with its data split.
            self.nusc_scene_split_map = {v_elem: k for k, v in nusc_scene_splits.items() for v_elem in v}
            self.nusc_obj: NuScenes = NuScenes(version='v1.0-trainval', dataroot=nusc_env.data_dir)

        if any('nusc_mini' in dataset_tuple for dataset_tuple in matching_datasets):
            print('Loading nuScenes mini dataset...', flush=True)
            all_scene_splits: Dict[str, List[str]] = create_splits_scenes()
            nusc_mini_scene_splits: Dict[str, List[str]] = {k: all_scene_splits[k] for k in ['mini_train', 'mini_val']}

            # Renaming keys
            nusc_mini_scene_splits['train'] = nusc_mini_scene_splits.pop('mini_train')
            nusc_mini_scene_splits['val'] = nusc_mini_scene_splits.pop('mini_val')

            # Inverting the dict from above, associating every scene with its data split.
            self.nusc_mini_scene_split_map: Dict[str, str] = {v_elem: k for k, v in nusc_mini_scene_splits.items() for v_elem in v}
            self.nusc_mini_obj: NuScenes = NuScenes(version='v1.0-mini', dataroot=nusc_mini_env.data_dir)

        if any('lyft_sample' in dataset_tuple for dataset_tuple in matching_datasets):
            print('Loading lyft sample dataset...', flush=True)

            self.lyft_sample_obj: ChunkedDataset = ChunkedDataset(lyft_sample_env.data_dir).open()

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

    def get_matching_datasets(self, queries: Optional[List[str]]) -> List[Tuple[str]]:
        if queries is None:
            return all_components

        dataset_tuples = [set(data.split('-')) for data in queries]

        matching_datasets = list()
        for dataset_tuple in dataset_tuples:
            for dataset_component in all_components:
                if dataset_tuple.issubset(dataset_component):
                    matching_datasets.append(dataset_component)

        return matching_datasets

    def create_scene_metadata(self, datasets: List[Tuple[str]]) -> List[SceneMetadata]:
        scenes_list: List[SceneMetadata] = list()
        for dataset_tuple in tqdm(datasets, desc='Loading Scene Metadata'):
            if 'nusc_mini' in dataset_tuple:
                for scene_record in self.nusc_mini_obj.scene:
                    scene_name: str = scene_record['name']
                    scene_location: str = self.nusc_mini_obj.get('log', scene_record['log_token'])['location']
                    scene_split: str = self.nusc_mini_scene_split_map[scene_name]
                    scene_length: int = scene_record['nbr_samples']
                    
                    if scene_location.split('-')[0] in dataset_tuple and scene_split in dataset_tuple:
                        scene_metadata = SceneMetadata(nusc_mini_env, scene_name, scene_location, scene_split, scene_length, scene_record)
                        scenes_list.append(scene_metadata)

            if 'lyft_sample' in dataset_tuple:
                all_scene_frames = self.lyft_sample_obj.scenes['frame_index_interval']
                
                for idx in range(all_scene_frames.shape[0]):
                    scene_length: int = (all_scene_frames[idx, 1] - all_scene_frames[idx, 0]).item() # Doing .item() otherwise it'll be a numpy.int64
                    scene_metadata = SceneMetadata(lyft_sample_env, f'scene-{idx:04d}', 'palo_alto', '', scene_length, all_scene_frames[idx])
                    scenes_list.append(scene_metadata)

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
                                                                nusc_obj=self.lyft_sample_obj,
                                                                cache_scene_dir=cache_scene_dir,
                                                                rebuild_cache=self.rebuild_cache)

            scene_info.update_agent_presence(agent_presence)
            with open(scene_file, 'wb') as f:
                dill.dump(scene_info, f)

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, idx) -> UnifiedBatchElement:
        print(self.data_index[idx])
        return UnifiedBatchElement(idx)

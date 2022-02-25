import dill
from math import ceil
from tqdm import tqdm
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from torch.utils.data import Dataset

from unified_dataset.data_structures import UnifiedBatchElement, SceneMetadata, SceneTimeMetadata, SceneTimeNodeMetadata, EnvMetadata, Agent
from unified_dataset.utils import string_utils, nusc_utils

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
                 centric: str = "node",
                 history_sec_between: Tuple[int, int] = (0.5, 1),
                 future_sec_between: Tuple[int, int] = (0.5, 3),
                 rebuild_cache: bool = False) -> None:
        
        self.rebuild_cache = rebuild_cache
        cache_dir = Path(cache_location)
        if not cache_dir.is_dir():
            cache_dir.mkdir()

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

        self.scene_index: List[SceneMetadata] = self.load_scene_metadata(matching_datasets)
        print(self.scene_index)
        
        if centric == "scene":
            self.data_index = self.create_scene_timestep_metadata(self.scene_index)
        elif centric == "node":
            self.data_index = self.create_scene_timestep_node_metadata(self.scene_index)

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

    def load_scene_metadata(self, datasets: List[Tuple[str]]) -> List[SceneMetadata]:
        scenes_list: List[SceneMetadata] = list()
        for dataset_tuple in tqdm(datasets, desc='Loading Scene Metadata'):
            if 'nusc_mini' in dataset_tuple:
                for scene_record in tqdm(self.nusc_mini_obj.scene):
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

        return scenes_list

    def create_scene_timestep_metadata(self, scenes: List[SceneMetadata]) -> List[SceneTimeMetadata]:
        scene_time_index = list()
        cache = Path(cache_location)

        scene_info: SceneMetadata
        for scene_info in scenes:
            if scene_info.env_name == 'nusc_mini':
                cache_scene_dir: Path = cache / scene_info.env_name / scene_info.name
                if not cache_scene_dir.is_dir():
                    cache_scene_dir.mkdir(parents=True)

                agent_presence: Dict[int, List[str]] = dict()
                for frame_idx, frame_info in enumerate(nusc_utils.frame_iterator(self.nusc_mini_obj, scene_info)):
                    
                    # TODO: Finish this all
                    ego_pose = nusc_utils.get_ego_pose(self.nusc_mini_obj, frame_info)

                    agent_presence[frame_idx] = list()
                    for agent_info in nusc_utils.agent_iterator(self.nusc_mini_obj, frame_info):
                        agent_id: str = agent_info['instance_token']
                        agent_file: Path = cache_scene_dir / f"{agent_id}.dill"
                        agent_presence[frame_idx].append(agent_id)
                        if agent_file.is_file() and not self.rebuild_cache:
                            continue
                        
                        agent: Agent = nusc_utils.agg_agent_data(self.nusc_mini_obj, agent_info, frame_idx)
                        dill.dump(agent, agent_file)

                scene_time_index += [(scene_info, i) for i in range(scene_info.length_timesteps)]

        return scene_time_index

    def create_scene_timestep_node_metadata(self, scenes: List[SceneMetadata]) -> List[SceneTimeNodeMetadata]:
        pass

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, idx) -> UnifiedBatchElement:
        return UnifiedBatchElement(idx)

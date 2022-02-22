import itertools
from tqdm import tqdm
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from torch.utils.data import Dataset

from unified_dataset.data_structures import UnifiedBatchElement, SceneMetadata, SceneTimeMetadata, SceneTimeNodeMetadata
from unified_dataset.utils import pretty_string_tuples

# NuScenes
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.utils.splits import create_splits_scenes


dataset_locations = {'nusc': '/home/bivanovic/datasets/nuScenes'}
cache_location = '/home/bivanovic/.unified_data_cache'

# nuScenes possibilities are the Cartesian product of these
nusc_parts = [
    ('nusc', ),
    ('train', 'val', 'test'),
    ('boston', 'singapore')
]
nusc_components = list(itertools.product(*nusc_parts))

# nuScenes-mini possibilities are the Cartesian product of these
nusc_mini_parts = [
    ('nusc_mini', ),
    ('train', 'val'),
    ('boston', 'singapore')
]
nusc_mini_components = list(itertools.product(*nusc_mini_parts))

all_components = nusc_components + nusc_mini_components

class UnifiedDataset(Dataset):
    def __init__(self, datasets: Optional[List[str]] = None, centric: str = "node") -> None:
        cache_dir = Path(cache_location)
        if not cache_dir.is_dir():
            cache_dir.mkdir()

        matching_datasets: List[Tuple[str]] = self.get_matching_datasets(datasets)
        print('Loading data for matched datasets:', pretty_string_tuples(matching_datasets), flush=True)

        if any('nusc' in dataset_tuple for dataset_tuple in matching_datasets):
            print('Loading nuScenes dataset...', flush=True)
            all_scene_splits: Dict[str, List[str]] = create_splits_scenes()
            nusc_scene_splits: Dict[str, List[str]] = {k: all_scene_splits[k] for k in ['train', 'val', 'test']}

            # Inverting the dict from above, associating every scene with its data split.
            self.nusc_scene_split_map = {v_elem: k for k, v in nusc_scene_splits.items() for v_elem in v}
            self.nusc_obj: NuScenes = NuScenes(version='v1.0-trainval', dataroot=dataset_locations['nusc'])

        if any('nusc_mini' in dataset_tuple for dataset_tuple in matching_datasets):
            print('Loading nuScenes mini dataset...', flush=True)
            all_scene_splits: Dict[str, List[str]] = create_splits_scenes()
            nusc_mini_scene_splits: Dict[str, List[str]] = {k: all_scene_splits[k] for k in ['mini_train', 'mini_val']}

            # Renaming keys
            nusc_mini_scene_splits['train'] = nusc_mini_scene_splits.pop('mini_train')
            nusc_mini_scene_splits['val'] = nusc_mini_scene_splits.pop('mini_val')

            # Inverting the dict from above, associating every scene with its data split.
            self.nusc_mini_scene_split_map: Dict[str, str] = {v_elem: k for k, v in nusc_mini_scene_splits.items() for v_elem in v}
            self.nusc_mini_obj: NuScenes = NuScenes(version='v1.0-mini', dataroot=dataset_locations['nusc'])

        self.scene_index: List[SceneMetadata] = self.load_scene_metadata(matching_datasets)
        print(self.scene_index)
        raise
        
        if centric == "scene":
            self.data_index = self.load_scene_timestep_metadata(self.scene_index)
        elif centric == "node":
            self.data_index = self.load_scene_timestep_node_metadata(self.scene_index)

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
        scenes_list = list()
        for dataset_tuple in tqdm(datasets, desc='Loading Scene Metadata'):
            if 'nusc_mini' in dataset_tuple:
                for scene_record in tqdm(self.nusc_mini_obj.scene):
                    scene_name: str = scene_record['name']
                    scene_location: str = self.nusc_mini_obj.get('log', scene_record['log_token'])['location']
                    scene_split: str = self.nusc_mini_scene_split_map[scene_name]

                    if scene_location.split('-')[0] in dataset_tuple and scene_split in dataset_tuple:
                        scene_metadata = SceneMetadata('nusc_mini', scene_name, scene_location, scene_split, scene_record)
                        scenes_list.append(scene_metadata)

        return scenes_list

    def load_scene_timestep_metadata(self, scenes: List[SceneMetadata]) -> List[SceneTimeMetadata]:
        pass

    def load_scene_timestep_node_metadata(self, scenes: List[SceneMetadata]) -> List[SceneTimeNodeMetadata]:
        pass

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, idx) -> UnifiedBatchElement:
        return UnifiedBatchElement(idx)

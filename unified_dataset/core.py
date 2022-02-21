from typing import List, Mapping
from torch.utils.data import Dataset

from .batch_item import SingleDatum

dataset_locations = {'nuScenes': '~/datasets/nuScenes'}
dataset_components = {'nuScenes': ['nusc-boston', 'nusc-singapore']}

class UnifiedDataset(Dataset):
    def __init__(self, data: List[str], centric: str = "node") -> None:
        pass

    def __len__(self) -> int:
        return 64

    def __getitem__(self, idx) -> SingleDatum:
        return SingleDatum(idx)

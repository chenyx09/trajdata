from typing import Dict, List, Type

from avdata.dataset_specific import RawDataset
from avdata.dataset_specific.lyft import LyftDataset
from avdata.dataset_specific.nusc import NuscDataset


def get_raw_datasets(data_dirs: Dict[str, str]) -> List[Type[RawDataset]]:
    raw_datasets: List[Type[RawDataset]] = list()

    if "nusc" in data_dirs:
        raw_datasets.append(NuscDataset("nusc", data_dirs["nusc"]))

    if "nusc_mini" in data_dirs:
        raw_datasets.append(NuscDataset("nusc_mini", data_dirs["nusc_mini"]))

    if "lyft_sample" in data_dirs:
        raw_datasets.append(LyftDataset("lyft_sample", data_dirs["lyft_sample"]))

    return raw_datasets

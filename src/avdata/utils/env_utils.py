from typing import Dict, List, Type

from avdata.data_structures import EnvMetadata
from avdata.dataset_specific import LyftDataset, NuscDataset, RawDataset
from avdata.utils import lyft_utils, nusc_utils


def get_raw_datasets(data_dirs: Dict[str, str]) -> List[Type[RawDataset]]:
    raw_datasets: List[Type[RawDataset]] = list()

    if "nusc" in data_dirs:
        env_metadata: EnvMetadata = nusc_utils.get_env_metadata(
            "nusc", data_dirs["nusc"]
        )
        raw_datasets.append(NuscDataset(env_metadata))

    if "nusc_mini" in data_dirs:
        env_metadata: EnvMetadata = nusc_utils.get_env_metadata(
            "nusc_mini", data_dirs["nusc_mini"]
        )
        raw_datasets.append(NuscDataset(env_metadata))

    if "lyft_sample" in data_dirs:
        env_metadata: EnvMetadata = lyft_utils.get_env_metadata(
            "lyft_sample", data_dirs["lyft_sample"]
        )
        raw_datasets.append(LyftDataset(env_metadata))

    return raw_datasets

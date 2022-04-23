from typing import Dict, List, Type

from avdata.dataset_specific import RawDataset
from avdata.dataset_specific.lyft import LyftDataset
from avdata.dataset_specific.nusc import NuscDataset


def get_raw_datasets(data_dirs: Dict[str, str]) -> List[RawDataset]:
    raw_datasets: List[RawDataset] = list()

    if "nusc" in data_dirs:
        raw_datasets.append(
            NuscDataset("nusc", data_dirs["nusc"], parallelizable=False)
        )

    if "nusc_mini" in data_dirs:
        raw_datasets.append(
            NuscDataset("nusc_mini", data_dirs["nusc_mini"], parallelizable=False)
        )

    if "lyft_sample" in data_dirs:
        raw_datasets.append(
            LyftDataset("lyft_sample", data_dirs["lyft_sample"], parallelizable=True)
        )

    if "lyft_train" in data_dirs:
        raw_datasets.append(
            LyftDataset("lyft_train", data_dirs["lyft_train"], parallelizable=True)
        )

    if "lyft_train_full" in data_dirs:
        raw_datasets.append(
            LyftDataset(
                "lyft_train_full", data_dirs["lyft_train_full"], parallelizable=True
            )
        )

    if "lyft_val" in data_dirs:
        raw_datasets.append(
            LyftDataset("lyft_val", data_dirs["lyft_val"], parallelizable=True)
        )

    return raw_datasets

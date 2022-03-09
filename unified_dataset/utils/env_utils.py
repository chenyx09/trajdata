from collections import defaultdict
from typing import Dict, List

from nuscenes.utils.splits import create_splits_scenes

from unified_dataset.data_structures import EnvMetadata
from unified_dataset.utils.lyft_utils import LYFT_DT
from unified_dataset.utils.nusc_utils import NUSC_DT


def get_env_metadata(data_dirs: Dict[str, str]) -> Dict[str, EnvMetadata]:
    env_metadata: Dict[str, EnvMetadata] = dict()

    if "nusc" in data_dirs:
        all_scene_splits: Dict[str, List[str]] = create_splits_scenes()
        nusc_scene_splits: Dict[str, List[str]] = {
            k: all_scene_splits[k] for k in ["train", "val", "test"]
        }

        # Inverting the dict from above, associating every scene with its data split.
        nusc_scene_split_map: Dict[str, str] = {
            v_elem: k for k, v in nusc_scene_splits.items() for v_elem in v
        }

        nusc_env = EnvMetadata(
            name="nusc",
            data_dir=data_dirs["nusc"],
            dt=NUSC_DT,
            parts=[  # nuScenes possibilities are the Cartesian product of these
                ("train", "val", "test"),
                ("boston", "singapore"),
            ],
            scene_split_map=nusc_scene_split_map,
        )
        env_metadata["nusc"] = nusc_env

    if "nusc_mini" in data_dirs:
        all_scene_splits: Dict[str, List[str]] = create_splits_scenes()
        nusc_mini_scene_splits: Dict[str, List[str]] = {
            k: all_scene_splits[k] for k in ["mini_train", "mini_val"]
        }

        # Renaming keys
        nusc_mini_scene_splits["train"] = nusc_mini_scene_splits.pop("mini_train")
        nusc_mini_scene_splits["val"] = nusc_mini_scene_splits.pop("mini_val")

        # Inverting the dict from above, associating every scene with its data split.
        nusc_mini_scene_split_map: Dict[str, str] = {
            v_elem: k for k, v in nusc_mini_scene_splits.items() for v_elem in v
        }

        nusc_mini_env = EnvMetadata(
            name="nusc_mini",
            data_dir=data_dirs["nusc_mini"],
            dt=NUSC_DT,
            parts=[  # nuScenes mini possibilities are the Cartesian product of these
                ("train", "val"),
                ("boston", "singapore"),
            ],
            scene_split_map=nusc_mini_scene_split_map,
        )
        env_metadata["nusc_mini"] = nusc_mini_env

    if "lyft_sample" in data_dirs:
        lyft_sample_env = EnvMetadata(
            name="lyft_sample",
            data_dir=data_dirs["lyft_sample"],
            dt=LYFT_DT,
            parts=[  # Lyft Level 5 Sample dataset possibilities are the Cartesian product of these
                ("palo_alto",)
            ],
            scene_split_map=defaultdict(lambda: ""),
        )
        env_metadata["lyft_sample"] = lyft_sample_env

    return env_metadata

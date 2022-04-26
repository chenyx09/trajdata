from collections import defaultdict

from torch.utils.data import DataLoader
from tqdm import tqdm

from avdata import AgentType, UnifiedDataset
from avdata.augmentation import NoiseHistories


# @profile
def main():
    # noise_hists = NoiseHistories()

    dataset = UnifiedDataset(
        desired_data=[
            "nusc",
            "nusc_mini",
            "lyft_sample",
            "lyft_train",
            "lyft_train_full",
            "lyft_val",
        ],
        # history_sec=(1.5, 1.5),
        # future_sec=(5.0, 5.0),
        # desired_dt=0.1,
        # only_types=[AgentType.VEHICLE],
        # agent_interaction_distances=defaultdict(lambda: 50.0),
        # incl_robot_future=False,
        # incl_map=True,
        # map_params={"px_per_m": 2, "map_size_px": 224, "offset_frac_xy": (-0.5, 0.0)},
        # augmentations=[noise_hists],
        data_dirs={
            "nusc": "/workspace/datasets/nuScenes",
            "nusc_mini": "/workspace/datasets/nuScenes",
            "lyft_sample": "/workspace/datasets/lyft/lyft_prediction/scenes/sample.zarr",
            "lyft_train": "/workspace/datasets/lyft/lyft_prediction/scenes/train.zarr",
            "lyft_train_full": "/workspace/datasets/lyft/lyft_prediction/scenes/train_full.zarr",
            "lyft_val": "/workspace/datasets/lyft/lyft_prediction/scenes/validate.zarr",
        },
        cache_location="/workspace/av_cache",
        rebuild_cache=True,
        rebuild_maps=True,
        num_workers=64,
        verbose=True,
    )

    print(f"Total Data Samples: {len(dataset):,}")

    # dataloader = DataLoader(
    #     dataset,
    #     batch_size=64,
    #     shuffle=False,
    #     collate_fn=dataset.get_collate_fn(),
    #     num_workers=64,
    # )

    # for batch in tqdm(dataloader):
    #     pass


if __name__ == "__main__":
    main()

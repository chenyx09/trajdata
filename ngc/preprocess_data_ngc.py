from avdata import UnifiedDataset


def main():
    dataset = UnifiedDataset(
        desired_data=[
            # "nusc",
            # "nusc_mini",
            # "lyft_sample",
            # "lyft_train",
            "lyft_train_full",
            # "lyft_val",
        ],
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
        # rebuild_maps=True,
        num_workers=64,
        verbose=True,
    )

    print(f"Total Data Samples: {len(dataset):,}")


if __name__ == "__main__":
    main()

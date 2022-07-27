from trajdata import UnifiedDataset


def main():
    dataset = UnifiedDataset(
        desired_data=[
            "nusc_trainval",
            "nusc_mini",
            "lyft_sample",
            "lyft_train",
            # "lyft_train_full",
            "lyft_val",
        ],
        data_dirs={
            "nusc_trainval": "/workspace/nuScenes",
            "nusc_mini": "/workspace/nuScenes",
            "lyft_sample": "/workspace/lyft/lyft_prediction/scenes/sample.zarr",
            "lyft_train": "/workspace/lyft/lyft_prediction/scenes/train.zarr",
            # "lyft_train_full": "/workspace/lyft/lyft_prediction/scenes/train_full.zarr",
            "lyft_val": "/workspace/lyft/lyft_prediction/scenes/validate.zarr",
        },
        cache_location="/workspace/trajdata_cache",
        rebuild_cache=True,
        rebuild_maps=True,
        num_workers=64,
        verbose=True,
    )

    print(f"Total Data Samples: {len(dataset):,}")


if __name__ == "__main__":
    main()

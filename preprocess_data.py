import os

from avdata import UnifiedDataset


# @profile
def main():
    dataset = UnifiedDataset(
        desired_data=["nusc_mini", "lyft_sample"],
        rebuild_cache=True,
        rebuild_maps=True,
        num_workers=os.cpu_count(),
        verbose=True,
    )
    print(f"Total Data Samples: {len(dataset):,}")


if __name__ == "__main__":
    main()

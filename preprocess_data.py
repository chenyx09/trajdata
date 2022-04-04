import os

from avdata import UnifiedDataset


# @profile
def main():
    dataset = UnifiedDataset(
        desired_data=["nusc_mini", "lyft_sample"],
        rebuild_cache=True,
        num_workers=os.cpu_count(),
    )
    print(f"Total Data Samples: {len(dataset):,}")


if __name__ == "__main__":
    main()

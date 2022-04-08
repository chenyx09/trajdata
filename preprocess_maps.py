import os

from avdata import UnifiedDataset


# @profile
def main():
    dataset = UnifiedDataset(
        desired_data=["nusc_mini"],
        rebuild_maps=True,
        num_workers=os.cpu_count(),
    )
    print(f"Finished Caching Maps!")


if __name__ == "__main__":
    main()

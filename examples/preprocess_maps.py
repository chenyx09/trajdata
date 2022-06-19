import os

from avdata import UnifiedDataset


# @profile
def main():
    dataset = UnifiedDataset(
        desired_data=["nusc_mini", "lyft_sample"],
        rebuild_maps=True,
    )
    print(f"Finished Caching Maps!")


if __name__ == "__main__":
    main()
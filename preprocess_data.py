from avdata import UnifiedDataset


# @profile
def main():
    dataset = UnifiedDataset(datasets=["nusc_mini", "lyft_sample"], rebuild_cache=True)
    print(len(dataset))


if __name__ == "__main__":
    main()

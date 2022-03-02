from tqdm import tqdm, trange
from unified_dataset import UnifiedDataset, unified_collate, UnifiedBatch
from torch.utils.data import DataLoader

# @profile
def main():
    dataset = UnifiedDataset(datasets=['nusc_mini'],
                             centric='agent',
                             history_sec_between=(1, 2), # Both inclusive
                             future_sec_between=(3, 3))  # Both inclusive
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=unified_collate, num_workers=0)

    for epoch in trange(100):
        batch: UnifiedBatch
        for batch in dataloader:
            # print(batch.nums)
            pass


if __name__ == "__main__":
    main()

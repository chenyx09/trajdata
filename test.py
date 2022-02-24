from tqdm import tqdm
from unified_dataset import UnifiedDataset, unified_collate, UnifiedBatch
from torch.utils.data import DataLoader

def main():
    dataset = UnifiedDataset(datasets=['nusc_mini', 'lyft_sample'])
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=unified_collate)

    batch: UnifiedBatch
    for batch in tqdm(dataloader):
        print(batch.nums)


if __name__ == "__main__":
    main()

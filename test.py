from tqdm import tqdm
from unified_dataset import UnifiedDataset, unified_collate
from torch.utils.data import DataLoader

def main():
    dataset = UnifiedDataset(data=['nuScenes'])
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=unified_collate)

    for batch in tqdm(dataloader):
        print(batch)


if __name__ == "__main__":
    main()

from tqdm import tqdm, trange
from unified_dataset import UnifiedDataset, AgentBatch, AgentType
from torch.utils.data import DataLoader

# @profile
def main():
    dataset = UnifiedDataset(datasets=['nusc_mini', 'lyft_sample'],
                             centric='agent',
                             history_sec_at_most=2.7,
                             future_sec_at_most=3.4,
                             only_types=None)
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=dataset.collate_fn, num_workers=0)

    batch: AgentBatch
    for batch in tqdm(dataloader):
        # print(batch.nums)
        pass


if __name__ == "__main__":
    main()

from tqdm import tqdm, trange
from unified_dataset import UnifiedDataset, AgentBatch, AgentType
from torch.utils.data import DataLoader

# @profile
def main():
    dataset = UnifiedDataset(datasets=['nusc_mini', 'lyft_sample'],
                             centric='agent',
                             history_sec=(0.1, 1.0),
                             future_sec=(0.1, 2.0),
                             incl_robot_future=True,
                             no_types=[AgentType.UNKNOWN])
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=dataset.collate_fn, num_workers=0)

    batch: AgentBatch
    for batch in tqdm(dataloader):
        # print(batch.nums)
        pass


if __name__ == "__main__":
    main()

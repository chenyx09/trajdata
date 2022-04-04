import os

from torch.utils.data import DataLoader
from tqdm import tqdm

from avdata import AgentBatch, AgentType, UnifiedDataset


# @profile
def main():
    dataset = UnifiedDataset(
        desired_data=["nusc_mini", "lyft_sample"],
        centric="agent",
        history_sec=(0.1, 1.0),
        future_sec=(0.1, 2.0),
        incl_robot_future=True,
        num_workers=os.cpu_count(),
    )

    print(f"# Data Samples: {len(dataset):,}")

    # TODO(bivanovic): Create a method like finalize() which writes all the batch information to a TFRecord?

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=0,
    )

    batch: AgentBatch
    for batch in tqdm(dataloader):
        pass


if __name__ == "__main__":
    main()

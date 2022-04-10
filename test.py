from collections import defaultdict
import os

from torch.utils.data import DataLoader
from tqdm import tqdm

from avdata import AgentBatch, AgentType, UnifiedDataset
from avdata.visualization.vis import plot_agent_batch


# @profile
def main():
    dataset = UnifiedDataset(
        desired_data=["nusc_mini"],
        centric="agent",
        history_sec=(1.5, 1.5),
        future_sec=(5.0, 5.0),
        only_types=[AgentType.VEHICLE],
        agent_interaction_distances=defaultdict(lambda: 30.0),
        # incl_robot_future=True,
        incl_map=True,
        map_params={"px_per_m": 2, "map_size_px": 50},
        num_workers=4,
        verbose=True,
    )

    print(f"# Data Samples: {len(dataset):,}")

    # TODO(bivanovic): Create a method like finalize() which writes all the batch information to a TFRecord?

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        # num_workers=os.cpu_count(),
    )

    batch: AgentBatch
    for batch in tqdm(dataloader):
        pass
        # plot_agent_batch(batch, batch_idx=0)


if __name__ == "__main__":
    main()

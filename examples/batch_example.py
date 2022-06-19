from collections import defaultdict

from torch.utils.data import DataLoader
from tqdm import tqdm

from avdata import AgentBatch, AgentType, UnifiedDataset
from avdata.augmentation import NoiseHistories
from avdata.visualization.vis import plot_agent_batch


def main():
    noise_hists = NoiseHistories()

    dataset = UnifiedDataset(
        desired_data=["nusc_mini-mini_train"],
        centric="agent",
        desired_dt=0.1,
        history_sec=(3.2, 3.2),
        future_sec=(4.8, 4.8),
        only_types=[AgentType.VEHICLE],
        agent_interaction_distances=defaultdict(lambda: 30.0),
        incl_robot_future=True,
        incl_map=True,
        map_params={"px_per_m": 2, "map_size_px": 224, "offset_frac_xy": (-0.5, 0.0)},
        augmentations=[noise_hists],
        num_workers=0,
        verbose=True,
    )

    print(f"# Data Samples: {len(dataset):,}")

    # TODO(bivanovic): Create a method like finalize() which writes all the batch information to a TFRecord?

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=dataset.get_collate_fn(),
        num_workers=4,
        persistent_workers=True,
    )

    batch: AgentBatch
    for batch in tqdm(dataloader):
        plot_agent_batch(batch, batch_idx=0)


if __name__ == "__main__":
    main()

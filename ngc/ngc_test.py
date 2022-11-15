import os
from collections import defaultdict

from torch.utils.data import DataLoader
from tqdm import tqdm

from trajdata import AgentBatch, AgentType, UnifiedDataset
from trajdata.augmentation import NoiseHistories
from trajdata.visualization.vis import plot_agent_batch


# @profile
def main():
    noise_hists = NoiseHistories()

    dataset = UnifiedDataset(
        desired_data=["nusc"],
        centric="agent",
        # desired_dt=0.1,
        history_sec=(0.1, 1.5),
        future_sec=(0.1, 5.0),
        only_types=[AgentType.VEHICLE, AgentType.PEDESTRIAN],
        agent_interaction_distances=defaultdict(lambda: 40.0),
        incl_robot_future=True,
        incl_raster_map=True,
        raster_map_params={
            "px_per_m": 2,
            "map_size_px": 224,
            "offset_frac_xy": (-0.5, 0.0),
        },
        # augmentations=[noise_hists],
        data_dirs={
            "nusc": "/workspace/datasets/nuScenes",
        },
        cache_location="/workspace/unified_data_cache",
        num_workers=os.cpu_count(),
        # verbose=True,
    )

    print(f"# Data Samples: {len(dataset):,}")

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=dataset.get_collate_fn(),
        num_workers=os.cpu_count(),
    )

    batch: AgentBatch
    for batch in tqdm(dataloader):
        pass
        # plot_agent_batch(batch, batch_idx=0)


if __name__ == "__main__":
    main()

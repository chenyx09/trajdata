from collections import defaultdict

from torch.utils.data import DataLoader
from tqdm import tqdm

from avdata import AgentBatch, AgentType, UnifiedDataset
from avdata.augmentation import NoiseHistories
from avdata.visualization.vis import plot_agent_batch


# @profile
def main():
    noise_hists = NoiseHistories()

    dataset = UnifiedDataset(
        desired_data=["nusc-val"],
        centric="agent",
        data_dirs= {'nusc': '/home/yuxiaoc/repos/Trajectron-plus-plus/experiments/nuScenes/v1.0-trainval_meta'},
        desired_dt=0.1,
        history_sec=(1.5, 1.5),
        future_sec=(5.0, 5.0),
        only_types=[AgentType.VEHICLE],
        agent_interaction_distances=defaultdict(lambda: 50.0),
        incl_robot_future=False,
        incl_neighbor_map=False,
        vectorize_lane="ego",
        incl_map=False,
        map_params={"px_per_m": 2, "map_size_px": 224, "offset_frac_xy": (-0.5, 0.0)},
        augmentations=[],
        num_workers=4,
        verbose=True,
        max_agent_num=None,
        standardize_data = False,
    )

    print(f"# Data Samples: {len(dataset):,}")

    # TODO(bivanovic): Create a method like finalize() which writes all the batch information to a TFRecord?

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=dataset.get_collate_fn(),
        num_workers=0,
    )

    batch: AgentBatch
    for batch in tqdm(dataloader):
        pass
        # plot_agent_batch(batch, batch_idx=0)


if __name__ == "__main__":
    main()

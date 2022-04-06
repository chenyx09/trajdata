import os

from torch.utils.data import DataLoader
from tqdm import tqdm

from avdata import AgentBatch, AgentType, UnifiedDataset
from avdata.data_structures.map import Map


# @profile
def main():
    dataset = UnifiedDataset(
        desired_data=["nusc_mini"],
        centric="agent",
        history_sec=(0.1, 1.0),
        future_sec=(0.1, 2.0),
        # only_types=[AgentType.VEHICLE],
        incl_robot_future=True,
        incl_map=True,
        map_patch_size=200,
        num_workers=os.cpu_count(),
    )

    print(f"# Data Samples: {len(dataset):,}")

    # TODO(bivanovic): Create a method like finalize() which writes all the batch information to a TFRecord?

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=os.cpu_count(),
    )

    import matplotlib.pyplot as plt

    batch: AgentBatch
    for batch in tqdm(dataloader):
        for batch_idx in range(batch.dt.shape[0]):
            fig, ax = plt.subplots()
            center = batch.agent_hist[batch_idx, -1, :2]
            patch_size = dataset.map_patch_size
            ax.imshow(
                Map.to_img(
                    batch.maps[batch_idx],
                    [[0, 1, 2], [3, 4], [5, 6]],
                ),
                origin="lower",
                extent=(
                    center[0] - patch_size // 2,
                    center[0] + patch_size // 2,
                    center[1] - patch_size // 2,
                    center[1] + patch_size // 2,
                ),
            )

            ax.scatter(
                center[0],
                center[1],
                s=20,
                c="white",
            )
            ax.plot(
                batch.agent_hist[batch_idx, :, 0]*10,
                batch.agent_hist[batch_idx, :, 1]*10,
                c="white",
            )
            ax.plot(
                batch.agent_fut[batch_idx, :, 0]*10,
                batch.agent_fut[batch_idx, :, 1]*10,
                c="grey",
            )
            ax.grid(False)
            plt.show()
            plt.close(fig)


if __name__ == "__main__":
    main()

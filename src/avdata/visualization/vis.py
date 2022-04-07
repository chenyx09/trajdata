from typing import Union

import matplotlib.pyplot as plt
from matplotlib import ticker
from torch import Tensor

from avdata.data_structures.batch import AgentBatch
from avdata.data_structures.map import Map


def plot_batch(batch: AgentBatch, batch_idx: int) -> None:
    fig, ax = plt.subplots()

    history_xy: Tensor = batch.agent_hist[batch_idx, :, :2].cpu()
    center_xy: Tensor = batch.agent_hist[batch_idx, -1, :2].cpu()
    future_xy: Tensor = batch.agent_fut[batch_idx, :, :2].cpu()

    map_res: float = (
        batch.maps_resolution[batch_idx].item()
        if batch.maps_resolution is not None
        else 1.0
    )

    if batch.maps is not None:
        patch_size: int = batch.maps[batch_idx].shape[-1]
        world_extent: float = patch_size / map_res
        ax.imshow(
            Map.to_img(
                batch.maps[batch_idx].cpu(),
                [[0, 1, 2], [3, 4], [5, 6]],
            ),
            origin="lower",
            extent=(
                center_xy[0] - world_extent // 2,
                center_xy[0] + world_extent // 2,
                center_xy[1] - world_extent // 2,
                center_xy[1] + world_extent // 2,
            ),
            alpha=0.3,
        )

    ax.scatter(center_xy[0], center_xy[1], s=20, c="orangered")
    ax.plot(history_xy[..., 0], history_xy[..., 1], c="orange")
    ax.plot(future_xy[..., 0], future_xy[..., 1], c="violet")

    num_neigh = batch.num_neigh[batch_idx]
    neighbor_hist = batch.neigh_hist[batch_idx]

    for n in range(num_neigh):
        ax.plot(neighbor_hist[n, :, 0], neighbor_hist[n, :, 1], c="olive")

    ax.scatter(
        neighbor_hist[:num_neigh, -1, 0],
        neighbor_hist[:num_neigh, -1, 1],
        s=20,
        c="gold",
    )

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    ax.grid(False)
    plt.show()
    plt.close(fig)

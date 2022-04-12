from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from torch import Tensor

from avdata.data_structures.agent import AgentType
from avdata.data_structures.batch import AgentBatch
from avdata.data_structures.map import Map


def plot_agent_batch(
    batch: AgentBatch,
    batch_idx: int,
    ax: Optional[Axes] = None,
    show: bool = True,
    close: bool = True,
) -> None:
    if ax is None:
        _, ax = plt.subplots()

    agent_name: str = batch.agent_name[batch_idx]
    agent_type: AgentType = AgentType(batch.agent_type[batch_idx].item())
    ax.set_title(f"{str(agent_type)}/{agent_name}")

    history_xy: Tensor = batch.agent_hist[batch_idx].cpu()
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

    ax.scatter(center_xy[0], center_xy[1], s=20, c="orangered", label="Agent Current")
    ax.plot(
        history_xy[..., 0],
        history_xy[..., 1],
        c="orange",
        ls="--",
        label="Agent History",
    )
    ax.quiver(
        history_xy[..., 0],
        history_xy[..., 1],
        history_xy[..., -1],
        history_xy[..., -2],
        color="k",
    )
    ax.plot(future_xy[..., 0], future_xy[..., 1], c="violet", label="Agent Future")

    num_neigh = batch.num_neigh[batch_idx]
    if num_neigh > 0:
        neighbor_hist = batch.neigh_hist[batch_idx]
        neighbor_fut = batch.neigh_fut[batch_idx]

        ax.plot([], [], c="olive", ls="--", label="Neighbor History")
        for n in range(num_neigh):
            ax.plot(neighbor_hist[n, :, 0], neighbor_hist[n, :, 1], c="olive", ls="--")

        ax.plot([], [], c="darkgreen", label="Neighbor Future")
        for n in range(num_neigh):
            ax.plot(neighbor_fut[n, :, 0], neighbor_fut[n, :, 1], c="darkgreen")

        ax.scatter(
            neighbor_hist[:num_neigh, -1, 0],
            neighbor_hist[:num_neigh, -1, 1],
            s=20,
            c="gold",
            label="Neighbor Current",
        )

    if batch.robot_fut is not None and batch.robot_fut.shape[1] > 0:
        ax.scatter(
            batch.robot_fut[batch_idx, 0, 0],
            batch.robot_fut[batch_idx, 0, 1],
            s=20,
            c="green",
            label="Ego Current",
        )
        ax.plot(
            batch.robot_fut[batch_idx, 1:, 0],
            batch.robot_fut[batch_idx, 1:, 1],
            label="Ego Future",
            c="green",
        )

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    ax.grid(False)
    ax.legend(loc="best", frameon=True)
    ax.axis("equal")

    if show:
        plt.show()

    if close:
        plt.close()

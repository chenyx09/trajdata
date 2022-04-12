from __future__ import annotations

from collections import namedtuple
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from kornia.geometry.transform import center_crop, rotate
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data._utils.collate import default_collate

from avdata.data_structures.agent import AgentType
from avdata.data_structures.batch_element import AgentBatchElement, SceneBatchElement


@dataclass
class AgentBatch:
    data_idx: Tensor
    dt: Tensor
    agent_name: List[str]
    agent_type: Tensor
    curr_agent_state: Tensor
    agent_hist: Tensor
    agent_hist_extent: Tensor
    agent_hist_len: Tensor
    agent_fut: Tensor
    agent_fut_extent: Tensor
    agent_fut_len: Tensor
    num_neigh: Tensor
    neigh_types: Tensor
    neigh_hist: Tensor
    neigh_hist_len: Tensor
    neigh_fut: Tensor
    neigh_fut_len: Tensor
    robot_fut: Optional[Tensor]
    robot_fut_len: Tensor
    maps: Optional[Tensor]
    maps_resolution: Optional[Tensor]

    def to(self, device) -> None:
        excl_vals = {"data_idx", "agent_name", "agent_type", "neigh_types", "num_neigh"}
        for val in vars(self).keys():
            tensor_val = getattr(self, val)
            if val not in excl_vals and tensor_val is not None:
                tensor_val.to(device)

    def agent_types(self) -> List[AgentType]:
        unique_types: Tensor = torch.unique(self.agent_type)
        return [AgentType(unique_type.item()) for unique_type in unique_types]

    def for_agent_type(self, agent_type: AgentType) -> AgentBatch:
        match_type = self.agent_type == agent_type
        return AgentBatch(
            data_idx=self.data_idx[match_type],
            dt=self.dt[match_type],
            agent_name=[
                name for idx, name in enumerate(self.agent_name) if match_type[idx]
            ],
            agent_type=agent_type.value,
            curr_agent_state=self.curr_agent_state[match_type],
            agent_hist=self.agent_hist[match_type],
            agent_hist_extent=self.agent_hist_extent[match_type],
            agent_hist_len=self.agent_hist_len[match_type],
            agent_fut=self.agent_fut[match_type],
            agent_fut_extent=self.agent_fut_extent[match_type],
            agent_fut_len=self.agent_fut_len[match_type],
            num_neigh=self.num_neigh[match_type],
            neigh_types=self.neigh_types[match_type],
            neigh_hist=self.neigh_hist[match_type],
            neigh_hist_len=self.neigh_hist_len[match_type],
            neigh_fut=self.neigh_fut[match_type],
            neigh_fut_len=self.neigh_fut_len[match_type],
            robot_fut=self.robot_fut[match_type]
            if self.robot_fut is not None
            else None,
            robot_fut_len=self.robot_fut_len[match_type],
            maps=self.maps[match_type] if self.maps is not None else None,
            maps_resolution=self.maps_resolution[match_type]
            if self.maps_resolution is not None
            else None,
        )


SceneBatch = namedtuple("SceneBatch", "")


def map_collate_fn(
    batch_elems: List[AgentBatchElement],
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    if batch_elems[0].map_patch is None:
        return None, None

    patch_data: Tensor = torch.as_tensor(
        np.stack([batch_elem.map_patch.data for batch_elem in batch_elems]),
        dtype=torch.float,
    )
    rot_angles: Tensor = torch.as_tensor(
        [batch_elem.map_patch.rot_angle for batch_elem in batch_elems],
        dtype=torch.float,
    )
    patch_size: int = batch_elems[0].map_patch.crop_size
    assert all(
        batch_elem.map_patch.crop_size == patch_size for batch_elem in batch_elems
    )

    resolution: Tensor = torch.as_tensor(
        [batch_elem.map_patch.resolution for batch_elem in batch_elems],
        dtype=torch.float,
    )

    if (
        torch.count_nonzero(rot_angles) == 0
        and patch_size == patch_data.shape[-1] == patch_data.shape[-2]
    ):
        return patch_data, resolution

    rot_crop_patches: Tensor = center_crop(
        rotate(patch_data, torch.rad2deg(rot_angles)), (patch_size, patch_size)
    )

    return rot_crop_patches, resolution


def agent_collate_fn(
    batch_elems: List[AgentBatchElement], return_dict: bool
) -> Union[AgentBatch, Dict[str, Any]]:
    batch_size: int = len(batch_elems)

    data_index_t: Tensor = torch.zeros((batch_size,), dtype=torch.int)
    dt_t: Tensor = torch.zeros((batch_size,), dtype=torch.float)
    agent_type_t: Tensor = torch.zeros((batch_size,), dtype=torch.int)
    agent_names: List[str] = list()

    curr_agent_state: List[Tensor] = list()

    agent_history: List[Tensor] = list()
    agent_history_extent: List[Tensor] = list()
    agent_history_len: Tensor = torch.zeros((batch_size,), dtype=torch.long)

    agent_future: List[Tensor] = list()
    agent_future_extent: List[Tensor] = list()
    agent_future_len: Tensor = torch.zeros((batch_size,), dtype=torch.long)

    num_neighbors_t: Tensor = torch.as_tensor(
        [elem.num_neighbors for elem in batch_elems], dtype=torch.long
    )
    max_num_neighbors: int = num_neighbors_t.max().item()

    neighbor_types: List[Tensor] = list()
    neighbor_histories: List[Tensor] = list()
    neighbor_futures: List[Tensor] = list()

    # Doing this one up here so that I can use it later in the loop.
    if max_num_neighbors > 0:
        neighbor_history_lens_t: Tensor = pad_sequence(
            [
                torch.as_tensor(elem.neighbor_history_lens_np, dtype=torch.long)
                for elem in batch_elems
            ],
            batch_first=True,
            padding_value=0,
        )
        max_neigh_history_len: int = neighbor_history_lens_t.max().item()

        neighbor_future_lens_t: Tensor = pad_sequence(
            [
                torch.as_tensor(elem.neighbor_future_lens_np, dtype=torch.long)
                for elem in batch_elems
            ],
            batch_first=True,
            padding_value=0,
        )
        max_neigh_future_len: int = neighbor_future_lens_t.max().item()
    else:
        neighbor_history_lens_t: Tensor = torch.full((batch_size, 0), np.nan)
        max_neigh_history_len: int = 0

        neighbor_future_lens_t: Tensor = torch.full((batch_size, 0), np.nan)
        max_neigh_future_len: int = 0

    robot_future: List[Tensor] = list()
    robot_future_len: Tensor = torch.zeros((batch_size,), dtype=torch.long)

    elem: AgentBatchElement
    for idx, elem in enumerate(batch_elems):
        data_index_t[idx] = elem.data_index
        dt_t[idx] = elem.dt
        agent_names.append(elem.agent_name)
        agent_type_t[idx] = elem.agent_type.value

        curr_agent_state.append(
            torch.as_tensor(elem.curr_agent_state_np, dtype=torch.float)
        )

        agent_history.append(
            torch.as_tensor(elem.agent_history_np, dtype=torch.float).flip(-2)
        )
        agent_history_extent.append(
            torch.as_tensor(elem.agent_history_extent_np, dtype=torch.float).flip(-2)
        )
        agent_history_len[idx] = elem.agent_history_len

        agent_future.append(torch.as_tensor(elem.agent_future_np, dtype=torch.float))
        agent_future_extent.append(
            torch.as_tensor(elem.agent_future_extent_np, dtype=torch.float)
        )
        agent_future_len[idx] = elem.agent_future_len

        neighbor_types.append(
            torch.as_tensor(elem.neighbor_types_np, dtype=torch.float)
        )

        if elem.num_neighbors > 0:
            # History
            padded_neighbor_histories = pad_sequence(
                [
                    torch.as_tensor(nh, dtype=torch.float).flip(-2)
                    for nh in elem.neighbor_histories
                ],
                batch_first=True,
                padding_value=np.nan,
            ).flip(-2)
            if padded_neighbor_histories.shape[-2] < max_neigh_history_len:
                to_add = max_neigh_history_len - padded_neighbor_histories.shape[-2]
                padded_neighbor_histories = F.pad(
                    padded_neighbor_histories,
                    pad=(0, 0, to_add, 0),
                    mode="constant",
                    value=np.nan,
                )

            neighbor_histories.append(
                padded_neighbor_histories.reshape(
                    (-1, padded_neighbor_histories.shape[-1])
                )
            )

            # Future
            padded_neighbor_futures = pad_sequence(
                [
                    torch.as_tensor(nh, dtype=torch.float)
                    for nh in elem.neighbor_futures
                ],
                batch_first=True,
                padding_value=np.nan,
            )
            if padded_neighbor_futures.shape[-2] < max_neigh_history_len:
                to_add = max_neigh_future_len - padded_neighbor_futures.shape[-2]
                padded_neighbor_futures = F.pad(
                    padded_neighbor_futures,
                    pad=(0, 0, 0, to_add),
                    mode="constant",
                    value=np.nan,
                )

            neighbor_futures.append(
                padded_neighbor_futures.reshape((-1, padded_neighbor_futures.shape[-1]))
            )
        else:
            # If there's no neighbors, make the state dimension match the
            # agent history state dimension (presumably they'll be the same
            # since they're obtained from the same cached data source).
            neighbor_histories.append(
                torch.full((0, elem.agent_history_np.shape[-1]), np.nan)
            )

            neighbor_futures.append(
                torch.full((0, elem.agent_future_np.shape[-1]), np.nan)
            )

        if elem.robot_future_np is not None:
            robot_future.append(
                torch.as_tensor(elem.robot_future_np, dtype=torch.float)
            )
            robot_future_len[idx] = elem.robot_future_len

    curr_agent_state_t: Tensor = torch.stack(curr_agent_state)

    agent_history_t: Tensor = pad_sequence(
        agent_history, batch_first=True, padding_value=np.nan
    ).flip(-2)
    agent_history_extent_t: Tensor = pad_sequence(
        agent_history_extent, batch_first=True, padding_value=np.nan
    ).flip(-2)

    agent_future_t: Tensor = pad_sequence(
        agent_future, batch_first=True, padding_value=np.nan
    )
    agent_future_extent_t: Tensor = pad_sequence(
        agent_future_extent, batch_first=True, padding_value=np.nan
    )

    if max_num_neighbors > 0:
        neighbor_types_t: Tensor = pad_sequence(
            neighbor_types, batch_first=True, padding_value=-1
        )

        neighbor_histories_t: Tensor = pad_sequence(
            neighbor_histories, batch_first=True, padding_value=np.nan
        ).reshape((batch_size, max_num_neighbors, max_neigh_history_len, -1))

        neighbor_futures_t: Tensor = pad_sequence(
            neighbor_futures, batch_first=True, padding_value=np.nan
        ).reshape((batch_size, max_num_neighbors, max_neigh_future_len, -1))
    else:
        neighbor_types_t: Tensor = torch.full((batch_size, 0), np.nan)

        neighbor_histories_t: Tensor = torch.full(
            (batch_size, 0, max_neigh_history_len, curr_agent_state_t.shape[-1]), np.nan
        )

        neighbor_futures_t: Tensor = torch.full(
            (batch_size, 0, max_neigh_future_len, curr_agent_state_t.shape[-1]), np.nan
        )

    robot_future_t: Optional[Tensor] = (
        pad_sequence(robot_future, batch_first=True, padding_value=np.nan)
        if robot_future
        else None
    )
    map_patches, maps_resolution = map_collate_fn(batch_elems)

    batch = AgentBatch(
        data_idx=data_index_t,
        dt=dt_t,
        agent_name=agent_names,
        agent_type=agent_type_t,
        curr_agent_state=curr_agent_state_t,
        agent_hist=agent_history_t,
        agent_hist_extent=agent_history_extent_t,
        agent_hist_len=agent_history_len,
        agent_fut=agent_future_t,
        agent_fut_extent=agent_future_extent_t,
        agent_fut_len=agent_future_len,
        num_neigh=num_neighbors_t,
        neigh_types=neighbor_types_t,
        neigh_hist=neighbor_histories_t,
        neigh_hist_len=neighbor_history_lens_t,
        neigh_fut=neighbor_futures_t,
        neigh_fut_len=neighbor_future_lens_t,
        robot_fut=robot_future_t,
        robot_fut_len=robot_future_len,
        maps=map_patches,
        maps_resolution=maps_resolution,
    )

    if return_dict:
        return asdict(batch)

    return batch


def scene_collate_fn(batch_elems: List[SceneBatchElement]) -> SceneBatch:
    return SceneBatch(
        nums=default_collate(
            [batch_elem.history_sec_at_most for batch_elem in batch_elems]
        )
    )

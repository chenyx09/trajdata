from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data._utils.collate import default_collate

from avdata.data_structures.agent import AgentType

from .batch_element import AgentBatchElement, SceneBatchElement


@dataclass
class AgentBatch:
    data_idx: Tensor
    dt: Tensor
    agent_type: Tensor
    curr_agent_state: Tensor
    agent_hist: Tensor
    agent_hist_len: Tensor
    agent_fut: Tensor
    agent_fut_len: Tensor
    num_neigh: Tensor
    neigh_types: Tensor
    neigh_hist: Tensor
    neigh_hist_len: Tensor
    robot_fut: Optional[Tensor]
    robot_fut_len: Tensor
    maps: Optional[Tensor]

    def to(self, device) -> None:
        excl_vals = {"data_idx", "agent_type", "neigh_types", "num_neigh"}
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
            agent_type=agent_type.value,
            curr_agent_state=self.curr_agent_state[match_type],
            agent_hist=self.agent_hist[match_type],
            agent_hist_len=self.agent_hist_len[match_type],
            agent_fut=self.agent_fut[match_type],
            agent_fut_len=self.agent_fut_len[match_type],
            num_neigh=self.num_neigh[match_type],
            neigh_types=self.neigh_types[match_type],
            neigh_hist=self.neigh_hist[match_type],
            neigh_hist_len=self.neigh_hist_len[match_type],
            robot_fut=self.robot_fut[match_type]
            if self.robot_fut is not None
            else None,
            robot_fut_len=self.robot_fut_len[match_type],
            maps=self.maps[match_type] if self.maps is not None else None,
        )


SceneBatch = namedtuple("SceneBatch", "")


def agent_collate_fn(batch_elems: List[AgentBatchElement]) -> AgentBatch:
    batch_size: int = len(batch_elems)

    data_index_t: Tensor = torch.zeros((batch_size,), dtype=torch.int)
    dt_t: Tensor = torch.zeros((batch_size,), dtype=torch.float)
    agent_type_t: Tensor = torch.zeros((batch_size,), dtype=torch.int)

    curr_agent_state: List[Tensor] = list()

    agent_history: List[Tensor] = list()
    agent_history_len: Tensor = torch.zeros((batch_size,), dtype=torch.long)

    agent_future: List[Tensor] = list()
    agent_future_len: Tensor = torch.zeros((batch_size,), dtype=torch.long)

    num_neighbors_t: Tensor = torch.zeros((batch_size,), dtype=torch.long)
    neighbor_types: List[Tensor] = list()
    neighbor_histories: List[Tensor] = list()

    # Doing this one up here so that I can use it later in the loop.
    neighbor_history_lens_t: Tensor = pad_sequence(
        [
            torch.as_tensor(elem.neighbor_history_lens_np, dtype=torch.long)
            for elem in batch_elems
        ],
        batch_first=True,
        padding_value=0,
    )
    max_neigh_history_len: int = neighbor_history_lens_t.max().item()

    robot_future: List[Tensor] = list()
    robot_future_len: Tensor = torch.zeros((batch_size,), dtype=torch.long)

    map_info: List[Tensor] = list()

    elem: AgentBatchElement
    for idx, elem in enumerate(batch_elems):
        data_index_t[idx] = elem.data_index
        dt_t[idx] = elem.dt
        agent_type_t[idx] = elem.agent_type.value

        curr_agent_state.append(
            torch.as_tensor(elem.curr_agent_state_np, dtype=torch.float)
        )

        agent_history.append(
            torch.as_tensor(elem.agent_history_np, dtype=torch.float).flip(-2)
        )
        agent_history_len[idx] = elem.agent_history_len

        agent_future.append(torch.as_tensor(elem.agent_future_np, dtype=torch.float))
        agent_future_len[idx] = elem.agent_future_len

        num_neighbors_t[idx] = elem.num_neighbors
        neighbor_types.append(
            torch.as_tensor(elem.neighbor_types_np, dtype=torch.float)
        )

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
            padded_neighbor_histories.reshape((-1, padded_neighbor_histories.shape[-1]))
        )

        if elem.robot_future_np is not None:
            robot_future.append(
                torch.as_tensor(elem.robot_future_np, dtype=torch.float)
            )
            robot_future_len[idx] = elem.robot_future_len

        if elem.map_np is not None:
            map_info.append(torch.as_tensor(elem.map_np, dtype=torch.float))

    curr_agent_state_t: Tensor = torch.stack(curr_agent_state)
    agent_history_t: Tensor = pad_sequence(
        agent_history, batch_first=True, padding_value=np.nan
    ).flip(-2)
    agent_future_t: Tensor = pad_sequence(
        agent_future, batch_first=True, padding_value=np.nan
    )

    neighbor_types_t: Tensor = pad_sequence(
        neighbor_types, batch_first=True, padding_value=-1
    )
    neighbor_histories_t: Tensor = pad_sequence(
        neighbor_histories, batch_first=True, padding_value=np.nan
    ).reshape((batch_size, num_neighbors_t.max(), max_neigh_history_len, -1))

    robot_future_t: Optional[Tensor] = (
        pad_sequence(robot_future, batch_first=True, padding_value=np.nan)
        if robot_future
        else None
    )
    map_info: Optional[Tensor] = torch.stack(map_info) if map_info else None

    return AgentBatch(
        data_idx=data_index_t,
        dt=dt_t,
        agent_type=agent_type_t,
        curr_agent_state=curr_agent_state_t,
        agent_hist=agent_history_t,
        agent_hist_len=agent_history_len,
        agent_fut=agent_future_t,
        agent_fut_len=agent_future_len,
        num_neigh=num_neighbors_t,
        neigh_types=neighbor_types_t,
        neigh_hist=neighbor_histories_t,
        neigh_hist_len=neighbor_history_lens_t,
        robot_fut=robot_future_t,
        robot_fut_len=robot_future_len,
        maps=map_info,
    )


def scene_collate_fn(batch_elems: List[SceneBatchElement]) -> SceneBatch:
    return SceneBatch(
        nums=default_collate(
            [batch_elem.history_sec_at_most for batch_elem in batch_elems]
        )
    )

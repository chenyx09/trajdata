import torch
import numpy as np

from typing import List, Optional
from collections import namedtuple

from torch import Tensor
from torch.utils.data._utils.collate import default_collate
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from .batch_element import AgentBatchElement, SceneBatchElement


AgentBatch = namedtuple("AgentBatch", ["data_idx", "dt", 
                                       "agent_type", "curr_agent_state", 
                                       "agent_hist", "agent_hist_len",
                                       "agent_fut", "agent_fut_len",
                                       "num_neigh", "neigh_types", "neigh_hist", "neigh_hist_len",
                                       "robot_fut", "robot_fut_len",
                                       "maps"])
SceneBatch = namedtuple("SceneBatch", "")


def agent_collate_fn(batch_elems: List[AgentBatchElement]) -> AgentBatch:
    batch_size: int = len(batch_elems)

    data_index_t: Tensor = torch.zeros((batch_size, ), dtype=torch.int)
    dt_t: Tensor = torch.zeros((batch_size, ), dtype=torch.float)
    agent_type_t: Tensor = torch.zeros((batch_size, ), dtype=torch.int)
    
    curr_agent_state: List[Tensor] = list()

    agent_history: List[Tensor] = list()
    agent_history_len: Tensor = torch.zeros((batch_size, ), dtype=torch.int)

    agent_future: List[Tensor] = list()
    agent_future_len: Tensor = torch.zeros((batch_size, ), dtype=torch.int)

    num_neighbors_t: Tensor = torch.zeros((batch_size, ), dtype=torch.int)
    neighbor_types: List[Tensor] = list()
    neighbor_histories: List[Tensor] = list()

    # Doing this one up here so that I can use it later in the loop.
    neighbor_history_lens_t: Tensor = pad_sequence([torch.as_tensor(elem.neighbor_history_lens_np) for elem in batch_elems], batch_first=True)
    max_neigh_history_len: int = neighbor_history_lens_t.max()

    robot_future: List[Tensor] = list()
    robot_future_len: Tensor = torch.zeros((batch_size, ), dtype=torch.int)

    map_info: List[Tensor] = list()

    elem: AgentBatchElement
    for idx, elem in enumerate(batch_elems):
        data_index_t[idx] = elem.data_index
        dt_t[idx] = elem.dt
        agent_type_t[idx] = elem.agent_type.value

        curr_agent_state.append(torch.as_tensor(elem.curr_agent_state_np))

        agent_history.append(torch.as_tensor(elem.agent_history_np))
        agent_history_len[idx] = elem.agent_history_len

        agent_future.append(torch.as_tensor(elem.agent_future_np))
        agent_future_len[idx] = elem.agent_future_len

        num_neighbors_t[idx] = elem.num_neighbors
        neighbor_types.append(torch.as_tensor(elem.neighbor_types_np))
        
        padded_neighbor_histories = pad_sequence([torch.as_tensor(nh) for nh in elem.neighbor_histories], batch_first=True)
        if padded_neighbor_histories.shape[-2] < max_neigh_history_len:
            to_add = max_neigh_history_len - padded_neighbor_histories.shape[-2]
            padded_neighbor_histories = F.pad(padded_neighbor_histories, pad=(0, 0, 0, to_add), mode='constant', value=0)

        neighbor_histories.append(padded_neighbor_histories.reshape((-1, padded_neighbor_histories.shape[-1])))

        if elem.robot_future_np is not None:
            robot_future.append(torch.as_tensor(elem.robot_future_np))
            robot_future_len[idx] = elem.robot_future_len
        
        if elem.map_np is not None:
            map_info.append(torch.as_tensor(elem.map_np))

    curr_agent_state_t: Tensor = torch.stack(curr_agent_state)
    agent_history_t: Tensor = pad_sequence(agent_history, batch_first=True)
    agent_future_t: Tensor = pad_sequence(agent_future, batch_first=True)

    neighbor_types_t: Tensor = pad_sequence(neighbor_types, batch_first=True)
    neighbor_histories_t: Tensor = pad_sequence(neighbor_histories, batch_first=True).reshape((batch_size,
                                                                                               num_neighbors_t.max(),
                                                                                               max_neigh_history_len,
                                                                                               -1))
    
    robot_future_t: Optional[Tensor] = pad_sequence(robot_future, batch_first=True) if robot_future else None
    map_info: Optional[Tensor] = torch.stack(map_info) if map_info else None

    return AgentBatch(data_idx=data_index_t, dt=dt_t, 
                      agent_type=agent_type_t, curr_agent_state=curr_agent_state_t,
                      agent_hist=agent_history_t, agent_hist_len=agent_history_len,
                      agent_fut=agent_future_t, agent_fut_len=agent_future_len,
                      num_neigh=num_neighbors_t, neigh_types=neighbor_types_t, 
                      neigh_hist=neighbor_histories_t, neigh_hist_len=neighbor_history_lens_t,
                      robot_fut=robot_future_t, robot_fut_len=robot_future_len,
                      maps=map_info)


def scene_collate_fn(batch_elems: List[SceneBatchElement]) -> SceneBatch:
    return SceneBatch(nums=default_collate([batch_elem.history_sec_at_most for batch_elem in batch_elems]))
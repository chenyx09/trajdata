import torch
import numpy as np

from typing import List, Optional

from torch import Tensor
from torch.utils.data._utils.collate import default_collate

from .batch_element import AgentBatchElement, SceneBatchElement


class AgentBatch:
    """A batch of agent-centric data.
    """
    def __init__(self, nums: Tensor) -> None:
        self.nums = nums


class SceneBatch:
    """A batch of scene-centric data.
    """
    def __init__(self, nums: Tensor) -> None:
        self.nums = nums


def agent_collate_fn(batch_elems: List[AgentBatchElement]) -> AgentBatch:
    batch_size: int = len(batch_elems)

    data_index_t: Tensor = torch.zeros((batch_size, ), dtype=torch.int)
    dt_t: Tensor = torch.zeros((batch_size, ), dtype=torch.float)
    agent_type_t: Tensor = torch.zeros((batch_size, ), dtype=torch.int)
    
    agent_history: List[Tensor] = list()
    curr_agent_state: List[Tensor] = list()
    agent_future: List[Tensor] = list()

    num_neighbors_t: Tensor = torch.zeros((batch_size, ), dtype=torch.int)
    neighbor_types: List[Tensor] = list()
    neighbor_histories: List[Tensor] = list()
    neighbor_lens: List[Tensor] = list()

    robot_future: List[Tensor] = list()
    map_info: List[Tensor] = list()

    elem: AgentBatchElement
    for idx, elem in enumerate(batch_elems):
        data_index_t[idx] = elem.data_index
        dt_t[idx] = elem.dt
        agent_type_t[idx] = elem.agent_type.value

        agent_history.append(torch.as_tensor(elem.agent_history_np))
        curr_agent_state.append(torch.as_tensor(elem.curr_agent_state_np))
        agent_future.append(torch.as_tensor(elem.agent_future_np))

        num_neighbors_t[idx] = elem.num_neighbors
        neighbor_types.append(torch.as_tensor(elem.neighbor_types_np))
        # TODO(bivanovic): CONTINUE FROM HERE! THIS IS WHERE THE FUN BEGINS
        # neighbor_histories.append(torch.as_tensor(elem.neighbor_histories))
        neighbor_lens.append(torch.as_tensor(elem.neighbor_lens_np))

        if elem.robot_future_np is not None:
            robot_future.append(torch.as_tensor(elem.robot_future_np))
        
        if elem.map_np is not None:
            map_info.append(torch.as_tensor(elem.map_np))

    curr_agent_state_t: Tensor = torch.stack(curr_agent_state)

    return AgentBatch(nums=[7])


def scene_collate_fn(batch_elems: List[SceneBatchElement]) -> SceneBatch:
    return SceneBatch(nums=default_collate([batch_elem.history_sec_at_most for batch_elem in batch_elems]))
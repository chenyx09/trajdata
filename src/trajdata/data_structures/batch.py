from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import Tensor

from trajdata.data_structures.agent import AgentType
from trajdata.utils.arr_utils import PadDirection


def _filter_tensor_or_list(tensor_or_list, filter_mask):
    if isinstance(tensor_or_list, torch.Tensor):
        return tensor_or_list[filter_mask]
    else:
        return type(tensor_or_list)([el for idx, el in enumerate(tensor_or_list) if filter_mask[idx]])


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
    neigh_hist_extents: Tensor
    neigh_hist_len: Tensor
    neigh_fut: Tensor
    neigh_fut_extents: Tensor
    neigh_fut_len: Tensor
    robot_fut: Optional[Tensor]
    robot_fut_len: Optional[Tensor]
    maps: Optional[Tensor]
    maps_resolution: Optional[Tensor]
    rasters_from_world_tf: Optional[Tensor]
    agents_from_world_tf: Tensor
    scene_ids: Optional[List]
    history_pad_dir: PadDirection
    extras: Dict[str, Tensor]

    def to(self, device) -> None:
        excl_vals = {
            "data_idx",
            "agent_name",
            "agent_type",
            "agent_hist_len",
            "agent_fut_len",
            "neigh_hist_len",
            "neigh_fut_len",
            "neigh_types",
            "num_neigh",
            "robot_fut_len",
            "scene_ids",
            "history_pad_dir",
            "extras",
        }
        for val in vars(self).keys():
            tensor_val = getattr(self, val)
            if val not in excl_vals and tensor_val is not None:
                tensor_val: Tensor
                setattr(self, val, tensor_val.to(device, non_blocking=True))

        for key, val in self.extras.items():
            # Allow for custom .to() method for objects that define a __to__ function.
            if hasattr(val, "__to__"):
                self.extras[key] = val.__to__(device, non_blocking=True)
            else:
                self.extras[key] = val.to(device, non_blocking=True)

    def agent_types(self) -> List[AgentType]:
        unique_types: Tensor = torch.unique(self.agent_type)
        return [AgentType(unique_type.item()) for unique_type in unique_types]

    def for_agent_type(self, agent_type: AgentType) -> AgentBatch:
        match_type = self.agent_type == agent_type
        return self.filter_batch(match_type)

    def filter_batch(self, filter_mask: torch.tensor) -> AgentBatch:
        """Build a new batch with elements for which filter_mask[i] == True."""
        return AgentBatch(
            data_idx=self.data_idx[filter_mask],
            dt=self.dt[filter_mask],
            agent_name=[
                name for idx, name in enumerate(self.agent_name) if filter_mask[idx]
            ],
            agent_type=self.agent_type[filter_mask],
            curr_agent_state=self.curr_agent_state[filter_mask],
            agent_hist=self.agent_hist[filter_mask],
            agent_hist_extent=self.agent_hist_extent[filter_mask],
            agent_hist_len=self.agent_hist_len[filter_mask],
            agent_fut=self.agent_fut[filter_mask],
            agent_fut_extent=self.agent_fut_extent[filter_mask],
            agent_fut_len=self.agent_fut_len[filter_mask],
            num_neigh=self.num_neigh[filter_mask],
            neigh_types=self.neigh_types[filter_mask],
            neigh_hist=self.neigh_hist[filter_mask],
            neigh_hist_extents=self.neigh_hist_extents[filter_mask],
            neigh_hist_len=self.neigh_hist_len[filter_mask],
            neigh_fut=self.neigh_fut[filter_mask],
            neigh_fut_extents=self.neigh_fut_extents[filter_mask],
            neigh_fut_len=self.neigh_fut_len[filter_mask],
            robot_fut=self.robot_fut[filter_mask]
            if self.robot_fut is not None
            else None,
            robot_fut_len=self.robot_fut_len[filter_mask]
            if self.robot_fut_len is not None
            else None,
            maps=self.maps[filter_mask] if self.maps is not None else None,
            maps_resolution=self.maps_resolution[filter_mask]
            if self.maps_resolution is not None
            else None,
            rasters_from_world_tf=self.rasters_from_world_tf[filter_mask]
            if self.rasters_from_world_tf is not None
            else None,
            agents_from_world_tf=self.agents_from_world_tf[filter_mask],
            scene_ids=[
                scene_id
                for idx, scene_id in enumerate(self.scene_ids)
                if filter_mask[idx]
            ],
            history_pad_dir=self.history_pad_dir,
            extras={
                key: _filter_tensor_or_list(val, filter_mask) 
                for key, val in self.extras.items()},
        )


@dataclass
class SceneBatch:
    data_idx: Tensor
    dt: Tensor
    num_agents: Tensor
    agent_type: Tensor
    centered_agent_state: Tensor
    agent_hist: Tensor
    agent_hist_extent: Tensor
    agent_hist_len: Tensor
    agent_fut: Tensor
    agent_fut_extent: Tensor
    agent_fut_len: Tensor
    robot_fut: Optional[Tensor]
    robot_fut_len: Optional[Tensor]
    maps: Optional[Tensor]
    maps_resolution: Optional[Tensor]
    rasters_from_world_tf: Optional[Tensor]
    centered_agent_from_world_tf: Tensor
    centered_world_from_agent_tf: Tensor
    scene_ids: Optional[List]
    history_pad_dir: PadDirection
    extras: Dict[str, Tensor]

    def to(self, device) -> None:
        excl_vals = {
            "history_pad_dir",
            "scene_ids",
            "extras",
        }

        for val in vars(self).keys():
            tensor_val = getattr(self, val)
            if val not in excl_vals and tensor_val is not None:
                setattr(self, val, tensor_val.to(device))

        for key, val in self.extras.items():
            # Allow for custom .to() method for objects that define a __to__ function.
            if hasattr(val, "__to__"):
                self.extras[key] = val.__to__(device, non_blocking=True)
            else:
                self.extras[key] = val.to(device, non_blocking=True)

    def agent_types(self) -> List[AgentType]:
        unique_types: Tensor = torch.unique(self.agent_type)
        return [AgentType(unique_type.item()) for unique_type in unique_types if unique_type >= 0]

    def for_agent_type(self, agent_type: AgentType) -> SceneBatch:
        match_type = self.agent_type == agent_type
        return self.filter_batch(match_type)

    def filter_batch(self, filter_mask: torch.tensor) -> SceneBatch:
        """Build a new batch with elements for which filter_mask[i] == True."""        
        return SceneBatch(
            data_idx=self.data_idx[filter_mask],
            dt=self.dt[filter_mask],
            num_agents=self.num_agents[filter_mask],
            agent_type=self.agent_type[filter_mask],
            centered_agent_state=self.centered_agent_state[filter_mask],
            agent_hist=self.agent_hist[filter_mask],
            agent_hist_extent=self.agent_hist_extent[filter_mask],
            agent_hist_len=self.agent_hist_len[filter_mask],
            agent_fut=self.agent_fut[filter_mask],
            agent_fut_extent=self.agent_fut_extent[filter_mask],
            agent_fut_len=self.agent_fut_len[filter_mask],
            robot_fut=self.robot_fut[filter_mask]
            if self.robot_fut is not None
            else None,
            robot_fut_len=self.robot_fut_len[filter_mask]
            if self.robot_fut_len is not None
            else None,
            maps=self.maps[filter_mask] if self.maps is not None else None,
            maps_resolution=self.maps_resolution[filter_mask]
            if self.maps_resolution is not None
            else None,
            rasters_from_world_tf=self.rasters_from_world_tf[filter_mask]
            if self.rasters_from_world_tf is not None
            else None,
            centered_agent_from_world_tf=self.centered_agent_from_world_tf[filter_mask],
            centered_world_from_agent_tf=self.centered_world_from_agent_tf[filter_mask],
            scene_ids=[
                scene_id
                for idx, scene_id in enumerate(self.scene_ids)
                if filter_mask[idx]
            ],
            history_pad_dir=self.history_pad_dir,
            extras={
                key: _filter_tensor_or_list(val, filter_mask) 
                for key, val in self.extras.items()},
        )

    def to_agent_batch(self, agent_inds: torch.Tensor) -> AgentBatch:
        """
        Converts SeceneBatch to AgentBatch for agents defined by `agent_inds`.  

        self.extras will be simply copied over, any custom conversion must be 
        implemented externally.

        TODO(pkarkus) agent names are not yet supported because SceneBatch 
            does not store them
        """

        batch_size = self.agent_hist.shape[0]
        num_agents = self.agent_hist.shape[1]

        if (agent_inds.ndim != 1 or agent_inds.shape[0] != batch_size):
            raise ValueError("Wrong shape for agent_inds, expected [batch_size].")

        if (agent_inds < 0).any() or (agent_inds >= num_agents).any():
            raise ValueError("Invalid agent index")

        batch_inds = torch.arange(batch_size)
        others_mask = torch.ones((batch_size, num_agents), dtype=torch.bool)
        others_mask[batch_inds, agent_inds] = False
        index_agent = lambda x: x[batch_inds, agent_inds]
        index_neighbors = lambda x: x[others_mask].reshape([batch_size, num_agents-1, ] + list(x.shape[2:]))

        return AgentBatch(
            data_idx=self.data_idx,
            dt=self.dt,
            agent_name=["unknown"] * batch_size,
            agent_type=index_agent(self.agent_type),
            curr_agent_state=self.centered_agent_state,  # TODO this is not actually the agent but the `global` coordinate frame
            agent_hist=index_agent(self.agent_hist),
            agent_hist_extent=index_agent(self.agent_hist_extent),
            agent_hist_len=index_agent(self.agent_hist_len),
            agent_fut=index_agent(self.agent_fut),
            agent_fut_extent=index_agent(self.agent_fut_extent),
            agent_fut_len=index_agent(self.agent_fut_len),
            num_neigh=self.num_agents-1,
            neigh_types=index_neighbors(self.agent_type),
            neigh_hist=index_neighbors(self.agent_hist),
            neigh_hist_extents=index_neighbors(self.agent_hist_extent),
            neigh_hist_len=index_neighbors(self.agent_hist_len),
            neigh_fut=index_neighbors(self.agent_fut),
            neigh_fut_extents=index_neighbors(self.agent_fut_extent),
            neigh_fut_len=index_neighbors(self.agent_fut_len),
            robot_fut=self.robot_fut,
            robot_fut_len=self.robot_fut_len,
            maps=self.maps,
            maps_resolution=self.maps_resolution,
            rasters_from_world_tf=self.rasters_from_world_tf,
            agents_from_world_tf=self.centered_agent_from_world_tf,
            scene_ids=self.scene_ids,
            history_pad_dir=self.history_pad_dir,
            extras=self.extras,
        )          


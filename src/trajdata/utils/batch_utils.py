from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from pathlib import Path

from trajdata.data_structures import (
    AgentBatch,
    AgentBatchElement,
    AgentType,
    SceneBatch,
    SceneBatchElement,
    SceneTimeAgent,
)
from trajdata.data_structures.collation import agent_collate_fn, batch_rotate_raster_maps_for_agents_in_scene
from trajdata.maps import RasterizedMapPatch
from trajdata.utils.map_utils import load_map_patch
from trajdata.utils.arr_utils import batch_nd_transform_xyvvaahh_pt, batch_select, PadDirection
from trajdata.caching.df_cache import DataFrameCache


def convert_to_agent_batch(
    scene_batch_element: SceneBatchElement,
    only_types: Optional[List[AgentType]] = None,
    no_types: Optional[List[AgentType]] = None,
    agent_interaction_distances: Dict[Tuple[AgentType, AgentType], float] = defaultdict(
        lambda: np.inf
    ),
    incl_map: bool = False,
    map_params: Optional[Dict[str, Any]] = None,
    max_neighbor_num: Optional[int] = None,
    standardize_data: bool = True,
    standardize_derivatives: bool = False,
    pad_format: str = "outside",
) -> AgentBatch:
    """
    Converts a SceneBatchElement into a AgentBatch consisting of
    AgentBatchElements for all agents present at the given scene at the given
    time step.

    Args:
        scene_batch_element (SceneBatchElement): element to process
        only_types (Optional[List[AgentType]], optional): AgentsTypes to consider. Defaults to None.
        no_types (Optional[List[AgentType]], optional): AgentTypes to ignore. Defaults to None.
        agent_interaction_distances (_type_, optional): Distance threshold for interaction. Defaults to defaultdict(lambda: np.inf).
        incl_map (bool, optional): Whether to include map info. Defaults to False.
        map_params (Optional[Dict[str, Any]], optional): Map params. Defaults to None.
        max_neighbor_num (Optional[int], optional): Max number of neighbors to allow. Defaults to None.
        standardize_data (bool): Whether to return data relative to current agent state. Defaults to True.
        standardize_derivatives: Whether to transform relative velocities and accelerations as well. Defaults to False.
        pad_format (str, optional): Pad format when collating agent trajectories. Defaults to "outside".

    Returns:
        AgentBatch: batch of AgentBatchElements corresponding to all agents in the SceneBatchElement
    """
    data_idx = scene_batch_element.data_index
    cache = scene_batch_element.cache
    scene = cache.scene
    dt = scene_batch_element.dt
    ts = scene_batch_element.scene_ts

    batch_elems: List[AgentBatchElement] = []
    for j, agent_name in enumerate(scene_batch_element.agent_names):
        history_sec = dt * (scene_batch_element.agent_histories[j].shape[0] - 1)
        future_sec = dt * (scene_batch_element.agent_futures[j].shape[0])
        cache.reset_transforms()
        scene_time_agent: SceneTimeAgent = SceneTimeAgent.from_cache(
            scene,
            ts,
            agent_name,
            cache,
            only_types=only_types,
            no_types=no_types,
        )

        batch_elems.append(
            AgentBatchElement(
                cache=cache,
                data_index=data_idx,
                scene_time_agent=scene_time_agent,
                history_sec=(history_sec, history_sec),
                future_sec=(future_sec, future_sec),
                agent_interaction_distances=agent_interaction_distances,
                incl_raster_map=incl_map,
                raster_map_params=map_params,
                standardize_data=standardize_data,
                standardize_derivatives=standardize_derivatives,
                max_neighbor_num=max_neighbor_num,
            )
        )

    return agent_collate_fn(batch_elems, return_dict=False, pad_format=pad_format)


def get_agents_map_patch(
    cache_path: Path, 
    map_name: str,
    patch_params: Dict[str, int], 
    agent_world_states_xyh: Union[np.ndarray, torch.Tensor], 
    allow_nan: float = False,
) -> List[RasterizedMapPatch]:

    if isinstance(agent_world_states_xyh, torch.Tensor):
        agent_world_states_xyh = agent_world_states_xyh.cpu().numpy()
    assert agent_world_states_xyh.ndim == 2
    assert agent_world_states_xyh.shape[-1] == 3
    
    desired_patch_size: int = patch_params["map_size_px"]
    resolution: float = patch_params["px_per_m"]
    offset_xy: Tuple[float, float] = patch_params.get("offset_frac_xy", (0.0, 0.0))
    return_rgb: bool = patch_params.get("return_rgb", True)
    no_map_fill_val: float = patch_params.get("no_map_fill_value", 0.0)

    env_name, location_name = map_name.split(':')  # assumes map_name format nusc_mini:boston-seaport

    map_patches = list()

    (
        maps_path,
        _,
        _,
        raster_map_path,
        raster_metadata_path,
    ) = DataFrameCache.get_map_paths(
        cache_path, env_name, location_name, resolution
    )

    for i in range(agent_world_states_xyh.shape[0]):
        patch_data, raster_from_world_tf, has_data = load_map_patch(
            raster_map_path,
            raster_metadata_path,
            agent_world_states_xyh[i, 0],
            agent_world_states_xyh[i, 1],
            desired_patch_size,
            resolution,
            offset_xy,
            agent_world_states_xyh[i, 2],
            return_rgb,
            rot_pad_factor=np.sqrt(2),
            no_map_val=no_map_fill_val,
        )
        map_patches.append(
            RasterizedMapPatch(
                data=patch_data,
                rot_angle=agent_world_states_xyh[i, 2],
                crop_size=desired_patch_size,
                resolution=resolution,
                raster_from_world_tf=raster_from_world_tf,
                has_data=has_data,
            )
        )

    return map_patches


def get_raster_maps_for_scene_batch(batch: SceneBatch, cache_path: Path, raster_map_params: Dict):

    # Get current states
    if batch.history_pad_dir == PadDirection.AFTER:
        agent_states = batch_select(batch.agent_hist, index=batch.agent_hist_len-1, batch_dims=2)  # b, N, t, 8           
    else:
        agent_states = batch.agent_hist[:, :, -1]

    agent_world_states_xyvvaahh = batch_nd_transform_xyvvaahh_pt(
        agent_states.type_as(batch.centered_world_from_agent_tf), 
        batch.centered_world_from_agent_tf
    ).type_as(batch.agent_hist)

    assert agent_world_states_xyvvaahh.shape[-1] == 8
    agent_world_states_xyh = torch.concat((
        agent_world_states_xyvvaahh[..., :2], 
        torch.atan2(agent_world_states_xyvvaahh[..., 6:7], agent_world_states_xyvvaahh[..., 7:8])), dim=-1)

    maps: List[torch.Tensor] = []
    maps_resolution: List[torch.Tensor] = []
    raster_from_world_tf: List[torch.Tensor] = []

    # Collect map patches for all elements and agents into a flat list
    num_agents: List[int] = []
    map_patches: List[RasterizedMapPatch] = []

    for b_i in range(agent_world_states_xyh.shape[0]):
        num_agents.append(batch.num_agents[b_i])
        map_patches += get_agents_map_patch(
            cache_path, batch.map_names[b_i], raster_map_params, agent_world_states_xyh[b_i, :batch.num_agents[b_i]])

    # Batch transform map patches and pad
    (
        maps, 
        maps_resolution, 
        raster_from_world_tf
    ) = batch_rotate_raster_maps_for_agents_in_scene(
        map_patches, num_agents, agent_world_states_xyh.shape[1], pad_value=np.nan,
    )

    return maps, maps_resolution, raster_from_world_tf

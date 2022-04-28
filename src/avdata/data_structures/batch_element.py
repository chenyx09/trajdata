from collections import defaultdict
from math import sqrt
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from avdata.caching import SceneCache
from avdata.data_structures.agent import AgentMetadata, AgentType
from avdata.data_structures.map_patch import MapPatch
from avdata.data_structures.scene import SceneTime, SceneTimeAgent


class AgentBatchElement:
    """A single element of an agent-centric batch."""

    # @profile
    def __init__(
        self,
        cache: SceneCache,
        data_index: int,
        scene_time_agent: SceneTimeAgent,
        history_sec: Tuple[Optional[float], Optional[float]],
        future_sec: Tuple[Optional[float], Optional[float]],
        agent_interaction_distances: Dict[
            Tuple[AgentType, AgentType], float
        ] = defaultdict(lambda: np.inf),
        incl_robot_future: bool = False,
        incl_map: bool = False,
        map_params: Optional[Dict[str, int]] = None,
        standardize_data: bool = False,
    ) -> None:
        self.cache: SceneCache = cache
        self.data_index: int = data_index
        self.dt: float = scene_time_agent.metadata.dt
        self.scene_ts: int = scene_time_agent.ts

        agent_info: AgentMetadata = scene_time_agent.agent
        self.agent_name: str = agent_info.name
        self.agent_type: AgentType = agent_info.type

        self.curr_agent_state_np: np.ndarray = cache.get_state(
            agent_info.name, self.scene_ts
        )

        self.standardize_data = standardize_data
        if self.standardize_data:
            agent_pos: np.ndarray = self.curr_agent_state_np[:2]
            agent_heading: float = self.curr_agent_state_np[-1]

            cos_agent, sin_agent = np.cos(agent_heading), np.sin(agent_heading)
            world_from_agent_tf: np.ndarray = np.array(
                [
                    [cos_agent, -sin_agent, agent_pos[0]],
                    [sin_agent, cos_agent, agent_pos[1]],
                    [0.0, 0.0, 1.0],
                ]
            )
            self.agent_from_world_tf: np.ndarray = np.linalg.inv(world_from_agent_tf)

            cache.transform_data(
                shift_mean_to=self.curr_agent_state_np,
                rotate_by=agent_heading,
                sincos_heading=True,
            )
        else:
            self.agent_from_world_tf: np.ndarray = np.eye(3)

        ### AGENT-SPECIFIC DATA ###
        self.agent_history_np, self.agent_history_extent_np = self.get_agent_history(
            agent_info, history_sec
        )
        self.agent_history_len: int = self.agent_history_np.shape[0]

        self.agent_future_np, self.agent_future_extent_np = self.get_agent_future(
            agent_info, future_sec
        )
        self.agent_future_len: int = self.agent_future_np.shape[0]

        ### NEIGHBOR-SPECIFIC DATA ###
        def distance_limit(agent_types: np.ndarray, target_type: int) -> np.ndarray:
            return np.array(
                [
                    agent_interaction_distances[(agent_type, target_type)]
                    for agent_type in agent_types
                ]
            )

        (
            self.num_neighbors,
            self.neighbor_types_np,
            self.neighbor_histories,
            self.neighbor_history_extents,
            self.neighbor_history_lens_np,
        ) = self.get_neighbor_history(
            scene_time_agent, agent_info, history_sec, distance_limit
        )

        (
            _,
            _,
            self.neighbor_futures,
            self.neighbor_future_extents,
            self.neighbor_future_lens_np,
        ) = self.get_neighbor_future(
            scene_time_agent, agent_info, future_sec, distance_limit
        )

        ### ROBOT DATA ###
        self.robot_future_np: Optional[np.ndarray] = None
        if incl_robot_future:
            self.robot_future_np: np.ndarray = self.get_robot_current_and_future(
                scene_time_agent.robot, future_sec
            )
            
            # -1 because this is meant to hold the number of future steps
            # (whereas the above returns the current + future, yielding
            # one more timestep).
            self.robot_future_len: int = self.robot_future_np.shape[0] - 1

        ### MAP ###
        self.map_patch: Optional[MapPatch] = None
        if incl_map:
            self.map_patch = self.get_agent_map_patch(map_params)

    def get_agent_history(
        self,
        agent_info: AgentMetadata,
        history_sec: Tuple[Optional[float], Optional[float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        agent_history_np, agent_extent_history_np = self.cache.get_agent_history(
            agent_info, self.scene_ts, history_sec
        )
        return agent_history_np, agent_extent_history_np

    def get_agent_future(
        self,
        agent_info: AgentMetadata,
        future_sec: Tuple[Optional[float], Optional[float]],
    ) -> np.ndarray:
        agent_future_np, agent_extent_future_np = self.cache.get_agent_future(
            agent_info, self.scene_ts, future_sec
        )
        return agent_future_np, agent_extent_future_np

    # @profile
    def get_neighbor_history(
        self,
        scene_time: SceneTimeAgent,
        agent_info: AgentMetadata,
        history_sec: Tuple[Optional[float], Optional[float]],
        distance_limit: Callable[[np.ndarray, int], np.ndarray],
    ) -> Tuple[int, np.ndarray, List[np.ndarray], List[np.ndarray], np.ndarray]:
        # The indices of the returned ndarray match the scene_time agents list (including the index of the central agent,
        # which would have a distance of 0 to itself).
        agent_distances: np.ndarray = scene_time.get_agent_distances_to(agent_info)
        agent_idx: int = scene_time.agents.index(agent_info)

        neighbor_types: np.ndarray = np.array([a.type.value for a in scene_time.agents])
        nearby_mask: np.ndarray = agent_distances <= distance_limit(
            neighbor_types, agent_info.type
        )
        nearby_mask[agent_idx] = False

        nearby_agents: List[AgentMetadata] = [
            agent for (idx, agent) in enumerate(scene_time.agents) if nearby_mask[idx]
        ]
        neighbor_types_np: np.ndarray = neighbor_types[nearby_mask]

        num_neighbors: int = len(nearby_agents)
        (
            neighbor_histories,
            neighbor_history_extents,
            neighbor_history_lens_np,
        ) = self.cache.get_agents_history(self.scene_ts, nearby_agents, history_sec)

        return (
            num_neighbors,
            neighbor_types_np,
            neighbor_histories,
            neighbor_history_extents,
            neighbor_history_lens_np,
        )

    # @profile
    def get_neighbor_future(
        self,
        scene_time: SceneTimeAgent,
        agent_info: AgentMetadata,
        future_sec: Tuple[Optional[float], Optional[float]],
        distance_limit: Callable[[np.ndarray, int], np.ndarray],
    ) -> Tuple[int, np.ndarray, List[np.ndarray], List[np.ndarray], np.ndarray]:
        scene_ts: int = self.scene_ts

        # The indices of the returned ndarray match the scene_time agents list (including the index of the central agent,
        # which would have a distance of 0 to itself).
        agent_distances: np.ndarray = scene_time.get_agent_distances_to(agent_info)
        agent_idx: int = scene_time.agents.index(agent_info)

        neighbor_types: np.ndarray = np.array([a.type.value for a in scene_time.agents])
        nearby_mask: np.ndarray = agent_distances <= distance_limit(
            neighbor_types, agent_info.type
        )
        nearby_mask[agent_idx] = False

        nearby_agents: List[AgentMetadata] = [
            agent for (idx, agent) in enumerate(scene_time.agents) if nearby_mask[idx]
        ]
        neighbor_types_np: np.ndarray = neighbor_types[nearby_mask]

        num_neighbors: int = len(nearby_agents)
        (
            neighbor_futures,
            neighbor_future_extents,
            neighbor_future_lens_np,
        ) = self.cache.get_agents_future(scene_ts, nearby_agents, future_sec)

        return (
            num_neighbors,
            neighbor_types_np,
            neighbor_futures,
            neighbor_future_extents,
            neighbor_future_lens_np,
        )

    def get_robot_current_and_future(
        self,
        robot_info: AgentMetadata,
        future_sec: Tuple[Optional[float], Optional[float]],
    ) -> np.ndarray:
        # self.scene_ts - 1 because we want to get the current timestep (scene_ts) too
        # and get_agent_future(...) gets data starting from the timestep AFTER the
        # given one.
        (
            robot_curr_and_fut_np,
            robot_curr_and_fut_extents_np,
        ) = self.cache.get_agent_future(robot_info, self.scene_ts - 1, future_sec)
        return robot_curr_and_fut_np

    def get_agent_map_patch(self, patch_params: Dict[str, int]) -> MapPatch:
        world_x, world_y = self.curr_agent_state_np[:2]
        desired_patch_size: int = patch_params["map_size_px"]
        resolution: int = patch_params["px_per_m"]
        offset_xy: Tuple[float, float] = patch_params.get("offset_frac_xy", (0.0, 0.0))
        return_rgb: bool = patch_params.get("return_rgb", True)

        if self.standardize_data:
            heading = self.curr_agent_state_np[-1]
            patch_data, raster_from_world_tf = self.cache.load_map_patch(
                world_x,
                world_y,
                desired_patch_size,
                resolution,
                offset_xy,
                heading,
                return_rgb,
                rot_pad_factor=sqrt(2),
            )
        else:
            heading = 0.0
            patch_data, raster_from_world_tf = self.cache.load_map_patch(
                world_x,
                world_y,
                desired_patch_size,
                resolution,
                offset_xy,
                heading,
                return_rgb,
            )

        return MapPatch(
            data=patch_data,
            rot_angle=heading,
            crop_size=desired_patch_size,
            resolution=resolution,
            raster_from_world_tf=raster_from_world_tf,
        )


class SceneBatchElement:
    """A single batch element."""

    def __init__(
        self,
        scene_time: SceneTime,
        history_sec_at_most: float,
        future_sec_at_most: float,
    ) -> None:
        self.history_sec_at_most = history_sec_at_most

import contextlib
import sqlite3
from collections import defaultdict
from math import floor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from avdata.data_structures.agent import Agent, AgentMetadata, AgentType
from avdata.data_structures.scene import SceneTime, SceneTimeAgent


class AgentBatchElement:
    """A single element of an agent-centric batch."""

    # @profile
    def __init__(
        self,
        scene_cache_dir: Path,
        data_index: int,
        scene_time_agent: SceneTimeAgent,
        history_sec: Tuple[Optional[float], Optional[float]],
        future_sec: Tuple[Optional[float], Optional[float]],
        agent_interaction_distances: Dict[
            Tuple[AgentType, AgentType], float
        ] = defaultdict(lambda: np.inf),
        incl_robot_future: bool = False,
        incl_map: bool = False,
    ) -> None:
        self.scene_cache_dir: Path = scene_cache_dir
        self.data_index: int = data_index
        self.dt: float = scene_time_agent.metadata.dt
        self.scene_ts: int = scene_time_agent.ts

        agent: Agent = scene_time_agent.agent
        self.agent_type: AgentType = agent.type

        ### AGENT-SPECIFIC DATA ###
        self.curr_agent_state_np, self.agent_history_np = self.get_agent_history(
            agent, history_sec
        )
        self.agent_history_len: int = self.agent_history_np.shape[0]

        self.agent_future_np: np.ndarray = self.get_agent_future(agent, future_sec)
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
            self.neighbor_history_lens_np,
        ) = self.get_neighbor_history(
            scene_time_agent, agent, history_sec, distance_limit
        )

        ### ROBOT DATA ###
        self.robot_future_np: Optional[np.ndarray] = None
        if incl_robot_future:
            self.robot_future_np: np.ndarray = self.get_robot_future(
                scene_time_agent.robot, future_sec
            )
            self.robot_future_len: int = self.robot_future_np.shape[0]

        ### MAP ###
        self.map_np: Optional[np.ndarray] = None
        if incl_map:
            self.map_np = self.get_map(scene_time_agent, agent)

    def get_agent_history(
        self, agent: Agent, history_sec: Tuple[Optional[float], Optional[float]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        dt: float = self.dt
        scene_ts: int = self.scene_ts

        # We don't have to check the mins here because our data_index filtering in dataset.py already
        # took care of it.
        if history_sec[1] is not None:
            max_history: int = floor(history_sec[1] / dt)
            agent_history_df: pd.DataFrame = agent.data.loc[
                max(scene_ts - max_history, agent.metadata.first_timestep) : scene_ts
            ].copy()
        else:
            agent_history_df: pd.DataFrame = agent.data.loc[:scene_ts].copy()

        curr_agent_pos_np: np.ndarray = agent_history_df.loc[scene_ts].values
        agent_history_df -= curr_agent_pos_np
        agent_history_df["sin_heading"] = np.sin(agent_history_df["heading"])
        agent_history_df["cos_heading"] = np.cos(agent_history_df["heading"])
        del agent_history_df["heading"]

        return curr_agent_pos_np, agent_history_df.values

    def get_agent_future(
        self, agent: Agent, future_sec: Tuple[Optional[float], Optional[float]]
    ) -> np.ndarray:
        dt: float = self.dt
        scene_ts: int = self.scene_ts

        # We don't have to check the mins here because our data_index filtering in dataset.py already
        # took care of it.
        if future_sec[1] is not None:
            max_future = floor(future_sec[1] / dt)
            agent_future_df = agent.data.loc[
                scene_ts + 1 : min(scene_ts + max_future, agent.metadata.last_timestep),
                ["x", "y"],
            ]
        else:
            agent_future_df = agent.data.loc[scene_ts + 1 :, ["x", "y"]]

        return agent_future_df.values

    # @profile
    def get_neighbor_history(
        self,
        scene_time: SceneTimeAgent,
        agent: Agent,
        history_sec: Tuple[Optional[float], Optional[float]],
        distance_limit: Callable[[np.ndarray, int], np.ndarray],
    ) -> Tuple[int, np.ndarray, List[np.ndarray], np.ndarray]:
        dt: float = self.dt
        scene_ts: int = self.scene_ts

        # The indices of the returned ndarray match the scene_time agents list (including the index of the central agent,
        # which would have a distance of 0 to itself).
        agent_distances: np.ndarray = scene_time.get_agent_distances_to(agent)
        agent_idx: int = scene_time.agents.index(agent.metadata)

        neighbor_types: np.ndarray = np.array([a.type.value for a in scene_time.agents])
        nearby_mask: np.ndarray = agent_distances <= distance_limit(
            neighbor_types, agent.type
        )
        nearby_mask[agent_idx] = False

        nearby_agents: List[AgentMetadata] = [
            agent for (idx, agent) in enumerate(scene_time.agents) if nearby_mask[idx]
        ]
        neighbor_types_np: np.ndarray = neighbor_types[nearby_mask]

        if history_sec[1] is not None:
            max_history: int = floor(history_sec[1] / dt)
        else:
            max_history: int = scene_ts

        num_neighbors: int = len(nearby_agents)
        agent_indices: Dict[str, int] = {
            a.name: idx for idx, a in enumerate(nearby_agents)
        }
        neighbor_history_lens_np: np.ndarray = np.zeros((num_neighbors,), dtype=int)

        with contextlib.closing(
            sqlite3.connect(self.scene_cache_dir / "agent_data.db")
        ) as conn:
            all_agents_df = pd.read_sql_query(
                "SELECT * FROM agent_data WHERE scene_ts BETWEEN ? AND ?",
                conn,
                params=(scene_ts - max_history, scene_ts),
                index_col=["agent_id", "scene_ts"],
            )

        all_agents_df -= self.curr_agent_state_np
        all_agents_df["sin_heading"] = np.sin(all_agents_df["heading"])
        all_agents_df["cos_heading"] = np.cos(all_agents_df["heading"])
        del all_agents_df["heading"]

        all_agents_grouped = all_agents_df.groupby(level=0, sort=False)

        neighbor_histories: List[np.ndarray] = [None for _ in range(num_neighbors)]
        for group_name, neighbor_history_df in all_agents_grouped:
            if group_name in agent_indices:
                idx = agent_indices[group_name]
                neighbor_history_lens_np[idx] = len(
                    neighbor_history_df
                )
                neighbor_histories[idx] = neighbor_history_df.values

        return (
            num_neighbors,
            neighbor_types_np,
            neighbor_histories,
            neighbor_history_lens_np,
        )

    def get_robot_future(
        self, robot: Agent, future_sec: Tuple[Optional[float], Optional[float]]
    ) -> np.ndarray:
        dt: float = self.dt
        scene_ts: int = self.scene_ts

        if future_sec[1] is not None:
            max_future = floor(future_sec[1] / dt)
            robot_future_df = robot.data.loc[
                scene_ts + 1 : min(scene_ts + max_future, robot.metadata.last_timestep)
            ].copy()
        else:
            robot_future_df = robot.data.loc[scene_ts + 1 :].copy()

        robot_future_df -= self.curr_agent_state_np
        robot_future_df["sin_heading"] = np.sin(robot_future_df["heading"])
        robot_future_df["cos_heading"] = np.cos(robot_future_df["heading"])
        del robot_future_df["heading"]

        return robot_future_df.values

    def get_map(self, scene_time: SceneTime, agent: Agent):
        pass


class SceneBatchElement:
    """A single batch element."""

    def __init__(
        self,
        scene_time: SceneTime,
        history_sec_at_most: float,
        future_sec_at_most: float,
    ) -> None:
        self.history_sec_at_most = history_sec_at_most

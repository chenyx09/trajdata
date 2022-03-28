from collections import defaultdict
from math import floor
from typing import Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd

from avdata.caching import SceneCache
from avdata.data_structures.agent import Agent, AgentMetadata, AgentType
from avdata.data_structures.scene import SceneTime, SceneTimeAgent


class AgentBatchElement:
    """A single element of an agent-centric batch."""

    # @profile
    def __init__(
        self,
        cache: Type[SceneCache],
        data_index: int,
        scene_time_agent: SceneTimeAgent,
        history_sec: Tuple[Optional[float], Optional[float]],
        future_sec: Tuple[Optional[float], Optional[float]],
        agent_interaction_distances: Dict[
            Tuple[AgentType, AgentType], float
        ] = defaultdict(lambda: np.inf),
        incl_robot_future: bool = False,
        incl_map: bool = False,
        standardize_rotation: bool = False,
    ) -> None:
        self.cache: Type[SceneCache] = cache
        self.data_index: int = data_index
        self.dt: float = scene_time_agent.metadata.dt
        self.scene_ts: int = scene_time_agent.ts

        agent: Agent = scene_time_agent.agent
        self.agent_type: AgentType = agent.type

        self.standardize_rotation = standardize_rotation
        if self.standardize_rotation:
            agent_heading: float = agent.data.at[self.scene_ts, "heading"]
            self.rot_matrix: np.ndarray = np.array(
                [
                    [np.cos(agent_heading), -np.sin(agent_heading)],
                    [np.sin(agent_heading), np.cos(agent_heading)],
                ]
            )

        ### AGENT-SPECIFIC DATA ###
        self.curr_agent_state_np, self.agent_history_np = self.get_agent_history(
            agent, history_sec
        )
        self.agent_history_len: int = self.agent_history_np.shape[0]

        self.agent_future_np, self.agent_future_st_np = self.get_agent_future(
            agent, future_sec
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

        # self.plot()

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

        curr_agent_pos_np: np.ndarray = agent_history_df.loc[scene_ts].to_numpy()
        agent_history_df -= curr_agent_pos_np
        agent_history_df["sin_heading"] = np.sin(agent_history_df["heading"])
        agent_history_df["cos_heading"] = np.cos(agent_history_df["heading"])
        del agent_history_df["heading"]

        if self.standardize_rotation:
            agent_history_df.loc[:, ["x", "y"]] = (
                agent_history_df.loc[:, ["x", "y"]].to_numpy() @ self.rot_matrix
            )
            agent_history_df.loc[:, ["vx", "vy"]] = (
                agent_history_df.loc[:, ["vx", "vy"]].to_numpy() @ self.rot_matrix
            )
            agent_history_df.loc[:, ["ax", "ay"]] = (
                agent_history_df.loc[:, ["ax", "ay"]].to_numpy() @ self.rot_matrix
            )

        return curr_agent_pos_np, agent_history_df.to_numpy()

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

        if self.standardize_rotation:
            agent_future_st_np = (
                agent_future_df.to_numpy() - self.curr_agent_state_np[:2]
            ) @ self.rot_matrix
        else:
            agent_future_st_np = (
                agent_future_df.to_numpy() - self.curr_agent_state_np[:2]
            )

        return agent_future_df.to_numpy(), agent_future_st_np

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

        all_agents_df = self.cache.load_data_between_times(
            scene_ts - max_history, scene_ts, scene_time.metadata
        )
        all_agents_df -= self.curr_agent_state_np
        all_agents_df["sin_heading"] = np.sin(all_agents_df["heading"])
        all_agents_df["cos_heading"] = np.cos(all_agents_df["heading"])
        del all_agents_df["heading"]

        if self.standardize_rotation:
            all_agents_df.loc[:, ["x", "y"]] = (
                all_agents_df.loc[:, ["x", "y"]].to_numpy() @ self.rot_matrix
            )
            all_agents_df.loc[:, ["vx", "vy"]] = (
                all_agents_df.loc[:, ["vx", "vy"]].to_numpy() @ self.rot_matrix
            )
            all_agents_df.loc[:, ["ax", "ay"]] = (
                all_agents_df.loc[:, ["ax", "ay"]].to_numpy() @ self.rot_matrix
            )

        all_agents_grouped = all_agents_df.groupby(level=0, sort=False)

        neighbor_histories: List[np.ndarray] = [None for _ in range(num_neighbors)]
        for group_name, neighbor_history_df in all_agents_grouped:
            if group_name in agent_indices:
                idx = agent_indices[group_name]
                neighbor_history_lens_np[idx] = len(neighbor_history_df)
                neighbor_histories[idx] = neighbor_history_df.to_numpy()

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
                scene_ts : min(scene_ts + max_future, robot.metadata.last_timestep)
            ].copy()
        else:
            robot_future_df = robot.data.loc[scene_ts:].copy()

        robot_future_df -= self.curr_agent_state_np
        robot_future_df["sin_heading"] = np.sin(robot_future_df["heading"])
        robot_future_df["cos_heading"] = np.cos(robot_future_df["heading"])
        del robot_future_df["heading"]

        if self.standardize_rotation:
            robot_future_df.loc[:, ["x", "y"]] = (
                robot_future_df.loc[:, ["x", "y"]].to_numpy() @ self.rot_matrix
            )
            robot_future_df.loc[:, ["vx", "vy"]] = (
                robot_future_df.loc[:, ["vx", "vy"]].to_numpy() @ self.rot_matrix
            )
            robot_future_df.loc[:, ["ax", "ay"]] = (
                robot_future_df.loc[:, ["ax", "ay"]].to_numpy() @ self.rot_matrix
            )

        return robot_future_df.to_numpy()

    def get_map(self, scene_time: SceneTimeAgent, agent: Agent):
        pass

    def plot(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        ax.plot(
            self.agent_history_np[:, 0],
            self.agent_history_np[:, 1],
            label="Agent History",
            c="blue",
        )
        ax.scatter(0, 0, s=100, c="r", label="Agent Current")
        ax.arrow(
            0,
            0,
            dx=self.agent_history_np[-1, -1] / 10,
            dy=self.agent_history_np[-1, -2] / 10,
        )
        ax.plot(
            self.agent_future_st_np[:, 0],
            self.agent_future_st_np[:, 1],
            label="Agent Future",
            c="orange",
        )

        if self.standardize_rotation:
            ax.arrow(
                0,
                0,
                dx=np.cos(self.curr_agent_state_np[-1]) / 10,
                dy=np.sin(self.curr_agent_state_np[-1]) / 10,
                alpha=0.5,
            )

            rotated_check = self.agent_future_st_np[:, :2] @ self.rot_matrix.T
            ax.plot(rotated_check[:, 0], rotated_check[:, 1], c="orange", alpha=0.5)

        ax.scatter(np.nan, np.nan, s=100, c="k", label="Other Current")
        for neigh_hist in self.neighbor_histories:
            ax.scatter(neigh_hist[-1, 0], neigh_hist[-1, 1], s=100, c="k")
            ax.plot(neigh_hist[:, 0], neigh_hist[:, 1], c="k")

        if self.robot_future_np is not None:
            ax.scatter(
                self.robot_future_np[0, 0],
                self.robot_future_np[0, 1],
                s=100,
                c="green",
                label="Ego Current",
            )
            ax.plot(
                self.robot_future_np[1:, 0],
                self.robot_future_np[1:, 1],
                label="Ego Future",
                c="green",
            )

        ax.legend(loc="best", frameon=True)
        ax.axis("equal")
        plt.show()


class SceneBatchElement:
    """A single batch element."""

    def __init__(
        self,
        scene_time: SceneTime,
        history_sec_at_most: float,
        future_sec_at_most: float,
    ) -> None:
        self.history_sec_at_most = history_sec_at_most

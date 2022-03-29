from collections import defaultdict
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
        standardize_data: bool = False,
    ) -> None:
        self.cache: Type[SceneCache] = cache
        self.data_index: int = data_index
        self.dt: float = scene_time_agent.metadata.dt
        self.scene_ts: int = scene_time_agent.ts

        agent_info: AgentMetadata = scene_time_agent.agent
        self.agent_type: AgentType = agent_info.type

        self.curr_agent_state_np: np.ndarray = cache.get_state(
            agent_info.name, self.scene_ts
        )

        self.standardize_data = standardize_data
        if self.standardize_data:
            agent_heading: float = cache.get_value(
                agent_info.name, self.scene_ts, "heading"
            )
            cache.transform_data(
                shift_mean_to=self.curr_agent_state_np,
                rotate_by=agent_heading,
                sincos_heading=True,
            )

        ### AGENT-SPECIFIC DATA ###
        self.agent_history_np: np.ndarray = self.get_agent_history(
            agent_info, history_sec
        )
        self.agent_history_len: int = self.agent_history_np.shape[0]

        self.agent_future_np: np.ndarray = self.get_agent_future(agent_info, future_sec)
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
            scene_time_agent, agent_info, history_sec, distance_limit
        )

        ### ROBOT DATA ###
        self.robot_future_np: Optional[np.ndarray] = None
        if incl_robot_future:
            self.robot_future_np: np.ndarray = self.get_robot_current_and_future(
                scene_time_agent.robot, future_sec
            )
            self.robot_future_len: int = self.robot_future_np.shape[0]

        ### MAP ###
        self.map_np: Optional[np.ndarray] = None
        if incl_map:
            self.map_np = self.get_map(scene_time_agent, agent_info)

        # self.plot()

    def get_agent_history(
        self,
        agent_info: AgentMetadata,
        history_sec: Tuple[Optional[float], Optional[float]],
    ) -> np.ndarray:
        agent_history_df: pd.DataFrame = self.cache.get_agent_history(
            agent_info, self.scene_ts, history_sec
        )
        return agent_history_df.to_numpy()

    def get_agent_future(
        self,
        agent_info: AgentMetadata,
        future_sec: Tuple[Optional[float], Optional[float]],
    ) -> np.ndarray:
        agent_future_df: pd.DataFrame = self.cache.get_agent_future(
            agent_info, self.scene_ts, future_sec
        )
        return agent_future_df.to_numpy()

    # @profile
    def get_neighbor_history(
        self,
        scene_time: SceneTimeAgent,
        agent_info: AgentMetadata,
        history_sec: Tuple[Optional[float], Optional[float]],
        distance_limit: Callable[[np.ndarray, int], np.ndarray],
    ) -> Tuple[int, np.ndarray, List[np.ndarray], np.ndarray]:
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
        all_agents_np, neighbor_history_lens_np = self.cache.get_agents_history(
            scene_ts, nearby_agents, history_sec
        )

        neighbor_histories: List[np.ndarray] = np.vsplit(
            all_agents_np, neighbor_history_lens_np.cumsum()
        )

        return (
            num_neighbors,
            neighbor_types_np,
            # The last one will always be empty because of what cumsum returns above.
            neighbor_histories[:-1],
            neighbor_history_lens_np,
        )

    def get_robot_current_and_future(
        self,
        robot_info: AgentMetadata,
        future_sec: Tuple[Optional[float], Optional[float]],
    ) -> np.ndarray:
        # self.scene_ts - 1 because we want to get the current timestep (scene_ts) too
        # and get_agent_future(...) gets data starting from the timestep AFTER the
        # given one.
        robot_curr_and_fut_df: pd.DataFrame = self.cache.get_agent_future(
            robot_info, self.scene_ts - 1, future_sec
        )
        return robot_curr_and_fut_df.to_numpy()

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

        if self.standardize_data:
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

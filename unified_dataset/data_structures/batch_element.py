import numpy as np
import pandas as pd

from math import floor
from typing import Tuple, Optional

from unified_dataset.data_structures.scene import SceneTime
from unified_dataset.data_structures.agent import Agent, AgentMetadata


class AgentBatchElement:
    """A single element of an agent-centric batch.
    """
    def __init__(self, 
                 scene_time: SceneTime,
                 agent_name: str, 
                 history_sec: Tuple[Optional[float], Optional[float]],
                 future_sec: Tuple[Optional[float], Optional[float]]) -> None:
        self.dt: float = scene_time.metadata.dt
        self.scene_ts: int = scene_time.ts

        agent: Agent = next((a for a in scene_time.agents if a.name == agent_name), None)

        ### AGENT-SPECIFIC DATA ###
        curr_agent_pos_np, agent_history_np = self.get_agent_history_data(scene_time, agent, history_sec)
        agent_future_np: np.ndarray = self.get_agent_future_data(scene_time, agent, future_sec)

        ### NEIGHBOR-SPECIFIC DATA ###
        neighbor_history_np: np.ndarray = self.get_neighbor_history_data(scene_time, agent, history_sec)

    def get_agent_history_data(self, scene_time: SceneTime, agent: Agent, history_sec: Tuple[Optional[float], Optional[float]]) -> Tuple[np.ndarray, np.ndarray]:
        dt: float = self.dt
        scene_ts: int = self.scene_ts

        # We don't have to check the mins here because our data_index filtering in dataset.py already
        # took care of it.
        if history_sec[1] is not None:
            max_history: int = floor(history_sec[1] / dt)
            agent_history_df: pd.DataFrame = agent.data.loc[max(scene_ts - max_history, agent.metadata.first_timestep) : scene_ts].copy()
        else:
            agent_history_df: pd.DataFrame = agent.data.loc[ : scene_ts].copy()

        curr_agent_pos_np: np.ndarray = np.array([agent_history_df.at[scene_ts, 'x'], agent_history_df.at[scene_ts, 'y']])
        agent_history_df.loc[:, ['x', 'y']] -= curr_agent_pos_np
            
        agent_history_df['sin_heading'] = np.sin(agent_history_df['heading'])
        agent_history_df['cos_heading'] = np.cos(agent_history_df['heading'])

        agent_history_np: np.ndarray = agent_history_df.loc[:, ['x', 'y', 'vx', 'vy', 'ax', 'ay', 'sin_heading', 'cos_heading']].values
        return curr_agent_pos_np, agent_history_np

    def get_agent_future_data(self, scene_time: SceneTime, agent: Agent, future_sec: Tuple[Optional[float], Optional[float]]):
        dt: float = self.dt
        scene_ts: int = self.scene_ts

        # We don't have to check the mins here because our data_index filtering in dataset.py already
        # took care of it.
        if future_sec[1] is not None:
            max_future = floor(future_sec[1] / dt)
            agent_future_df = agent.data.loc[scene_ts + 1 : min(scene_ts + max_future, agent.metadata.last_timestep), ['x', 'y']]
        else:
            agent_future_df = agent.data.loc[scene_ts + 1 : , ['x', 'y']]

        agent_future_np: np.ndarray = agent_future_df.values
        return agent_future_np

    def get_neighbor_history_data(self, scene_time: SceneTime, agent: Agent, history_sec: Tuple[Optional[float], Optional[float]]) -> np.ndarray:
        # The indices of the returned ndarray match the scene_time agents list (including the index of the central agent,
        # which would have a distance of 0 to itself).
        distance_matrix: np.ndarray = scene_time.get_agent_distances_to(agent)
        agent_idx = scene_time.agents.index(agent)

        # TODO(bivanovic): Implement distance limits based on edge (agent-agent) type.

        agent_types: np.ndarray = np.array([a.type.value for a in scene_time.agents])

        neighbor_histories = None


class SceneBatchElement:
    """A single batch element.
    """
    def __init__(self, scene_time: SceneTime, history_sec_at_most: float, future_sec_at_most: float) -> None:
        self.history_sec_at_most = history_sec_at_most

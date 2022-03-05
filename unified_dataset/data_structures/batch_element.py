import numpy as np

from math import ceil, floor
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
        dt = scene_time.metadata.dt
        scene_ts = scene_time.ts

        agent: Agent = next((a for a in scene_time.agents if a.name == agent_name), None)
        agent_info: AgentMetadata = agent.metadata

        # We don't have to check the mins here because our data_index filtering in dataset.py already
        # took care of it.
        if history_sec[1] is not None:
            max_history = floor(history_sec[1] / dt)
            agent_history = agent.data.loc[max(scene_ts - max_history, agent_info.first_timestep) : scene_ts]
        else:
            agent_history = agent.data.loc[ : scene_ts]

        if future_sec[1] is not None:
            max_future = floor(future_sec[1] / dt)
            agent_future = agent.data.loc[scene_ts + 1 : min(scene_ts + max_future, agent_info.last_timestep)]
        else:
            agent_future = agent.data.loc[scene_ts + 1 : ]

        pass


class SceneBatchElement:
    """A single batch element.
    """
    def __init__(self, scene_time: SceneTime, history_sec_at_most: float, future_sec_at_most: float) -> None:
        self.history_sec_at_most = history_sec_at_most

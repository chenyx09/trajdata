from unified_dataset.data_structures.scene import SceneTime
from unified_dataset.data_structures.agent import Agent


class AgentBatchElement:
    """A single element of an agent-centric batch.
    """
    def __init__(self, scene_time: SceneTime, agent_name: str, history_sec_at_most: float, future_sec_at_most: float) -> None:
        dt = scene_time.metadata.dt
        scene_ts = scene_time.ts

        history_timesteps = int(history_sec_at_most / dt)
        future_timesteps = int(future_sec_at_most / dt)

        agent: Agent = next((a for a in scene_time.agents if a.name == agent_name), None)

        agent_history = agent


class SceneBatchElement:
    """A single batch element.
    """
    def __init__(self, scene_time: SceneTime, history_sec_at_most: float, future_sec_at_most: float) -> None:
        self.history_sec_at_most = history_sec_at_most

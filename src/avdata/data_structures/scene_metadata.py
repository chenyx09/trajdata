from typing import Any, List, Optional, Set, Type

from avdata.data_structures.agent import AgentMetadata
from avdata.data_structures.environment import EnvMetadata


class SceneMetadata:
    """Holds scene metadata, e.g., name, location, original data split, but without the memory footprint of all the actual underlying scene data."""

    def __init__(
        self,
        env_metadata: EnvMetadata,
        name: str,
        location: str,
        data_split: str,
        length_timesteps: int,
        data_access_info: Any,
        description: Optional[str] = None,
        agents: Optional[List[AgentMetadata]] = None,
        agent_presence: Optional[List[List[AgentMetadata]]] = None,
    ) -> None:
        self.env_metadata = env_metadata
        self.env_name = env_metadata.name
        self.name = name
        self.location = location
        self.data_split = data_split
        self.dt = env_metadata.dt
        self.length_timesteps = length_timesteps
        self.data_access_info = data_access_info
        self.description = description
        self.agents = agents
        self.agent_presence = agent_presence

    def length_seconds(self) -> float:
        return self.length_timesteps * self.dt

    def __repr__(self) -> str:
        return "/".join([self.env_name, self.name])

    def update_agent_info(
        self,
        new_agents: List[AgentMetadata],
        new_agent_presence: List[List[AgentMetadata]],
    ) -> None:
        self.agents = new_agents
        self.agent_presence = new_agent_presence
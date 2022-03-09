from pathlib import Path
from typing import Any, List, Optional, Set

import dill
import numpy as np

from unified_dataset.data_structures.agent import Agent, AgentMetadata, AgentType
from unified_dataset.data_structures.environment import EnvMetadata


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
        self.agent_presence = agent_presence

    def length_seconds(self) -> float:
        return self.length_timesteps * self.dt

    def __repr__(self) -> str:
        return "/".join([self.env_name, self.name])

    def update_agent_presence(
        self, new_agent_presence: List[List[AgentMetadata]]
    ) -> None:
        self.agent_presence = new_agent_presence


class Scene:
    """Holds the data for a particular scene."""

    def __init__(self, metadata: SceneMetadata) -> None:
        self.metadata = metadata


class SceneTime:
    """Holds the data for a particular scene at a particular timestep."""

    def __init__(
        self, metadata: SceneMetadata, scene_ts: int, agents: List[Agent]
    ) -> None:
        self.metadata = metadata
        self.ts = scene_ts
        self.agents = agents

    @classmethod
    def from_cache(
        cls,
        scene_info: SceneMetadata,
        scene_ts: int,
        scene_cache_dir: Path,
        only_types: Optional[Set[AgentType]] = None,
        no_types: Optional[Set[AgentType]] = None,
    ):
        agents_present: List[AgentMetadata] = scene_info.agent_presence[scene_ts]

        agents: List[Agent] = list()
        for agent_info in agents_present:
            if no_types is not None and agent_info.type in no_types:
                continue

            if only_types is None or agent_info.type in only_types:
                agent_file = scene_cache_dir / f"{agent_info.name}.dill"
                with open(agent_file, "rb") as f:
                    agent: Agent = dill.load(f)

                agents.append(agent)

        return cls(scene_info, scene_ts, agents)

    def get_agent_distances_to(self, agent: Agent) -> np.ndarray:
        agent_pos = np.array(
            [[agent.data.at[self.ts, "x"], agent.data.at[self.ts, "y"]]]
        )
        curr_poses = np.stack([a.data.loc[self.ts, ["x", "y"]] for a in self.agents])

        return np.linalg.norm(curr_poses - agent_pos, axis=1)

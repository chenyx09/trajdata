import contextlib
import sqlite3
from pathlib import Path
from typing import Any, List, Optional, Set

import numpy as np
import pandas as pd

from avdata.data_structures.agent import Agent, AgentMetadata, AgentType
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
        self,
        metadata: SceneMetadata,
        scene_ts: int,
        agents: List[Agent],
        scene_cache_dir: Path,
    ) -> None:
        self.metadata = metadata
        self.ts = scene_ts
        self.agents = agents
        self.scene_cache_dir = scene_cache_dir

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
        filtered_agents: List[AgentMetadata] = [
            agent_info
            for agent_info in agents_present
            if (no_types is None or agent_info.type not in no_types)
            and (only_types is None or agent_info.type in only_types)
        ]

        with contextlib.closing(
            sqlite3.connect(scene_cache_dir / "agent_data.db")
        ) as conn:
            data_df = pd.read_sql_query(
                f"SELECT * FROM agent_data WHERE agent_id IN ({','.join('?'*len(filtered_agents))})",
                conn,
                params=tuple(a.name for a in filtered_agents),
                index_col=["agent_id", "scene_ts"],
            )

        agents: List[Agent] = list()
        for agent_info in filtered_agents:
            agents.append(Agent(agent_info, data_df.loc[agent_info.name]))

        return cls(scene_info, scene_ts, agents, scene_cache_dir)

    def get_agent_distances_to(self, agent: Agent) -> np.ndarray:
        agent_pos = np.array(
            [[agent.data.at[self.ts, "x"], agent.data.at[self.ts, "y"]]]
        )

        with contextlib.closing(
            sqlite3.connect(self.scene_cache_dir / "agent_data.db")
        ) as connection:
            data_df = pd.read_sql_query(
                "SELECT agent_id,x,y FROM agent_data WHERE scene_ts=?",
                connection,
                params=(self.ts,),
                index_col="agent_id",
            )

        agent_ids = [a.name for a in self.agents]
        curr_poses = data_df.loc[agent_ids, ["x", "y"]].values
        return np.linalg.norm(curr_poses - agent_pos, axis=1)


class SceneTimeAgent:
    """Holds the data for a particular agent in a scene at a particular timestep."""

    def __init__(
        self,
        metadata: SceneMetadata,
        scene_ts: int,
        agents: List[AgentMetadata],
        agent: Agent,
        scene_cache_dir: Path,
        robot: Optional[Agent] = None,
    ) -> None:
        self.metadata = metadata
        self.ts = scene_ts
        self.agents = agents
        self.agent = agent
        self.scene_cache_dir = scene_cache_dir
        self.robot = robot

    @classmethod
    def from_cache(
        cls,
        scene_info: SceneMetadata,
        scene_ts: int,
        agent_id: str,
        scene_cache_dir: Path,
        only_types: Optional[Set[AgentType]] = None,
        no_types: Optional[Set[AgentType]] = None,
        incl_robot_future: bool = False,
    ):
        agents_present: List[AgentMetadata] = scene_info.agent_presence[scene_ts]
        filtered_agents: List[AgentMetadata] = [
            agent_info
            for agent_info in agents_present
            if (no_types is None or agent_info.type not in no_types)
            and (only_types is None or agent_info.type in only_types)
        ]

        agent_metadata = next((a for a in filtered_agents if a.name == agent_id), None)

        with contextlib.closing(
            sqlite3.connect(scene_cache_dir / "agent_data.db")
        ) as conn:
            if incl_robot_future:
                ego_metadata = next(
                    (a for a in filtered_agents if a.name == "ego"), None
                )

                data_df = pd.read_sql_query(
                    f"SELECT * FROM agent_data WHERE agent_id IN (?, ?)",
                    conn,
                    params=(agent_id, "ego"),
                    index_col=["agent_id", "scene_ts"],
                )

                return cls(
                    scene_info,
                    scene_ts,
                    agents=filtered_agents,
                    agent=Agent(agent_metadata, data_df.loc[agent_id]),
                    scene_cache_dir=scene_cache_dir,
                    robot=Agent(ego_metadata, data_df.loc["ego"]),
                )
            else:
                data_df = pd.read_sql_query(
                    f"SELECT * FROM agent_data WHERE agent_id=?",
                    conn,
                    params=(agent_id,),
                    index_col="scene_ts",
                )

                del data_df["agent_id"]

                return cls(
                    scene_info,
                    scene_ts,
                    agents=filtered_agents,
                    agent=Agent(agent_metadata, data_df),
                    scene_cache_dir=scene_cache_dir,
                )

    # @profile
    def get_agent_distances_to(self, agent: Agent) -> np.ndarray:
        agent_pos = np.array(
            [[agent.data.at[self.ts, "x"], agent.data.at[self.ts, "y"]]]
        )

        with contextlib.closing(
            sqlite3.connect(self.scene_cache_dir / "agent_data.db")
        ) as connection:
            data_df = pd.read_sql_query(
                "SELECT agent_id,x,y FROM agent_data WHERE scene_ts=?",
                connection,
                params=(self.ts,),
                index_col="agent_id",
            )

        agent_ids = [a.name for a in self.agents]
        curr_poses = data_df.loc[agent_ids, ["x", "y"]].values
        return np.linalg.norm(curr_poses - agent_pos, axis=1)

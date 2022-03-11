import contextlib
import sqlite3
from collections import namedtuple
from enum import IntEnum
from sqlite3 import Connection
from typing import Optional

import pandas as pd


class AgentType(IntEnum):
    UNKNOWN = -1
    VEHICLE = 0
    PEDESTRIAN = 1
    BICYCLE = 2
    MOTORCYCLE = 3


FixedSize = namedtuple("FixedSize", ["length", "width", "height"])


class AgentMetadata:
    """Holds node metadata, e.g., name, type, but without the memory footprint of all the actual underlying scene data."""

    def __init__(
        self,
        name: str,
        agent_type: AgentType,
        first_timestep: int,
        last_timestep: int,
        fixed_size: Optional[FixedSize] = None,
    ) -> None:
        self.name = name
        self.type = agent_type
        self.first_timestep = first_timestep
        self.last_timestep = last_timestep
        self.fixed_size = fixed_size


class Agent:
    """Holds the data for a particular node."""

    def __init__(
        self,
        metadata: AgentMetadata,
        data: pd.DataFrame,
    ) -> None:
        self.name = metadata.name
        self.type = metadata.type
        self.metadata = metadata
        self.data = data

    @classmethod
    def from_cache(cls, metadata: AgentMetadata, db_connection: Connection):
        data_df = pd.read_sql_query(
            "SELECT * FROM agent_data WHERE agent_id=?",
            db_connection,
            params=(metadata.name,),
            index_col="scene_ts",
        )

        del data_df["agent_id"]

        return cls(metadata, data_df)

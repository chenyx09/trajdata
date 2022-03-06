import pandas as pd
from enum import IntEnum
from typing import Optional
from collections import namedtuple


class AgentType(IntEnum):
    UNKNOWN = -1
    VEHICLE = 0
    PEDESTRIAN = 1
    BICYCLE = 2
    MOTORCYCLE = 3


FixedSize = namedtuple('FixedSize', ['length', 'width', 'height'])


class AgentMetadata:
    """Holds node metadata, e.g., name, type, but without the memory footprint of all the actual underlying scene data.
    """
    def __init__(self, name: str, agent_type: AgentType, first_timestep: int, last_timestep: int) -> None:
        self.name = name
        self.type = agent_type
        self.first_timestep = first_timestep
        self.last_timestep = last_timestep


class Agent:
    """Holds the data for a particular node.
    """
    def __init__(self, metadata: AgentMetadata, data: pd.DataFrame, fixed_size: Optional[FixedSize] = None) -> None:
        self.name = metadata.name
        self.type = metadata.type
        self.metadata = metadata
        self.data = data
        self.fixed_size = fixed_size

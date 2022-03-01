import numpy as np
import pandas as pd
from enum import Enum


class AgentType(Enum):
    UNKNOWN = -1
    VEHICLE = 0
    PEDESTRIAN = 1
    BICYCLE = 2
    MOTORCYCLE = 3


class AgentMetadata:
    """Holds node metadata, e.g., name, type, but without the memory footprint of all the actual underlying scene data.
    """
    def __init__(self, name: str, agent_type: str, first_timestep: int) -> None:
        self.name = name
        self.type = agent_type
        self.first_timestep = first_timestep
        

class Agent:
    """Holds the data for a particular node.
    """
    def __init__(self, metadata: AgentMetadata, data: pd.DataFrame) -> None:
        self.name = metadata.name
        self.type = metadata.type
        self.metadata = metadata
        self.data = data

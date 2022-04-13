from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import pandas as pd


class AgentType(IntEnum):
    UNKNOWN = 0
    VEHICLE = 1
    PEDESTRIAN = 2
    BICYCLE = 3
    MOTORCYCLE = 4


class Extent:
    pass


@dataclass
class FixedExtent(Extent):
    length: float
    width: float
    height: float


class VariableExtent(Extent):
    pass


class AgentMetadata:
    """Holds node metadata, e.g., name, type, but without the memory footprint of all the actual underlying scene data."""

    def __init__(
        self,
        name: str,
        agent_type: AgentType,
        first_timestep: int,
        last_timestep: int,
        extent: Extent = None,
    ) -> None:
        self.name = name
        self.type = agent_type
        self.first_timestep = first_timestep
        self.last_timestep = last_timestep
        self.extent = extent

    def __repr__(self) -> str:
        return "/".join([self.type.name, self.name])

    def get_extents(self, start_ts: int, end_ts: int) -> np.ndarray:
        """Get the agent's extents within the specified scene timesteps.

        Args:
            start_ts (int): The first scene timestep to get extents for (inclusive)
            end_ts (int): The last scene timestep to get extents for (inclusive)

        Returns:
            np.ndarray: The extents as a (T, 3)-shaped ndarray (length, width, height)
        """
        if isinstance(
            self.extent, FixedExtent
        ):  # TODO(bivanovic): Handle variable extents and implement it for Lyft.
            return np.repeat(
                np.array([[self.extent.length, self.extent.width, self.extent.height]]),
                end_ts - start_ts + 1,
                axis=0,
            )
        else:
            return self.extent[
                start_ts - self.first_timestep : end_ts - self.first_timestep + 1
            ]


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

from typing import Any

from unified_dataset.data_structures.environment import EnvMetadata


class SceneMetadata:
    """Holds scene metadata, e.g., name, location, original data split, but without the memory footprint of all the actual underlying scene data.
    """
    def __init__(self, 
                 env_metadata: EnvMetadata, 
                 name: str, 
                 location: str, 
                 data_split: str, 
                 length_timesteps: int,
                 data_access_info: Any) -> None:
        self.env_metadata = env_metadata
        self.name = name
        self.location = location
        self.data_split = data_split
        self.dt = env_metadata.dt
        self.length_timesteps = length_timesteps
        self.data_access_info = data_access_info

    def length_seconds(self) -> float:
        return self.length_timesteps * self.dt

    def __repr__(self) -> str:
        return '/'.join([self.env_metadata.name, self.name])


class Scene:
    """Holds the data for a particular scene at a particular timestep.
    """
    def __init__(self, metadata: SceneMetadata) -> None:
        self.metadata = metadata

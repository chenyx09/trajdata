from avdata.data_structures.agent import AgentMetadata
from avdata.data_structures.scene import SceneMetadata


class SceneTimeNodeMetadata:
    """Holds metadata for a particular node at a particular timestep in a particular scene, e.g., agent name, agent type, timestep, etc., but without the memory footprint of all the actual underlying node data."""

    def __init__(
        self, scene_metadata: SceneMetadata, timestep: int, node_metadata: AgentMetadata
    ) -> None:
        self.scene_metadata = scene_metadata
        self.timestep = timestep
        self.node_metadata = node_metadata


class SceneTimeNode:
    """Holds the data for a particular node at a particular timestep in a particular scene."""

    def __init__(self, metadata: SceneTimeNodeMetadata) -> None:
        self.metadata = metadata

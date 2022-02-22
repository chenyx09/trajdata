from unified_dataset.data_structures.scene import SceneMetadata

class SceneTimeMetadata:
    """Holds scene metadata at a particular timestep, e.g., name, location, original data split, and timestep, but without the memory footprint of all the actual underlying scene data.
    """
    def __init__(self, scene_metadata: SceneMetadata, timestep: int) -> None:
        self.scene_metadata = scene_metadata
        self.timestep = timestep


class SceneTime:
    """Holds the data for a particular scene at a particular timestep.
    """
    def __init__(self, metadata: SceneTimeMetadata) -> None:
        self.metadata = metadata

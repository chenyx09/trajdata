class SceneMetadata:
    """Holds scene metadata, e.g., name, location, original data split, but without the memory footprint of all the actual underlying scene data.
    """
    def __init__(self, dataset: str, name: str, location: str, data_split: str, data_access_info) -> None:
        self.dataset = dataset
        self.name = name
        self.location = location
        self.data_split = data_split
        self.data_access_info = data_access_info

    def __repr__(self) -> str:
        return '/'.join([self.dataset, self.name])


class Scene:
    """Holds the data for a particular scene at a particular timestep.
    """
    def __init__(self, metadata: SceneMetadata) -> None:
        self.metadata = metadata

class NodeMetadata:
    """Holds node metadata, e.g., name, type, but without the memory footprint of all the actual underlying scene data.
    """
    def __init__(self, name: str, agent_type: str) -> None:
        self.name = name
        self.type = agent_type


class Node:
    """Holds the data for a particular node.
    """
    def __init__(self, metadata: NodeMetadata) -> None:
        self.metadata = metadata

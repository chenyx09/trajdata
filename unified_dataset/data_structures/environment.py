import itertools
from typing import List, Tuple


class EnvMetadata:
    def __init__(self, name: str, data_dir: str, dt: float, parts: List[Tuple[str]]) -> None:
        self.name = name
        self.data_dir = data_dir
        self.dt = dt
        self.parts = parts
        self.components = list(
            itertools.product( # Cartesian product of the below list of tuples
                *([(name, )] + parts)
            )
        )
        
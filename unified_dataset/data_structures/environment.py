import itertools
from pathlib import Path
from typing import Dict, List, Tuple


class EnvMetadata:
    def __init__(
        self,
        name: str,
        data_dir: str,
        dt: float,
        parts: List[Tuple[str]],
        scene_split_map: Dict[str, str],
    ) -> None:
        self.name = name
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.dt = dt
        self.parts = parts
        self.components = list(
            itertools.product(  # Cartesian product of the given list of tuples
                *([(name,)] + parts)
            )
        )
        self.scene_split_map = scene_split_map

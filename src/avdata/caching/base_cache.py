from pathlib import Path
from typing import List, NamedTuple, Type

import dill
import pandas as pd

from avdata.data_structures.scene_metadata import SceneMetadata


class BaseCache:
    def __init__(self, cache_location: str) -> None:
        # Ensuring the specified cache folder exists
        self.path = Path(cache_location).expanduser().resolve()
        self.path.mkdir(parents=True, exist_ok=True)

    def load_scene_metadata(self, env_name: str, scene_name: str) -> SceneMetadata:
        scene_cache_dir: Path = self.path / env_name / scene_name
        scene_file: Path = scene_cache_dir / "scene_metadata.dill"
        with open(scene_file, "rb") as f:
            scene_metadata: SceneMetadata = dill.load(f)

        return scene_metadata

    def load_env_scenes_list(self, env_name: str) -> List[Type[NamedTuple]]:
        env_cache_dir: Path = self.path / env_name
        with open(env_cache_dir / "scenes_list.dill", "rb") as f:
            scenes_list: List[Type[NamedTuple]] = dill.load(f)

        return scenes_list

    def save_env_scenes_list(
        self, env_name: str, scenes_list: List[Type[NamedTuple]]
    ) -> None:
        env_cache_dir: Path = self.path / env_name
        env_cache_dir.mkdir(parents=True, exist_ok=True)
        with open(env_cache_dir / "scenes_list.dill", "wb") as f:
            dill.dump(scenes_list, f)

    def load_single_agent_data(self, agent_id: str) -> pd.DataFrame:
        return NotImplementedError()

    def load_multiple_agent_data(self, agent_ids: List[str]) -> pd.DataFrame:
        return NotImplementedError()

    def load_agent_xy_at_time(self, scene_ts: int) -> pd.DataFrame:
        return NotImplementedError()

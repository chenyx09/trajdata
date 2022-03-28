from pathlib import Path
from typing import Iterable
import pandas as pd
from avdata.data_structures.scene_metadata import SceneMetadata


class SceneCache:
    def __init__(self, cache_path: Path, scene_info: SceneMetadata) -> None:
        self.path = cache_path
        self.scene_info = scene_info

        # Ensuring the scene cache folder exists
        self.scene_dir: Path = self.path / self.scene_info.env_name / self.scene_info.name
        self.scene_dir.mkdir(parents=True, exist_ok=True)

    def save_agent_data(
        self, agent_data: pd.DataFrame
    ) -> None:
        raise NotImplementedError()

    def load_single_agent_data(
        self, agent_id: str
    ) -> pd.DataFrame:
        raise NotImplementedError()

    def load_multiple_agent_data(
        self, agent_ids: Iterable[str]
    ) -> pd.DataFrame:
        raise NotImplementedError()

    def load_all_agent_data(self) -> pd.DataFrame:
        raise NotImplementedError()

    def load_agent_xy_at_time(
        self, scene_ts: int
    ) -> pd.DataFrame:
        raise NotImplementedError()

    def load_data_between_times(
        from_ts: int, to_ts: int
    ) -> pd.DataFrame:
        raise NotImplementedError()

from pathlib import Path
from typing import List, NamedTuple, Type

import dill

from avdata.data_structures.scene_metadata import SceneMetadata


class EnvCache:
    def __init__(self, cache_location: str) -> None:
        # Ensuring the specified cache folder exists
        self.path = Path(cache_location).expanduser().resolve()
        self.path.mkdir(parents=True, exist_ok=True)

    def env_is_cached(self, env_name: str) -> bool:
        return (self.path / env_name / "scenes_list.dill").is_file()

    def scene_is_cached(self, env_name: str, scene_name: str) -> bool:
        return (self.path / env_name / scene_name / "scene_metadata.dill").is_file()

    def load_scene_metadata(self, env_name: str, scene_name: str) -> SceneMetadata:
        scene_cache_dir: Path = self.path / env_name / scene_name
        scene_file: Path = scene_cache_dir / "scene_metadata.dill"
        with open(scene_file, "rb") as f:
            scene_metadata: SceneMetadata = dill.load(f)

        return scene_metadata

    def save_scene_metadata(self, scene_info: SceneMetadata) -> None:
        scene_cache_dir: Path = self.path / scene_info.env_name / scene_info.name
        scene_cache_dir.mkdir(parents=True, exist_ok=True)

        scene_file: Path = scene_cache_dir / "scene_metadata.dill"
        with open(scene_file, "wb") as f:
            dill.dump(scene_info, f)

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

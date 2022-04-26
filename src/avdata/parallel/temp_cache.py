from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Union

import dill

from avdata.data_structures.scene_metadata import SceneMetadata


class TemporaryCache:
    def __init__(self, temp_dir: Optional[str] = None) -> None:
        self.temp_dir: Optional[TemporaryDirectory] = None
        if temp_dir is None:
            self.temp_dir: TemporaryDirectory = TemporaryDirectory()
            self.path: Path = Path(self.temp_dir.name)
        else:
            self.path: Path = Path(temp_dir)

    def cache(
        self, scene_info: SceneMetadata, ret_str: bool = False
    ) -> Union[Path, str]:
        tmp_file_path: Path = self.path / TemporaryCache.get_file_path(scene_info)
        with open(tmp_file_path, "wb") as f:
            dill.dump(scene_info, f)

        if ret_str:
            return str(tmp_file_path)
        else:
            return tmp_file_path

    def cleanup(self) -> None:
        self.temp_dir.cleanup()

    @staticmethod
    def get_file_path(scene_info: SceneMetadata) -> Path:
        return f"{scene_info.env_name}_{scene_info.name}.dill"

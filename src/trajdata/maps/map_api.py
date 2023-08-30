from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from trajdata.maps.map_kdtree import MapElementKDTree
    from trajdata.caching.scene_cache import SceneCache

from pathlib import Path
from typing import Dict

from trajdata.maps.vec_map import VectorMap
from trajdata.proto.vectorized_map_pb2 import VectorizedMap
from trajdata.utils import map_utils


class MapAPI:
    def __init__(self, unified_cache_path: Union[Path, str], data_dirs: Optional[Dict] = None) -> None:
        self.unified_cache_path: Path = Path(unified_cache_path)
        """A simple interface for loading trajdata's vector maps which does not require
        instantiation of a `UnifiedDataset` object.

        Args:
            unified_cache_path (Path): Path to trajdata's local cache on disk.
            keep_in_memory (bool): Whether loaded maps should be stored
            in memory (memoized) for later re-use. For most cases (e.g., batched dataloading),
            this is a good idea. However, this can cause rapid memory usage growth for some
            datasets (e.g., Waymo) and it can be better to disable this. Defaults to False.
        """
        self.unified_cache_path: Path = unified_cache_path
        self.maps: Dict[str, VectorMap] = dict()
        self.data_dirs = data_dirs

    def get_map(
        self, map_id: str, scene_cache: Optional[SceneCache] = None, **kwargs
    ) -> VectorMap:
        if map_id not in self.maps:
            env_name, map_name = map_id.split(":")
            env_maps_path: Path = self.unified_cache_path / env_name / "maps"
            vec_map_path: Path = env_maps_path / f"{map_name}.pb"

            if not Path.exists(vec_map_path):
                if self.data_dirs is None:
                    raise ValueError(
                        f"There is no cached map at {vec_map_path} and there was no " + 
                        "`data_dirs` provided to rebuild cache.")

                # Rebuild maps by creating a dummy dataset object.
                # TODO(pkarkus) We need support for rebuilding map files only, without creating dataset and building agent data.
                from trajdata.dataset import UnifiedDataset
                dataset = UnifiedDataset(
                    desired_data=[env_name],
                    rebuild_maps=True,
                    data_dirs=self.data_dirs,
                    cache_location=self.unified_cache_path,
                    verbose=True,
                )
                # Hopefully we successfully created map cache.

            stored_vec_map: VectorizedMap = map_utils.load_vector_map(vec_map_path)

            vec_map: VectorMap = VectorMap.from_proto(stored_vec_map, **kwargs)
            vec_map.search_kdtrees: Dict[
                str, MapElementKDTree
            ] = map_utils.load_kdtrees(env_maps_path / f"{map_name}_kdtrees.dill")

            self.maps[map_id] = vec_map

        if scene_cache is not None:
            self.maps[map_id].associate_scene_data(
                scene_cache.get_traffic_light_status_dict()
            )

        return self.maps[map_id]

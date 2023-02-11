from pathlib import Path

import numpy as np

from trajdata.maps.map_api import MapAPI
from trajdata.visualization.interactive_figure import InteractiveFigure


def main():
    cache_path = Path("~/.unified_data_cache").expanduser()
    map_api = MapAPI(cache_path)

    for carla_town in [f"Town0{x}" for x in range(1, 8)] + ["Town10", "Town10HD"]:
        vec_map = map_api.get_map(
            f"carla:{carla_town}",
            incl_road_lanes=True,
            incl_road_areas=True,
            incl_ped_crosswalks=True,
            incl_ped_walkways=True,
        )

        fig = InteractiveFigure()
        fig.add_map(
            map_from_world_tf=np.eye(4),
            vec_map=vec_map,
        )
        fig.show()


if __name__ == "__main__":
    main()

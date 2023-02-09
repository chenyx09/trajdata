from pathlib import Path

import numpy as np

from trajdata.maps.map_api import MapAPI
from trajdata.visualization.interactive_figure import InteractiveFigure


def main():
    cache_path = Path("~/.unified_data_cache").expanduser()
    map_api = MapAPI(cache_path)

    vec_map = map_api.get_map(
        "carla:Town01",
        incl_road_lanes=True,
        incl_road_areas=True,
        incl_ped_crosswalks=True,
        incl_ped_walkways=True,
    )

    fig = InteractiveFigure()
    fig.add_map(
        map_from_world_tf=np.eye(4),
        vec_map=vec_map,
        bbox=(
            vec_map.extent[0],
            vec_map.extent[3],
            vec_map.extent[1],
            vec_map.extent[4],
        ),
    )
    fig.show()


if __name__ == "__main__":
    main()

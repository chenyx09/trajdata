from pathlib import Path
from matplotlib import pyplot as plt

import numpy as np

from trajdata.maps.map_api import MapAPI
from trajdata.visualization.interactive_figure import InteractiveFigure


def main():
    cache_path = Path("~/.unified_data_cache").expanduser()
    map_api = MapAPI(cache_path)

    for carla_town in [f"Town0{x}" for x in range(1, 8)] + ["Town10", "Town10HD"]: # ["main"]:
    # for carla_town in ["main"]:
        vec_map = map_api.get_map(
            f"drivesim:main" if carla_town == "main" else f"carla:{carla_town}",
            incl_road_lanes=True,
            incl_road_areas=True,
            incl_ped_crosswalks=True,
            incl_ped_walkways=True,
        )

        # fig, ax = plt.subplots()
        # map_img, raster_from_world = vec_map.rasterize(
        #     resolution=2,
        #     return_tf_mat=True,
        #     incl_centerlines=False,
        #     area_color=(255, 255, 255),
        #     edge_color=(0, 0, 0),
        # )
        # ax.imshow(map_img, alpha=0.5, origin="lower")
        # vec_map.visualize_lane_graph(
        #     origin_lane=vec_map.get_road_lane("28_s0_-1"),
        #     num_hops=5,
        #     raster_from_world=raster_from_world,
        #     ax=ax
        # )
        # ax.axis("equal")
        # ax.grid(None)
        # plt.show()

        fig = InteractiveFigure()
        fig.add_map(
            map_from_world_tf=np.eye(4),
            vec_map=vec_map,
        )
        fig.show()


if __name__ == "__main__":
    main()

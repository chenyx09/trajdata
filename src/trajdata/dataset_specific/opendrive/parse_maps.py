import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List, Set, Union, Optional
import io

import imap.global_var as global_var
import matplotlib.pyplot as plt
import numpy as np
from imap.lib.opendrive.junction import Junction as ODRJunction
from imap.lib.opendrive.lanes import Lane as ODRLane
from imap.lib.opendrive.lanes import LaneSection as ODRLaneSection
from imap.lib.opendrive.map import Map as ODRMap
from imap.lib.opendrive.road import Road as ODRRoad
from tqdm import tqdm

from trajdata.caching.df_cache import DataFrameCache
from trajdata.maps.vec_map import VectorMap
from trajdata.maps.vec_map_elements import (
    MapElementType,
    PedCrosswalk,
    PedWalkway,
    Polyline,
    RoadLane,
)
from trajdata.visualization.interactive_figure import InteractiveFigure


driving_lane_types: Set[str] = {
    "driving",
    "bidirectional", # This is kinda wonky when it comes to lane/road connections.
    "exit",
    "entry",
    "onRamp",
    "offRamp",
    "connectingRamp",
    "parking",
}


def get_lane_id(road_id: str, section_idx: int, lane_id: str):
    return "_".join((road_id, f"s{section_idx}", lane_id))


def init_global_vars(sampling=0.3, disable_z_axis=False):
    # 1. Init global var
    global_var._init()
    global_var.set_element_vaule("sampling_length", sampling)
    global_var.set_element_vaule("debug_mode", True)
    global_var.set_element_vaule("enable_z_axis", not disable_z_axis)


def parse_maps(maps_dir_or_file: str, trajdata_cache_dir: str, sampling=0.3, disable_z_axis=False):

    init_global_vars(sampling, disable_z_axis)

    if maps_dir_or_file.endswith(".xodr"):
        map_files = [Path(maps_dir_or_file)]
    else:
        map_files = [Path(p) for p in glob.glob(os.path.join(maps_dir_or_file, "*.xodr"))]

    for map_file in map_files:
        map_env: str
        if map_file.stem.startswith("Town"):
            map_env = "carla"
        else:
            map_env = "drivesim"
        map_id = f"{map_env}:{map_file.stem}"
        
        vector_map = parse_opendrive_map(map_id, map_file)

        # Saving the resulting map.
        DataFrameCache.finalize_and_cache_map(
            Path(trajdata_cache_dir), vector_map, {"px_per_m": 2}
        )

        # fig, ax = plt.subplots()
        # map_img, raster_from_world = vector_map.rasterize(
        #     resolution=2,
        #     return_tf_mat=True,
        #     incl_centerlines=False,
        #     area_color=(255, 255, 255),
        #     edge_color=(0, 0, 0),
        # )
        # ax.imshow(map_img, alpha=0.5, origin="lower")
        # vector_map.visualize_lane_graph(
        #     origin_lane=vector_map.get_road_lane(get_lane_id("197", 0, "-5")),
        #     num_hops=5,
        #     raster_from_world=raster_from_world,
        #     ax=ax
        # )
        # ax.axis("equal")
        # ax.grid(None)
        # plt.show()

        # fig = InteractiveFigure()
        # fig.add_map(
        #     map_from_world_tf=np.eye(4),
        #     vec_map=vector_map,
        #     bbox=(
        #         minimum_bound[0],
        #         maximum_bound[0],
        #         minimum_bound[1],
        #         maximum_bound[1],
        #     ),
        # )
        # fig.show()


def parse_opendrive_map(map_id: str, map_filename: Optional[str], map_rawstring: Optional[str] = None) -> VectorMap:
    if map_filename is not None and map_rawstring is not None:
        raise ValueError("Only one of filename and rawstring can be specified.")

    vector_map = VectorMap(map_id=map_id)

    maximum_bound: np.ndarray = np.full((3,), np.nan)
    minimum_bound: np.ndarray = np.full((3,), np.nan)

    od_map = ODRMap()
    od_map.load(map_filename, rawstring=map_rawstring)

    road: ODRRoad
    for road in tqdm(od_map.roads.values(), desc=vector_map.map_name, leave=False):
        road_id = road.road_id

        road.generate_reference_line()
        road.process_lanes()

        # fig, ax = plt.subplots()
        # pts = np.stack([(p.x, p.y, p.z) for p in road.lanes.lane_sections[0].left[0].center_line], axis=0)
        # ax.plot(pts[:, 0], pts[:, 1], label="Left")
        # ax.scatter([pts[0, 0]], [pts[0, 1]])

        # pts = np.stack([(p.x, p.y, p.z) for p in road.lanes.lane_sections[0].right[0].center_line], axis=0)
        # ax.plot(pts[:, 0], pts[:, 1], label="Right")
        # ax.scatter([pts[0, 0]], [pts[0, 1]])
        # ax.legend(loc="best")
        # plt.show()

        # TODO(bivanovic): Crosswalks are wonky as of now, won't add them to map just yet.
        # for crosswalk in road.objects.objects:
        #     polygon = []

        #     for pt in crosswalk.outline_sth:
        #         xyz = road.get_xyz_at_sth(pt[0], pt[1], pt[2])
        #         polygon.append((xyz.x, xyz.y, xyz.z))

        #     polygon = np.asarray(polygon)

        #     # Computing the maximum and minimum map coordinates.
        #     maximum_bound = np.fmax(maximum_bound, polygon.max(axis=0))
        #     minimum_bound = np.fmin(minimum_bound, polygon.min(axis=0))

        #     ped_crosswalk = PedCrosswalk(
        #         id=crosswalk.id,
        #         polygon=Polyline(polygon),
        #     )
        #     vector_map.add_map_element(ped_crosswalk)

        lane_section_obj: ODRLaneSection
        for section_idx, lane_section_obj in enumerate(road.lanes.lane_sections):
            lane: ODRLane
            for lane in lane_section_obj.left + lane_section_obj.right:
                if (
                    lane.lane_type == "none"
                    or len(lane.left_boundary) == 0
                    or len(lane.right_boundary) == 0
                ):
                    continue

                lane_id = get_lane_id(road_id, section_idx, lane.lane_id)
                left_boundary = np.stack(
                    [(p.x, p.y, p.z) for p in lane.left_boundary], axis=0
                )
                right_boundary = np.stack(
                    [(p.x, p.y, p.z) for p in lane.right_boundary], axis=0
                )

                # Computing the maximum and minimum map coordinates.
                maximum_bound = np.fmax(maximum_bound, left_boundary.max(axis=0))
                minimum_bound = np.fmin(minimum_bound, left_boundary.min(axis=0))

                maximum_bound = np.fmax(maximum_bound, right_boundary.max(axis=0))
                minimum_bound = np.fmin(minimum_bound, right_boundary.min(axis=0))

                lane_type = lane.lane_type
                if lane_type in driving_lane_types:
                    centerline = np.stack(
                        [(p.x, p.y, p.z) for p in lane.center_line], axis=0
                    )

                    assert centerline.shape[0] >= 2, f"Invalid centerline for lane_id={lane_id}"

                    # In some cases centerline contains repeated points, which causes issues in map api.
                    # TODO(pkarkus) we might want to do the same for left and right boundary.
                    consec_dist = np.linalg.norm(centerline[1:, :2] - centerline[:-1, :2], axis=-1)
                    consec_dist = np.pad(consec_dist, (1, 0), mode="constant", constant_values=1000.)
                    keep = [0]
                    removed_dist = 0.
                    for i in range(1, len(centerline)):
                        if consec_dist[i] < 0.01 and removed_dist < 0.1:
                            continue
                        else:
                            keep.append(i)
                            removed_dist = 0.
                    # Make sure we are keeping at least 2 points. 
                    # We could also just drop this lane, but that requires extra logic for connectivity
                    # afterwards. One workaround is to go through all lanes after building connectivites 
                    # and remove + reconnect invalid lanes.
                    if len(keep) == 1:
                        keep.append(len(centerline)-1)
                    centerline = centerline[keep]

                    # Computing the maximum and minimum map coordinates.
                    maximum_bound = np.fmax(maximum_bound, centerline.max(axis=0))
                    minimum_bound = np.fmin(minimum_bound, centerline.min(axis=0))

                    # "partial" because we aren't adding lane connectivity until later.
                    partial_new_lane = RoadLane(
                        id=lane_id,
                        center=Polyline(centerline),
                        left_edge=Polyline(left_boundary),
                        right_edge=Polyline(right_boundary),
                        adj_lanes_left=set(
                            [
                                get_lane_id(road_id, section_idx, lid)
                                for lid in lane.left_neighbor_forward
                            ]
                        ),
                        adj_lanes_right=set(
                            [
                                get_lane_id(road_id, section_idx, lid)
                                for lid in lane.right_neighbor_forward
                            ]
                        ),
                    )
                    vector_map.add_map_element(partial_new_lane)
                elif lane_type == "sidewalk":
                    sidewalk = PedWalkway(
                        id=lane_id,
                        polygon=Polyline(
                            np.concatenate(
                                (left_boundary, right_boundary[::-1]), axis=0
                            )
                        ),
                    )
                    vector_map.add_map_element(sidewalk)

    # Linking roads/junctions together.
    road_elems: Dict[str, RoadLane] = vector_map.elements[MapElementType.ROAD_LANE]
    for find_successor in [True, False]:
        road: ODRRoad
        for road in tqdm(
            od_map.roads.values(),
            desc=f"Linking {vector_map.map_name} Roads",
            leave=False,
        ):
            road_link = (
                road.link.successor if find_successor else road.link.predecessor
            )
            if road_link.element_type != "road":
                continue

            next_road: ODRRoad = (
                road.link.successor_road
                if find_successor
                else road.link.predecessor_road
            )

            next_road_contact_lanesec_idx: int
            if road_link.contact_point == "start":
                next_road_contact_lanesec_idx = 0
            else:
                next_road_contact_lanesec_idx = (
                    len(next_road.lanes.lane_sections) - 1
                )

            next_road_contact_lanesec: ODRLaneSection = (
                next_road.lanes.lane_sections[next_road_contact_lanesec_idx]
            )

            lanesec: ODRLaneSection
            road_lane_secs: List[ODRLaneSection] = road.lanes.lane_sections
            for lanesec_idx, lanesec in enumerate(road_lane_secs):
                if find_successor and lanesec_idx == len(road_lane_secs) - 1:
                    # Take next road to find successor.
                    next_lanesec = next_road_contact_lanesec
                    next_lanesec_idx = next_road_contact_lanesec_idx
                    next_lanesecs_road = next_road
                elif not find_successor and lanesec_idx == 0:
                    # Take previous road to find predecessor
                    next_lanesec = next_road_contact_lanesec
                    next_lanesec_idx = next_road_contact_lanesec_idx
                    next_lanesecs_road = next_road
                else:
                    next_lanesec_idx = (
                        lanesec_idx + 1 if find_successor else lanesec_idx - 1
                    )
                    next_lanesec = road_lane_secs[next_lanesec_idx]
                    next_lanesecs_road = road

                lane: ODRLane
                for lane in lanesec.left + lanesec.right:
                    if lane.lane_type not in driving_lane_types:
                        continue

                    if (find_successor and lane.link.successor is None) or (
                        not find_successor and lane.link.predecessor is None
                    ):
                        continue

                    next_lane_id = (
                        lane.link.successor.link_id
                        if find_successor
                        else lane.link.predecessor.link_id
                    )
                    if next_lane_id == 0:
                        continue

                    next_lane = next_lanesec.get_lane(next_lane_id)

                    from_lane: ODRLane = lane if find_successor else next_lane
                    from_lanesection_idx: int = (
                        lanesec_idx if find_successor else next_lanesec_idx
                    )
                    from_road: ODRRoad = (
                        road if find_successor else next_lanesecs_road
                    )

                    to_lane: ODRLane = next_lane if find_successor else lane
                    to_lanesection_idx: int = (
                        next_lanesec_idx if find_successor else lanesec_idx
                    )
                    to_road: ODRRoad = (
                        next_lanesecs_road if find_successor else road
                    )

                    from_id: str = get_lane_id(
                        from_road.road_id, from_lanesection_idx, from_lane.lane_id
                    )
                    to_id: str = get_lane_id(
                        to_road.road_id, to_lanesection_idx, to_lane.lane_id
                    )

                    road_elems[from_id].next_lanes.add(to_id)
                    road_elems[to_id].prev_lanes.add(from_id)

    junction_obj: ODRJunction
    for junction_obj in tqdm(
        od_map.junctions.values(),
        desc=f"Linking {vector_map.map_name} Junctions",
        leave=False,
    ):
        for conn in junction_obj.connections:
            incoming_road: ODRRoad = conn.incoming_road_obj
            connecting_road: ODRRoad = conn.connecting_road_obj

            is_succ_junc: bool = (
                incoming_road.link.successor.element_type == "junction"
                and incoming_road.link.successor.element_id
                == junction_obj.junction_id
            )
            is_pred_junc: bool = (
                incoming_road.link.predecessor.element_type == "junction"
                and incoming_road.link.predecessor.element_id
                == junction_obj.junction_id
            )
            if not is_succ_junc and not is_pred_junc:
                continue

            incoming_lanesec_idx = (
                len(incoming_road.lanes.lane_sections) - 1 if is_succ_junc else 0
            )
            connecting_lanesec_idx = (
                0
                if conn.contact_point == "start"
                else len(connecting_road.lanes.lane_sections) - 1
            )

            incoming_lanesec = incoming_road.lanes.lane_sections[
                incoming_lanesec_idx
            ]
            connecting_lanesec = connecting_road.lanes.lane_sections[
                connecting_lanesec_idx
            ]

            for lane_link in conn.lane_links:
                if lane_link.from_id == 0 or lane_link.to_id == 0:
                    continue

                from_lane = incoming_lanesec.get_lane(lane_link.from_id)
                to_lane = connecting_lanesec.get_lane(lane_link.to_id)

                if (
                    from_lane.lane_type in driving_lane_types
                    and to_lane.lane_type in driving_lane_types
                ):
                    from_id: str = get_lane_id(
                        incoming_road.road_id,
                        incoming_lanesec_idx,
                        from_lane.lane_id,
                    )
                    to_id: str = get_lane_id(
                        connecting_road.road_id,
                        connecting_lanesec_idx,
                        to_lane.lane_id,
                    )

                    road_elems[from_id].next_lanes.add(to_id)
                    road_elems[to_id].prev_lanes.add(from_id)

    # TODO(pkarkus) hack: add all connected lanes to both next and prev lanes.
    for _, road_lane in tqdm(
            road_elems.items(),         
            desc=f"Copy {vector_map.map_name} predecessors and successors.",
            leave=False,):
        road_lane.next_lanes.update(road_lane.prev_lanes)
        road_lane.prev_lanes.update(road_lane.next_lanes)

    # Setting the map bounds.
    # vector_map.extent is [min_x, min_y, min_z, max_x, max_y, max_z]
    vector_map.extent = np.concatenate((minimum_bound, maximum_bound))

    return vector_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--maps_dir",
        help="path to folder containing map .xodr files",
        type=str,
    )
    parser.add_argument(
        "--trajdata_cache_dir",
        help="path to trajdata cache (within which the OpenDRIVE maps will be saved)",
        type=str,
    )
    parser.add_argument(
        "-s", "--sampling", type=float, default=0.3, help="sampling length"
    )
    parser.add_argument(
        "--disable_z_axis",
        action="store_true",
        help="Whether to extract z-axis coordination information",
    )

    args = parser.parse_args()

    parse_maps(args.maps_dir, args.trajdata_cache_dir, args.sampling, args.disable_z_axis)

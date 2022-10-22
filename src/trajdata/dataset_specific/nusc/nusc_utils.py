from typing import Any, Dict, Final, List, Tuple, Union

import numpy as np
import pandas as pd
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from scipy.spatial.distance import cdist
from tqdm import tqdm

from trajdata.data_structures import Agent, AgentMetadata, AgentType, FixedExtent, Scene
from trajdata.maps import map_utils
from trajdata.proto.vectorized_map_pb2 import (
    MapElement,
    PedCrosswalk,
    PedWalkway,
    Polyline,
    RoadArea,
    RoadLane,
    VectorizedMap,
)
from trajdata.utils import arr_utils

NUSC_DT: Final[float] = 0.5


def frame_iterator(nusc_obj: NuScenes, scene: Scene) -> Dict[str, Union[str, int]]:
    """Loops through all frames in a scene and yields them for the caller to deal with the information."""
    curr_scene_token: str = scene.data_access_info["first_sample_token"]
    while curr_scene_token:
        frame = nusc_obj.get("sample", curr_scene_token)

        yield frame

        curr_scene_token = frame["next"]


def agent_iterator(nusc_obj: NuScenes, frame_info: Dict[str, Any]) -> Dict[str, Any]:
    """Loops through all annotations (agents) in a frame and yields them for the caller to deal with the information."""
    ann_token: str
    for ann_token in frame_info["anns"]:
        ann_record = nusc_obj.get("sample_annotation", ann_token)

        agent_category: str = ann_record["category_name"]
        if agent_category.startswith("vehicle") or agent_category.startswith("human"):
            yield ann_record


def get_ego_pose(nusc_obj: NuScenes, frame_info: Dict[str, Any]) -> Dict[str, Any]:
    cam_front_data = nusc_obj.get("sample_data", frame_info["data"]["CAM_FRONT"])
    ego_pose = nusc_obj.get("ego_pose", cam_front_data["ego_pose_token"])
    return ego_pose


def agg_agent_data(
    nusc_obj: NuScenes,
    agent_data: Dict[str, Any],
    curr_scene_index: int,
    frame_idx_dict: Dict[str, int],
) -> Agent:
    """Loops through all annotations of a specific agent in a scene and aggregates their data into an Agent object."""
    if agent_data["prev"]:
        print("WARN: This is not the first frame of this agent!")

    translation_list = [np.array(agent_data["translation"][:2])[np.newaxis]]
    agent_size = agent_data["size"]
    yaw_list = [Quaternion(agent_data["rotation"]).yaw_pitch_roll[0]]

    prev_idx: int = curr_scene_index
    curr_sample_ann_token: str = agent_data["next"]
    while curr_sample_ann_token:
        agent_data = nusc_obj.get("sample_annotation", curr_sample_ann_token)

        translation = np.array(agent_data["translation"][:2])
        heading = Quaternion(agent_data["rotation"]).yaw_pitch_roll[0]
        curr_idx: int = frame_idx_dict[agent_data["sample_token"]]
        if curr_idx > prev_idx + 1:
            fill_time = np.arange(prev_idx + 1, curr_idx)
            xs = np.interp(
                x=fill_time,
                xp=[prev_idx, curr_idx],
                fp=[translation_list[-1][0, 0], translation[0]],
            )
            ys = np.interp(
                x=fill_time,
                xp=[prev_idx, curr_idx],
                fp=[translation_list[-1][0, 1], translation[1]],
            )
            headings: np.ndarray = arr_utils.angle_wrap(
                np.interp(
                    x=fill_time,
                    xp=[prev_idx, curr_idx],
                    fp=np.unwrap([yaw_list[-1], heading]),
                )
            )
            translation_list.append(np.stack([xs, ys], axis=1))
            yaw_list.extend(headings.tolist())

        translation_list.append(translation[np.newaxis])
        # size_list.append(agent_data['size'])
        yaw_list.append(heading)

        prev_idx = curr_idx
        curr_sample_ann_token = agent_data["next"]

    translations_np = np.concatenate(translation_list, axis=0)

    # Doing this prepending so that the first velocity isn't zero (rather it's just the first actual velocity duplicated)
    prepend_pos = translations_np[0] - (translations_np[1] - translations_np[0])
    velocities_np = (
        np.diff(translations_np, axis=0, prepend=np.expand_dims(prepend_pos, axis=0))
        / NUSC_DT
    )

    # Doing this prepending so that the first acceleration isn't zero (rather it's just the first actual acceleration duplicated)
    prepend_vel = velocities_np[0] - (velocities_np[1] - velocities_np[0])
    accelerations_np = (
        np.diff(velocities_np, axis=0, prepend=np.expand_dims(prepend_vel, axis=0))
        / NUSC_DT
    )

    anno_yaws_np = np.expand_dims(np.stack(yaw_list, axis=0), axis=1)
    # yaws_np = np.expand_dims(
    #     np.arctan2(velocities_np[:, 1], velocities_np[:, 0]), axis=1
    # )
    # sizes_np = np.stack(size_list, axis=0)

    # import matplotlib.pyplot as plt

    # fig, ax = plt.subplots()
    # ax.plot(translations_np[:, 0], translations_np[:, 1], color="blue")
    # ax.quiver(
    #     translations_np[:, 0],
    #     translations_np[:, 1],
    #     np.cos(anno_yaws_np),
    #     np.sin(anno_yaws_np),
    #     color="green",
    #     label="annotated heading"
    # )
    # ax.quiver(
    #     translations_np[:, 0],
    #     translations_np[:, 1],
    #     np.cos(yaws_np),
    #     np.sin(yaws_np),
    #     color="orange",
    #     label="velocity heading"
    # )
    # ax.scatter([translations_np[0, 0]], [translations_np[0, 1]], color="red", label="Start", zorder=20)
    # ax.legend(loc='best')
    # plt.show()

    agent_data_np = np.concatenate(
        [translations_np, velocities_np, accelerations_np, anno_yaws_np], axis=1
    )
    last_timestep = curr_scene_index + agent_data_np.shape[0] - 1
    agent_data_df = pd.DataFrame(
        agent_data_np,
        columns=["x", "y", "vx", "vy", "ax", "ay", "heading"],
        index=pd.MultiIndex.from_tuples(
            [
                (agent_data["instance_token"], idx)
                for idx in range(curr_scene_index, last_timestep + 1)
            ],
            names=["agent_id", "scene_ts"],
        ),
    )

    agent_type = nusc_type_to_unified_type(agent_data["category_name"])
    agent_metadata = AgentMetadata(
        name=agent_data["instance_token"],
        agent_type=agent_type,
        first_timestep=curr_scene_index,
        last_timestep=last_timestep,
        extent=FixedExtent(
            length=agent_size[1], width=agent_size[0], height=agent_size[2]
        ),
    )
    return Agent(
        metadata=agent_metadata,
        data=agent_data_df,
    )


def nusc_type_to_unified_type(nusc_type: str) -> AgentType:
    if nusc_type.startswith("human"):
        return AgentType.PEDESTRIAN
    elif nusc_type == "vehicle.bicycle":
        return AgentType.BICYCLE
    elif nusc_type == "vehicle.motorcycle":
        return AgentType.MOTORCYCLE
    elif nusc_type.startswith("vehicle"):
        return AgentType.VEHICLE
    else:
        return AgentType.UNKNOWN


def agg_ego_data(nusc_obj: NuScenes, scene: Scene) -> Agent:
    translation_list: List[np.ndarray] = list()
    yaw_list: List[float] = list()
    for frame_info in frame_iterator(nusc_obj, scene):
        ego_pose = get_ego_pose(nusc_obj, frame_info)
        yaw_list.append(Quaternion(ego_pose["rotation"]).yaw_pitch_roll[0])
        translation_list.append(ego_pose["translation"][:2])

    translations_np: np.ndarray = np.stack(translation_list, axis=0)

    # Doing this prepending so that the first velocity isn't zero (rather it's just the first actual velocity duplicated)
    prepend_pos: np.ndarray = translations_np[0] - (
        translations_np[1] - translations_np[0]
    )
    velocities_np: np.ndarray = (
        np.diff(translations_np, axis=0, prepend=np.expand_dims(prepend_pos, axis=0))
        / NUSC_DT
    )

    # Doing this prepending so that the first acceleration isn't zero (rather it's just the first actual acceleration duplicated)
    prepend_vel: np.ndarray = velocities_np[0] - (velocities_np[1] - velocities_np[0])
    accelerations_np: np.ndarray = (
        np.diff(velocities_np, axis=0, prepend=np.expand_dims(prepend_vel, axis=0))
        / NUSC_DT
    )

    yaws_np: np.ndarray = np.expand_dims(np.stack(yaw_list, axis=0), axis=1)
    # yaws_np = np.expand_dims(np.arctan2(velocities_np[:, 1], velocities_np[:, 0]), axis=1)

    ego_data_np: np.ndarray = np.concatenate(
        [translations_np, velocities_np, accelerations_np, yaws_np], axis=1
    )
    ego_data_df = pd.DataFrame(
        ego_data_np,
        columns=["x", "y", "vx", "vy", "ax", "ay", "heading"],
        index=pd.MultiIndex.from_tuples(
            [("ego", idx) for idx in range(ego_data_np.shape[0])],
            names=["agent_id", "scene_ts"],
        ),
    )

    ego_metadata = AgentMetadata(
        name="ego",
        agent_type=AgentType.VEHICLE,
        first_timestep=0,
        last_timestep=ego_data_np.shape[0] - 1,
        extent=FixedExtent(length=4.084, width=1.730, height=1.562),
    )
    return Agent(
        metadata=ego_metadata,
        data=ego_data_df,
    )


def extract_lane_and_edges(
    nusc_map: NuScenesMap, lane_record
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Getting the bounding polygon vertices.
    lane_polygon_obj = nusc_map.get("polygon", lane_record["polygon_token"])
    polygon_nodes = [
        nusc_map.get("node", node_token)
        for node_token in lane_polygon_obj["exterior_node_tokens"]
    ]
    polygon_pts: np.ndarray = np.array(
        [(node["x"], node["y"]) for node in polygon_nodes]
    )

    # Getting the lane center's points.
    curr_lane = nusc_map.arcline_path_3.get(lane_record["token"], [])
    lane_midline: np.ndarray = np.array(
        arcline_path_utils.discretize_lane(curr_lane, resolution_meters=0.5)
    )[:, :2]

    # For some reason, nuScenes duplicates a few entries
    # (likely how they're building their arcline representation).
    # We delete those duplicate entries here.
    duplicate_check: np.ndarray = np.where(
        np.linalg.norm(np.diff(lane_midline, axis=0, prepend=0), axis=1) < 1e-10
    )[0]
    if duplicate_check.size > 0:
        lane_midline = np.delete(lane_midline, duplicate_check, axis=0)

    # Computing the closest lane center point to each bounding polygon vertex.
    closest_midlane_pt: np.ndarray = np.argmin(cdist(polygon_pts, lane_midline), axis=1)
    # Computing the local direction of the lane at each lane center point.
    direction_vectors: np.ndarray = np.diff(
        lane_midline,
        axis=0,
        prepend=lane_midline[[0]] - (lane_midline[[1]] - lane_midline[[0]]),
    )

    # Selecting the direction vectors at the closest lane center point per polygon vertex.
    local_dir_vecs: np.ndarray = direction_vectors[closest_midlane_pt]
    # Calculating the vectors from the the closest lane center point per polygon vertex to the polygon vertex.
    origin_to_polygon_vecs: np.ndarray = polygon_pts - lane_midline[closest_midlane_pt]

    # Computing the perpendicular dot product.
    # See https://www.xarg.org/book/linear-algebra/2d-perp-product/
    # If perp_dot_product < 0, then the associated polygon vertex is
    # on the right edge of the lane.
    perp_dot_product: np.ndarray = (
        local_dir_vecs[:, 0] * origin_to_polygon_vecs[:, 1]
        - local_dir_vecs[:, 1] * origin_to_polygon_vecs[:, 0]
    )

    # Determining which indices are on the right of the lane center.
    on_right: np.ndarray = perp_dot_product < 0
    # Determining the boundary between the left/right polygon vertices
    # (they will be together in blocks due to the ordering of the polygon vertices).
    idx_changes: int = np.where(np.roll(on_right, 1) < on_right)[0].item()

    if idx_changes > 0:
        # If the block of left/right points spreads across the bounds of the array,
        # roll it until the boundary between left/right points is at index 0.
        # This is important so that the following index selection orders points
        # without jumps.
        polygon_pts = np.roll(polygon_pts, shift=-idx_changes, axis=0)
        on_right = np.roll(on_right, shift=-idx_changes)

    left_pts: np.ndarray = polygon_pts[~on_right]
    right_pts: np.ndarray = polygon_pts[on_right]

    # Final ordering check, ensuring that the beginning of left_pts/right_pts
    # matches the beginning of the lane.
    left_order_correct: bool = np.linalg.norm(
        left_pts[0] - lane_midline[0]
    ) < np.linalg.norm(left_pts[0] - lane_midline[-1])
    right_order_correct: bool = np.linalg.norm(
        right_pts[0] - lane_midline[0]
    ) < np.linalg.norm(right_pts[0] - lane_midline[-1])

    # Reversing left_pts/right_pts in case their first index is
    # at the end of the lane.
    if not left_order_correct:
        left_pts = left_pts[::-1]
    if not right_order_correct:
        right_pts = right_pts[::-1]

    # Ensuring that left and right have the same number of points.
    # This is necessary, not for data storage but for later rasterization.
    if left_pts.shape[0] < right_pts.shape[0]:
        left_pts = map_utils.interpolate(left_pts, right_pts.shape[0])
    elif right_pts.shape[0] < left_pts.shape[0]:
        right_pts = map_utils.interpolate(right_pts, left_pts.shape[0])

    return (
        lane_midline,
        left_pts,
        right_pts,
    )


def extract_area(nusc_map: NuScenesMap, area_record) -> np.ndarray:
    token_key: str
    if "exterior_node_tokens" in area_record:
        token_key = "exterior_node_tokens"
    elif "node_tokens" in area_record:
        token_key = "node_tokens"

    polygon_nodes = [
        nusc_map.get("node", node_token) for node_token in area_record[token_key]
    ]

    return np.array([(node["x"], node["y"]) for node in polygon_nodes])


def extract_vectorized(nusc_map: NuScenesMap) -> VectorizedMap:
    vec_map = VectorizedMap()

    # Setting the map bounds.
    vec_map.max_pt.x, vec_map.max_pt.y, vec_map.max_pt.z = (
        nusc_map.explorer.canvas_max_x,
        nusc_map.explorer.canvas_max_y,
        0.0,
    )
    vec_map.min_pt.x, vec_map.min_pt.y, vec_map.min_pt.z = (
        nusc_map.explorer.canvas_min_x,
        nusc_map.explorer.canvas_min_y,
        0.0,
    )

    overall_pbar = tqdm(
        total=len(nusc_map.lane)
        + len(nusc_map.drivable_area)
        + len(nusc_map.ped_crossing)
        + len(nusc_map.walkway),
        desc=f"Getting {nusc_map.map_name} Elements",
        position=1,
        leave=False,
    )

    for lane_record in nusc_map.lane:
        center_pts, left_pts, right_pts = extract_lane_and_edges(nusc_map, lane_record)

        lane_record_token: str = lane_record["token"]

        # Adding the element to the map.
        new_element: MapElement = vec_map.elements.add()
        new_element.id = lane_record_token.encode()

        new_lane: RoadLane = new_element.road_lane
        map_utils.populate_lane_polylines(new_lane, center_pts, left_pts, right_pts)

        new_lane.entry_lanes.extend(
            lane_id.encode()
            for lane_id in nusc_map.get_incoming_lane_ids(lane_record_token)
        )
        new_lane.exit_lanes.extend(
            lane_id.encode()
            for lane_id in nusc_map.get_outgoing_lane_ids(lane_record_token)
        )

        # new_lane.adjacent_lanes_left.append(
        #     l5_lane.adjacent_lane_change_left.id
        # )
        # new_lane.adjacent_lanes_right.append(
        #     l5_lane.adjacent_lane_change_right.id
        # )

        overall_pbar.update()

    for drivable_area in nusc_map.drivable_area:
        for polygon_token in drivable_area["polygon_tokens"]:
            if polygon_token is None and vec_map.elements[-1].id == str(None).encode():
                # See below, but essentially nuScenes has two None polygon_tokens
                # back-to-back, so we don't need the second one.
                continue

            polygon_record = nusc_map.get("polygon", polygon_token)
            polygon_pts = extract_area(nusc_map, polygon_record)

            # Adding the element to the map.
            # NOTE: nuScenes has some polygon_tokens that are None, although that
            # doesn't stop the above get(...) function call so it's fine,
            # just have to be mindful of this when creating the id.
            new_element: MapElement = vec_map.elements.add()
            new_element.id = str(polygon_token).encode()

            new_area: RoadArea = new_element.road_area
            map_utils.populate_polygon(new_area.exterior_polygon, polygon_pts)

            for hole in polygon_record["holes"]:
                polygon_pts = extract_area(nusc_map, hole)
                new_hole: Polyline = new_area.interior_holes.add()
                map_utils.populate_polygon(new_hole, polygon_pts)

        overall_pbar.update()

    for ped_area_record in nusc_map.ped_crossing:
        polygon_pts = extract_area(nusc_map, ped_area_record)

        # Adding the element to the map.
        new_element: MapElement = vec_map.elements.add()
        new_element.id = ped_area_record["token"].encode()

        new_crosswalk: PedCrosswalk = new_element.ped_crosswalk
        map_utils.populate_polygon(new_crosswalk.polygon, polygon_pts)

        overall_pbar.update()

    for ped_area_record in nusc_map.walkway:
        polygon_pts = extract_area(nusc_map, ped_area_record)

        # Adding the element to the map.
        new_element: MapElement = vec_map.elements.add()
        new_element.id = ped_area_record["token"].encode()

        new_walkway: PedWalkway = new_element.ped_walkway
        map_utils.populate_polygon(new_walkway.polygon, polygon_pts)

        overall_pbar.update()

    overall_pbar.close()

    return vec_map

import glob
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Dict, Final, Generator, Iterable, List, Optional, Tuple

import numpy as np
import nuplan.planning.script.config.common as common_cfg
import pandas as pd
import yaml
from nuplan.common.maps.nuplan_map.nuplan_map import NuPlanMap
from tqdm import tqdm

from trajdata.data_structures.agent import AgentType
from trajdata.data_structures.scene_metadata import Scene
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

NUPLAN_DT: Final[float] = 0.05
NUPLAN_MAP_NAME_DICT: Final[Dict[str, str]] = {
    "boston": "us-ma-boston",
    "singapore": "sg-one-north",
    "las_vegas": "us-nv-las-vegas-strip",
    "pittsburgh": "us-pa-pittsburgh-hazelwood",
}
NUPLAN_LOCATIONS: Final[Tuple[str, str, str, str]] = tuple(NUPLAN_MAP_NAME_DICT.keys())
NUPLAN_MAP_VERSION: Final[str] = "nuplan-maps-v1.0"


class NuPlanObject:
    def __init__(self, dataset_path: Path, subfolder: str) -> None:
        self.base_path: Path = dataset_path / subfolder

        self.connection: sqlite3.Connection = None
        self.cursor: sqlite3.Cursor = None

        self.scenes: List[Dict[str, str]] = self._load_scenes()

    def open_db(self, db_filename: str) -> None:
        self.connection = sqlite3.connect(str(self.base_path / db_filename))
        self.connection.row_factory = sqlite3.Row
        self.cursor = self.connection.cursor()

    def execute_query_one(
        self, query_text: str, query_params: Optional[Iterable] = None
    ) -> sqlite3.Row:
        self.cursor.execute(
            query_text, query_params if query_params is not None else []
        )
        return self.cursor.fetchone()

    def execute_query_all(
        self, query_text: str, query_params: Optional[Iterable] = None
    ) -> List[sqlite3.Row]:
        self.cursor.execute(
            query_text, query_params if query_params is not None else []
        )
        return self.cursor.fetchall()

    def execute_query_iter(
        self, query_text: str, query_params: Optional[Iterable] = None
    ) -> Generator[sqlite3.Row, None, None]:
        self.cursor.execute(
            query_text, query_params if query_params is not None else []
        )

        for row in self.cursor:
            yield row

    def _load_scenes(self) -> List[Dict[str, str]]:
        scene_info_query = """
        SELECT  sc.token AS scene_token,
                log.location,
                log.logfile,
                (
                    SELECT count(*)
                    FROM lidar_pc AS lpc
                    WHERE lpc.scene_token = sc.token
                ) AS num_timesteps
        FROM scene AS sc
        LEFT JOIN log ON sc.log_token = log.token
        """
        scenes: List[Dict[str, str]] = []

        for log_filename in glob.glob(str(self.base_path / "*.db")):
            self.open_db(log_filename)

            for row in self.execute_query_iter(scene_info_query):
                scenes.append(
                    {
                        "name": f"{row['logfile']}={row['scene_token'].hex()}",
                        "location": row["location"],
                        "num_timesteps": row["num_timesteps"],
                    }
                )

            self.close_db()

        return scenes

    def get_scene_frames(self, scene: Scene) -> pd.DataFrame:
        query = """
        SELECT  lpc.token AS lpc_token,
                ep.x AS ego_x,
                ep.y AS ego_y,
                ep.qw AS ego_qw,
                ep.qx AS ego_qx,
                ep.qy AS ego_qy,
                ep.qz AS ego_qz,
                ep.vx AS ego_vx,
                ep.vy AS ego_vy,
                ep.acceleration_x AS ego_ax,
                ep.acceleration_y AS ego_ay
        FROM lidar_pc AS lpc
        LEFT JOIN ego_pose AS ep ON lpc.ego_pose_token = ep.token
        WHERE scene_token = ?
        ORDER BY lpc.timestamp ASC;
        """
        log_filename, scene_token_str = scene.name.split("=")
        scene_token = bytearray.fromhex(scene_token_str)

        return pd.read_sql_query(
            query, self.connection, index_col="lpc_token", params=(scene_token,)
        )

    def get_detected_agents(self, binary_lpc_tokens: List[bytearray]) -> pd.DataFrame:
        query = f"""
        SELECT  lb.lidar_pc_token,
                lb.track_token,
                (SELECT category.name FROM category WHERE category.token = tr.category_token) AS category_name,
                tr.width,
                tr.length,
                tr.height,
                lb.x,
                lb.y,
                lb.vx,
                lb.vy,
                lb.yaw
        FROM lidar_box AS lb
        LEFT JOIN track AS tr ON lb.track_token = tr.token

        WHERE lidar_pc_token IN ({('?,'*len(binary_lpc_tokens))[:-1]}) AND category_name IN ('vehicle', 'bicycle', 'pedestrian')
        """
        return pd.read_sql_query(query, self.connection, params=binary_lpc_tokens)

    def close_db(self) -> None:
        self.cursor.close()
        self.connection.close()


def nuplan_type_to_unified_type(nuplan_type: str) -> AgentType:
    if nuplan_type == "pedestrian":
        return AgentType.PEDESTRIAN
    elif nuplan_type == "bicycle":
        return AgentType.BICYCLE
    elif nuplan_type == "vehicle":
        return AgentType.VEHICLE
    else:
        return AgentType.UNKNOWN


def create_splits_logs() -> Dict[str, List[str]]:
    yaml_filepath = Path(common_cfg.__path__[0]) / "splitter" / "nuplan.yaml"
    with open(yaml_filepath, "r") as stream:
        splits = yaml.safe_load(stream)

    return splits["log_splits"]


def extract_lane_and_edges(
    nuplan_map: NuPlanMap, lane_record
) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    lane_midline = np.stack(lane_record["geometry"].xy, axis=-1)

    # Getting the bounding polygon vertices.
    boundary_df = nuplan_map._vector_map["boundaries"]
    if np.isfinite(lane_record["lane_fid"]):
        fid = str(int(lane_record["lane_fid"]))
        lane_info = nuplan_map._vector_map["lanes_polygons"].loc[fid]
    elif np.isfinite(lane_record["lane_connector_fid"]):
        fid = int(lane_record["lane_connector_fid"])
        lane_info = nuplan_map._vector_map[
            "gen_lane_connectors_scaled_width_polygons"
        ].loc[fid]
    else:
        raise ValueError("Both lane_fid and lane_connector_fid are NaN!")

    left_pts = np.stack(
        boundary_df.loc[str(lane_info["left_boundary_fid"])]["geometry"].xy, axis=-1
    )
    right_pts = np.stack(
        boundary_df.loc[str(lane_info["right_boundary_fid"])]["geometry"].xy, axis=-1
    )

    # Ensuring that left and right have the same number of points.
    # This is necessary, not for data storage but for later rasterization.
    if left_pts.shape[0] < right_pts.shape[0]:
        left_pts = map_utils.interpolate(left_pts, right_pts.shape[0])
    elif right_pts.shape[0] < left_pts.shape[0]:
        right_pts = map_utils.interpolate(right_pts, left_pts.shape[0])

    return (
        str(fid),
        lane_midline,
        left_pts,
        right_pts,
    )


def extract_area(nuplan_map: NuPlanMap, area_record) -> np.ndarray:
    return np.stack(area_record["geometry"].exterior.xy, axis=-1)


def extract_vectorized(nuplan_map: NuPlanMap) -> VectorizedMap:
    vec_map = VectorizedMap()

    # Setting the map bounds.
    # NOTE: min_pt is especially important here since the world coordinates of nuPlan
    # are quite large in magnitude. We make them relative to the bottom-left by
    # subtracting all positions by min_pt and registering that offset as part of
    # the map_from_world (and related) transforms later.
    min_pt = np.min(
        [
            layer_df["geometry"].total_bounds[:2]
            for layer_df in nuplan_map._vector_map.values()
        ],
        axis=0,
    )
    max_pt = np.max(
        [
            layer_df["geometry"].total_bounds[2:]
            for layer_df in nuplan_map._vector_map.values()
        ],
        axis=0,
    )

    vec_map.max_pt.x, vec_map.max_pt.y, vec_map.max_pt.z = (
        max_pt[0],
        max_pt[1],
        0.0,
    )
    vec_map.min_pt.x, vec_map.min_pt.y, vec_map.min_pt.z = (
        min_pt[0],
        min_pt[1],
        0.0,
    )

    # We will be subtracting this point from all subsequent coordinates
    # (otherwise some values are too big in millimeters to fit within sint32...).
    vec_map.bottom_left_coords.x, vec_map.bottom_left_coords.y = min_pt

    overall_pbar = tqdm(
        total=len(nuplan_map._vector_map["baseline_paths"])
        + len(nuplan_map._vector_map["drivable_area"])
        + len(nuplan_map._vector_map["crosswalks"])
        + len(nuplan_map._vector_map["walkways"]),
        desc=f"Getting {nuplan_map.map_name} Elements",
        position=1,
        leave=False,
    )

    for _, lane_info in nuplan_map._vector_map["baseline_paths"].iterrows():
        fid, center_pts, left_pts, right_pts = extract_lane_and_edges(
            nuplan_map, lane_info
        )

        lane_record_token: str = fid

        # Adding the element to the map.
        new_element: MapElement = vec_map.elements.add()
        new_element.id = lane_record_token.encode()

        new_lane: RoadLane = new_element.road_lane
        map_utils.populate_lane_polylines(
            new_lane, center_pts - min_pt, left_pts - min_pt, right_pts - min_pt
        )

        # new_lane.adjacent_lanes_left.append(
        #     l5_lane.adjacent_lane_change_left.id
        # )
        # new_lane.adjacent_lanes_right.append(
        #     l5_lane.adjacent_lane_change_right.id
        # )

        overall_pbar.update()

    for fid, polygon_info in nuplan_map._vector_map["drivable_area"].iterrows():
        polygon_pts = extract_area(nuplan_map, polygon_info)

        # Adding the element to the map.
        new_element: MapElement = vec_map.elements.add()
        new_element.id = fid.encode()

        new_area: RoadArea = new_element.road_area
        map_utils.populate_polygon(new_area.exterior_polygon, polygon_pts - min_pt)

        for hole in polygon_info["geometry"].interiors:
            polygon_pts = extract_area(nuplan_map, hole)
            new_hole: Polyline = new_area.interior_holes.add()
            map_utils.populate_polygon(new_hole, polygon_pts - min_pt)

        overall_pbar.update()

    for fid, ped_area_record in nuplan_map._vector_map["crosswalks"].iterrows():
        polygon_pts = extract_area(nuplan_map, ped_area_record)

        # Adding the element to the map.
        new_element: MapElement = vec_map.elements.add()
        new_element.id = fid.encode()

        new_crosswalk: PedCrosswalk = new_element.ped_crosswalk
        map_utils.populate_polygon(new_crosswalk.polygon, polygon_pts - min_pt)

        overall_pbar.update()

    for fid, ped_area_record in nuplan_map._vector_map["walkways"].iterrows():
        polygon_pts = extract_area(nuplan_map, ped_area_record)

        # Adding the element to the map.
        new_element: MapElement = vec_map.elements.add()
        new_element.id = fid.encode()

        new_walkway: PedWalkway = new_element.ped_walkway
        map_utils.populate_polygon(new_walkway.polygon, polygon_pts - min_pt)

        overall_pbar.update()

    overall_pbar.close()

    # Lane connectivity
    lane_connectivity_exit_dict = defaultdict(list)
    lane_connectivity_entry_dict = defaultdict(list)
    for lane_connector_fid, lane_connector in tqdm(
        nuplan_map._vector_map["lane_connectors"].iterrows(),
        desc="Getting Lane Connectivity",
        total=len(nuplan_map._vector_map["lane_connectors"]),
        position=1,
        leave=False,
    ):
        lane_connectivity_exit_dict[str(lane_connector["exit_lane_fid"])].append(
            lane_connector_fid
        )
        lane_connectivity_entry_dict[lane_connector_fid].append(
            str(lane_connector["exit_lane_fid"])
        )

        lane_connectivity_exit_dict[lane_connector_fid].append(
            str(lane_connector["entry_lane_fid"])
        )
        lane_connectivity_entry_dict[str(lane_connector["entry_lane_fid"])].append(
            lane_connector_fid
        )

    map_elem: MapElement
    for map_elem in tqdm(
        vec_map.elements,
        desc="Storing Lane Connectivity",
        position=1,
        leave=False,
    ):
        if map_elem.HasField("road_lane"):
            map_elem.road_lane.entry_lanes.extend(
                lane_id.encode()
                for lane_id in lane_connectivity_entry_dict[map_elem.id.decode()]
            )
            map_elem.road_lane.exit_lanes.extend(
                lane_id.encode()
                for lane_id in lane_connectivity_exit_dict[map_elem.id.decode()]
            )

    return vec_map

import socket
import warnings
from collections import defaultdict
from contextlib import closing
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import seaborn as sns
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.document import Document
from bokeh.layouts import column, row
from bokeh.models import (
    BooleanFilter,
    Button,
    CDSView,
    ColumnDataSource,
    HoverTool,
    Legend,
    LegendItem,
    Slider,
)
from bokeh.plotting import figure
from bokeh.server.server import Server
from shapely.geometry import LineString, Polygon
from tornado.ioloop import IOLoop

from trajdata.data_structures.agent import AgentType
from trajdata.data_structures.batch import AgentBatch
from trajdata.data_structures.state import StateArray, StateTensor
from trajdata.maps.map_api import MapAPI
from trajdata.maps.vec_map import VectorMap
from trajdata.maps.vec_map_elements import (
    MapElementType,
    PedCrosswalk,
    PedWalkway,
    RoadArea,
    RoadLane,
)
from trajdata.utils.arr_utils import transform_coords_2d_np


class InteractiveAnimation:
    def __init__(
        self,
        main_func: Callable[[Document, IOLoop], None],
        port: Optional[int] = None,
        **kwargs,
    ) -> None:
        self.main_func = main_func
        self.port = port
        self.kwargs = kwargs

    def get_open_port(self) -> int:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("", 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]

    def show(self) -> None:
        io_loop = IOLoop()

        if self.port is None:
            self.port = self.get_open_port()

        def kill_on_tab_close(session_context):
            io_loop.stop()

        def app_init(doc: Document):
            doc.on_session_destroyed(kill_on_tab_close)
            self.main_func(doc=doc, io_loop=io_loop, **self.kwargs)
            return doc

        server = Server(
            {"/": Application(FunctionHandler(app_init))},
            io_loop=io_loop,
            port=self.port,
            check_unused_sessions_milliseconds=500,
            unused_session_lifetime_milliseconds=500,
        )
        server.start()

        # print(f"Opening Bokeh application on http://localhost:{self.port}/")
        server.io_loop.add_callback(server.show, "/")
        server.io_loop.start()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            server.io_loop.close()


def agent_type_to_str(agent_type: AgentType) -> str:
    return str(AgentType(agent_type))[len("AgentType.") :]


def get_agent_type_color(agent_type: AgentType) -> str:
    palette = sns.color_palette("husl", 4).as_hex()
    if agent_type == AgentType.VEHICLE:
        return palette[0]
    elif agent_type == AgentType.PEDESTRIAN:
        return palette[1]
    elif agent_type == AgentType.BICYCLE:
        return palette[2]
    elif agent_type == AgentType.MOTORCYCLE:
        return palette[3]
    else:
        return "#A9A9A9"


def get_map_patch_color(map_elem_type: MapElementType) -> str:
    if map_elem_type == MapElementType.ROAD_AREA:
        return "lightgray"
    elif map_elem_type == MapElementType.ROAD_LANE:
        return "red"
    elif map_elem_type == MapElementType.PED_CROSSWALK:
        return "blue"
    elif map_elem_type == MapElementType.PED_WALKWAY:
        return "green"
    else:
        raise ValueError()


def compute_agent_rect_coords(
    agent_type: int, hs: np.ndarray, lengths: np.ndarray, widths: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    raw_rect_coords = np.stack(
        (
            np.stack((-lengths / 2, -widths / 2), axis=-1),
            np.stack((-lengths / 2, widths / 2), axis=-1),
            np.stack((lengths / 2, widths / 2), axis=-1),
            np.stack((lengths / 2, -widths / 2), axis=-1),
        ),
        axis=-2,
    )

    agent_rect_coords = transform_coords_2d_np(
        raw_rect_coords,
        angle=hs[:, None].repeat(raw_rect_coords.shape[-2], axis=-1),
    )

    size = 1.0
    if agent_type == AgentType.PEDESTRIAN or agent_type == AgentType.BICYCLE:
        size = 0.25

    raw_tri_coords = size * np.array(
        [
            [
                [0, np.sqrt(3) / 3],
                [-1 / 2, -np.sqrt(3) / 6],
                [1 / 2, -np.sqrt(3) / 6],
            ]
        ]
    ).repeat(hs.shape[0], axis=0)

    dir_patch_coords = transform_coords_2d_np(
        raw_tri_coords,
        angle=hs[:, None].repeat(raw_tri_coords.shape[-2], axis=-1) - np.pi / 2,
    )

    return agent_rect_coords, dir_patch_coords


def extract_full_agent_data_df(batch: AgentBatch, batch_idx: int) -> ColumnDataSource:
    main_data_dict = defaultdict(list)

    # Historical information
    ## Agent
    H = batch.agent_hist_len[batch_idx].item()
    agent_type = batch.agent_type[batch_idx].item()
    agent_extent: np.ndarray = batch.agent_hist_extent[batch_idx, -H:].cpu().numpy()
    agent_hist_np: StateArray = batch.agent_hist[batch_idx, -H:].cpu().numpy()

    speed_mps = np.linalg.norm(agent_hist_np.velocity, axis=1)

    xs = agent_hist_np.get_attr("x")
    ys = agent_hist_np.get_attr("y")
    hs = agent_hist_np.get_attr("h")

    lengths = agent_extent[:, 0]
    widths = agent_extent[:, 1]

    agent_rect_coords, dir_patch_coords = compute_agent_rect_coords(
        agent_type, hs, lengths, widths
    )

    main_data_dict["id"].extend([0] * H)
    main_data_dict["t"].extend(range(-H + 1, 1))
    main_data_dict["x"].extend(xs)
    main_data_dict["y"].extend(ys)
    main_data_dict["h"].extend(hs)
    main_data_dict["rect_xs"].extend(agent_rect_coords[..., 0] + xs[:, None])
    main_data_dict["rect_ys"].extend(agent_rect_coords[..., 1] + ys[:, None])
    main_data_dict["dir_patch_xs"].extend(dir_patch_coords[..., 0] + xs[:, None])
    main_data_dict["dir_patch_ys"].extend(dir_patch_coords[..., 1] + ys[:, None])
    main_data_dict["speed_mps"].extend(speed_mps)
    main_data_dict["speed_kph"].extend(speed_mps * 3.6)
    main_data_dict["type"].extend([agent_type_to_str(agent_type)] * H)
    main_data_dict["length"].extend(lengths)
    main_data_dict["width"].extend(widths)
    main_data_dict["pred_agent"].extend([True] * H)
    main_data_dict["color"].extend([get_agent_type_color(agent_type)] * H)

    ## Neighbors
    num_neighbors: int = batch.num_neigh[batch_idx].item()

    for n_neigh in range(num_neighbors):
        H = batch.neigh_hist_len[batch_idx, n_neigh].item()
        agent_type = batch.neigh_types[batch_idx, n_neigh].item()
        agent_extent: np.ndarray = (
            batch.neigh_hist_extents[batch_idx, n_neigh, -H:].cpu().numpy()
        )
        agent_hist_np: StateArray = (
            batch.neigh_hist[batch_idx, n_neigh, -H:].cpu().numpy()
        )

        speed_mps = np.linalg.norm(agent_hist_np.velocity, axis=1)

        xs = agent_hist_np.get_attr("x")
        ys = agent_hist_np.get_attr("y")
        hs = agent_hist_np.get_attr("h")

        lengths = agent_extent[:, 0]
        widths = agent_extent[:, 1]

        agent_rect_coords, dir_patch_coords = compute_agent_rect_coords(
            agent_type, hs, lengths, widths
        )

        main_data_dict["id"].extend([n_neigh + 1] * H)
        main_data_dict["t"].extend(range(-H + 1, 1))
        main_data_dict["x"].extend(xs)
        main_data_dict["y"].extend(ys)
        main_data_dict["h"].extend(hs)
        main_data_dict["rect_xs"].extend(agent_rect_coords[..., 0] + xs[:, None])
        main_data_dict["rect_ys"].extend(agent_rect_coords[..., 1] + ys[:, None])
        main_data_dict["dir_patch_xs"].extend(dir_patch_coords[..., 0] + xs[:, None])
        main_data_dict["dir_patch_ys"].extend(dir_patch_coords[..., 1] + ys[:, None])
        main_data_dict["speed_mps"].extend(speed_mps)
        main_data_dict["speed_kph"].extend(speed_mps * 3.6)
        main_data_dict["type"].extend([agent_type_to_str(agent_type)] * H)
        main_data_dict["length"].extend(lengths)
        main_data_dict["width"].extend(widths)
        main_data_dict["pred_agent"].extend([False] * H)
        main_data_dict["color"].extend([get_agent_type_color(agent_type)] * H)

    # Future information
    ## Agent
    T = batch.agent_fut_len[batch_idx].item()
    agent_type = batch.agent_type[batch_idx].item()
    agent_extent: np.ndarray = batch.agent_fut_extent[batch_idx, :T].cpu().numpy()
    agent_fut_np: StateArray = batch.agent_fut[batch_idx, :T].cpu().numpy()

    speed_mps = np.linalg.norm(agent_fut_np.velocity, axis=1)

    xs = agent_fut_np.get_attr("x")
    ys = agent_fut_np.get_attr("y")
    hs = agent_fut_np.get_attr("h")

    lengths = agent_extent[:, 0]
    widths = agent_extent[:, 1]

    agent_rect_coords, dir_patch_coords = compute_agent_rect_coords(
        agent_type, hs, lengths, widths
    )

    main_data_dict["id"].extend([0] * T)
    main_data_dict["t"].extend(range(1, T + 1))
    main_data_dict["x"].extend(xs)
    main_data_dict["y"].extend(ys)
    main_data_dict["h"].extend(hs)
    main_data_dict["rect_xs"].extend(agent_rect_coords[..., 0] + xs[:, None])
    main_data_dict["rect_ys"].extend(agent_rect_coords[..., 1] + ys[:, None])
    main_data_dict["dir_patch_xs"].extend(dir_patch_coords[..., 0] + xs[:, None])
    main_data_dict["dir_patch_ys"].extend(dir_patch_coords[..., 1] + ys[:, None])
    main_data_dict["speed_mps"].extend(speed_mps)
    main_data_dict["speed_kph"].extend(speed_mps * 3.6)
    main_data_dict["type"].extend([agent_type_to_str(agent_type)] * T)
    main_data_dict["length"].extend(lengths)
    main_data_dict["width"].extend(widths)
    main_data_dict["pred_agent"].extend([True] * T)
    main_data_dict["color"].extend([get_agent_type_color(agent_type)] * T)

    ## Neighbors
    num_neighbors: int = batch.num_neigh[batch_idx].item()

    for n_neigh in range(num_neighbors):
        T = batch.neigh_fut_len[batch_idx, n_neigh].item()
        agent_type = batch.neigh_types[batch_idx, n_neigh].item()
        agent_extent: np.ndarray = (
            batch.neigh_fut_extents[batch_idx, n_neigh, :T].cpu().numpy()
        )
        agent_fut_np: StateArray = batch.neigh_fut[batch_idx, n_neigh, :T].cpu().numpy()

        speed_mps = np.linalg.norm(agent_fut_np.velocity, axis=1)

        xs = agent_fut_np.get_attr("x")
        ys = agent_fut_np.get_attr("y")
        hs = agent_fut_np.get_attr("h")

        lengths = agent_extent[:, 0]
        widths = agent_extent[:, 1]

        agent_rect_coords, dir_patch_coords = compute_agent_rect_coords(
            agent_type, hs, lengths, widths
        )

        main_data_dict["id"].extend([n_neigh + 1] * T)
        main_data_dict["t"].extend(range(1, T + 1))
        main_data_dict["x"].extend(xs)
        main_data_dict["y"].extend(ys)
        main_data_dict["h"].extend(hs)
        main_data_dict["rect_xs"].extend(agent_rect_coords[..., 0] + xs[:, None])
        main_data_dict["rect_ys"].extend(agent_rect_coords[..., 1] + ys[:, None])
        main_data_dict["dir_patch_xs"].extend(dir_patch_coords[..., 0] + xs[:, None])
        main_data_dict["dir_patch_ys"].extend(dir_patch_coords[..., 1] + ys[:, None])
        main_data_dict["speed_mps"].extend(speed_mps)
        main_data_dict["speed_kph"].extend(speed_mps * 3.6)
        main_data_dict["type"].extend([agent_type_to_str(agent_type)] * T)
        main_data_dict["length"].extend(lengths)
        main_data_dict["width"].extend(widths)
        main_data_dict["pred_agent"].extend([False] * T)
        main_data_dict["color"].extend([get_agent_type_color(agent_type)] * T)

    return pd.DataFrame(main_data_dict)


def convert_to_gpd(vec_map: VectorMap):
    geo_data = defaultdict(list)
    for elem in vec_map.iter_elems():
        geo_data["id"].append(elem.id)
        geo_data["type"].append(elem.elem_type)
        if isinstance(elem, RoadLane):
            geo_data["geometry"].append(LineString(elem.center.xyz))
        elif isinstance(elem, PedCrosswalk) or isinstance(elem, PedWalkway):
            geo_data["geometry"].append(Polygon(shell=elem.polygon.xyz))
        elif isinstance(elem, RoadArea):
            geo_data["geometry"].append(
                Polygon(
                    shell=elem.exterior_polygon.xyz,
                    holes=[hole.xyz for hole in elem.interior_holes],
                )
            )

    return gpd.GeoDataFrame(geo_data)


def get_map_cds(
    center_pt: StateTensor, vec_map: VectorMap, radius: float = 50.0, **kwargs
) -> Tuple[
    ColumnDataSource,
    ColumnDataSource,
    ColumnDataSource,
    ColumnDataSource,
    ColumnDataSource,
]:
    center_pt_np: StateArray = center_pt.cpu().numpy()
    x, y = center_pt_np.position

    road_lane_data = defaultdict(list)
    lane_center_data = defaultdict(list)
    ped_crosswalk_data = defaultdict(list)
    ped_walkway_data = defaultdict(list)
    road_area_data = defaultdict(list)

    map_gpd = convert_to_gpd(vec_map)
    elems_gdf: gpd.GeoDataFrame = map_gpd.cx[
        x - radius : x + radius, y - radius : y + radius
    ]

    for row_idx, row in elems_gdf.iterrows():
        if row["type"] == MapElementType.PED_CROSSWALK:
            xy = np.stack(row["geometry"].exterior.xy, axis=1)
            transformed_xy: np.ndarray = transform_coords_2d_np(
                xy - center_pt_np.position,
                angle=-center_pt_np.heading,
            )
            ped_crosswalk_data["xs"].append(transformed_xy[..., 0])
            ped_crosswalk_data["ys"].append(transformed_xy[..., 1])
        if row["type"] == MapElementType.PED_WALKWAY:
            xy = np.stack(row["geometry"].exterior.xy, axis=1)
            transformed_xy: np.ndarray = transform_coords_2d_np(
                xy - center_pt_np.position,
                angle=-center_pt_np.heading,
            )
            ped_walkway_data["xs"].append(transformed_xy[..., 0])
            ped_walkway_data["ys"].append(transformed_xy[..., 1])
        elif row["type"] == MapElementType.ROAD_LANE:
            xy = np.stack(row["geometry"].xy, axis=1)
            transformed_xy: np.ndarray = transform_coords_2d_np(
                xy - center_pt_np.position,
                angle=-center_pt_np.heading,
            )

            lane_center_data["xs"].append(transformed_xy[..., 0])
            lane_center_data["ys"].append(transformed_xy[..., 1])
            lane_obj: RoadLane = vec_map.elements[MapElementType.ROAD_LANE][row["id"]]
            if lane_obj.left_edge is not None and lane_obj.right_edge is not None:
                left_xy = lane_obj.left_edge.xy
                right_xy = lane_obj.right_edge.xy[::-1]
                patch_xy = np.concatenate((left_xy, right_xy), axis=0)

                transformed_xy: np.ndarray = transform_coords_2d_np(
                    patch_xy - center_pt_np.position,
                    angle=-center_pt_np.heading,
                )

                road_lane_data["xs"].append(transformed_xy[..., 0])
                road_lane_data["ys"].append(transformed_xy[..., 1])
        elif row["type"] == MapElementType.ROAD_AREA:
            xy = np.stack(row["geometry"].exterior.xy, axis=1)
            transformed_xy: np.ndarray = transform_coords_2d_np(
                xy - center_pt_np.position,
                angle=-center_pt_np.heading,
            )

            holes_xy: List[np.ndarray] = [
                transform_coords_2d_np(
                    np.stack(interior.xy, axis=1) - center_pt_np.position,
                    angle=-center_pt_np.heading,
                )
                for interior in row["geometry"].interiors
            ]

            road_area_data["xs"].append(
                [[transformed_xy[..., 0]] + [hole[..., 0] for hole in holes_xy]]
            )
            road_area_data["ys"].append(
                [[transformed_xy[..., 1]] + [hole[..., 1] for hole in holes_xy]]
            )
    return (
        ColumnDataSource(data=lane_center_data),
        ColumnDataSource(data=road_lane_data),
        ColumnDataSource(data=ped_crosswalk_data),
        ColumnDataSource(data=ped_walkway_data),
        ColumnDataSource(data=road_area_data),
    )


def plot_full_agent_batch_interactive(
    doc: Document, io_loop: IOLoop, batch: AgentBatch, batch_idx: int, cache_path: Path
) -> None:
    agent_data_df = extract_full_agent_data_df(batch, batch_idx)

    # Figure creation and a few initial settings.
    x_min = agent_data_df["x"].min()
    x_max = agent_data_df["x"].max()
    x_range = x_max - x_min

    y_min = agent_data_df["y"].min()
    y_max = agent_data_df["y"].max()
    y_range = y_max - y_min

    buffer = 10
    if x_range > y_range:
        half_range_diff = (x_range - y_range) / 2
        kwargs = {
            "x_range": (x_min - buffer, x_max + buffer),
            "y_range": (
                y_min - half_range_diff - buffer,
                y_max + half_range_diff + buffer,
            ),
        }
    else:
        half_range_diff = (y_range - x_range) / 2
        kwargs = {
            "y_range": (y_min - buffer, y_max + buffer),
            "x_range": (
                x_min - half_range_diff - buffer,
                x_max + half_range_diff + buffer,
            ),
        }

    fig = figure(match_aspect=True, width=800, sizing_mode="scale_width", **kwargs)

    agent_name: str = batch.agent_name[batch_idx]
    agent_type: AgentType = AgentType(batch.agent_type[batch_idx].item())
    current_state = batch.curr_agent_state[batch_idx].numpy()
    fig.title = f"{str(agent_type)}/{agent_name} at x={current_state[0]:.2f}, y={current_state[1]:.2f}, h={current_state[-1]:.2f}"

    # No gridlines.
    fig.grid.visible = False

    # Set autohide to true to only show the toolbar when mouse is over plot.
    fig.toolbar.autohide = True

    # Setting the match_aspect property of bokeh's default BoxZoomTool.
    fig.tools[2].match_aspect = True

    agent_cds = ColumnDataSource(agent_data_df)
    curr_time_view = CDSView(
        source=agent_cds, filters=[BooleanFilter(agent_cds.data["t"] == 0)]
    )

    full_H = batch.agent_hist[batch_idx].shape[0]
    full_T = batch.agent_fut[batch_idx].shape[0]

    def create_multi_line_data_df(agents_df: pd.DataFrame) -> Dict[str, Any]:
        lines_data = defaultdict(list)
        for agent_id, agent_df in agents_df.groupby(by="id"):
            xs, ys, color = (
                agent_df.x.to_numpy(),
                agent_df.y.to_numpy(),
                agent_df.color.iat[0],
            )
            if agent_id > 0:
                pad_before = (
                    full_H - batch.neigh_hist_len[batch_idx, agent_id - 1].item()
                )
                pad_after = full_T - batch.neigh_fut_len[batch_idx, agent_id - 1].item()
                xs = np.pad(xs, (pad_before, pad_after), constant_values=np.nan)
                ys = np.pad(ys, (pad_before, pad_after), constant_values=np.nan)

            lines_data["xs"].append(xs)
            lines_data["ys"].append(ys)
            lines_data["color"].append(color)

        return lines_data

    def get_sliced_multi_line_data_df(
        multi_line_df: Dict[str, Any], slice_obj, check_idx: int
    ) -> Dict[str, Any]:
        lines_data = defaultdict(list)
        for i in range(len(multi_line_df["xs"])):
            sliced_xs = multi_line_df["xs"][i][slice_obj]
            sliced_ys = multi_line_df["ys"][i][slice_obj]
            if (
                sliced_xs.shape[0] > 0
                and sliced_ys.shape[0] > 0
                and np.isfinite(sliced_xs[check_idx])
                and np.isfinite(sliced_ys[check_idx])
            ):
                lines_data["xs"].append(sliced_xs)
                lines_data["ys"].append(sliced_ys)
                lines_data["color"].append(multi_line_df["color"][i])

        return lines_data

    history_line_data_df = create_multi_line_data_df(agent_data_df)
    history_lines_cds = ColumnDataSource(
        get_sliced_multi_line_data_df(
            history_line_data_df, slice(None, full_H), check_idx=-1
        )
    )
    future_line_data_df = history_line_data_df.copy()
    future_lines_cds = ColumnDataSource(
        get_sliced_multi_line_data_df(
            future_line_data_df, slice(full_H, None), check_idx=0
        )
    )

    dt: float = batch.dt[batch_idx].item()
    scene_ts: int = batch.scene_ts[batch_idx].item()

    if batch.map_names is not None:
        mapAPI = MapAPI(cache_path)

        (
            lane_center_cds,
            road_lane_cds,
            ped_crosswalk_cds,
            ped_walkway_cds,
            road_area_cds,
        ) = get_map_cds(
            batch.curr_agent_state[batch_idx],
            mapAPI.get_map(
                batch.map_names[batch_idx],
                incl_road_lanes=True,
                incl_road_areas=True,
                incl_ped_crosswalks=True,
                incl_ped_walkways=True,
            ),
            alpha=1.0,
        )

        road_areas = fig.multi_polygons(
            source=road_area_cds,
            line_color="black",
            fill_alpha=0.1,
            fill_color=get_map_patch_color(MapElementType.ROAD_AREA),
        )

        road_lanes = fig.patches(
            source=road_lane_cds,
            line_color="black",
            fill_alpha=0.1,
            fill_color=get_map_patch_color(MapElementType.ROAD_LANE),
        )

        ped_crosswalks = fig.patches(
            source=ped_crosswalk_cds,
            line_color="black",
            fill_alpha=0.1,
            fill_color=get_map_patch_color(MapElementType.PED_CROSSWALK),
        )

        ped_walkways = fig.patches(
            source=ped_walkway_cds,
            line_color="black",
            fill_alpha=0.1,
            fill_color=get_map_patch_color(MapElementType.PED_WALKWAY),
        )

        lane_centers = fig.multi_line(
            source=lane_center_cds,
            line_color="gray",
            line_alpha=0.5,
        )

    history_lines = fig.multi_line(
        xs="xs",
        ys="ys",
        line_color="color",
        line_dash="dashed",
        source=history_lines_cds,
    )

    future_lines = fig.multi_line(
        xs="xs", ys="ys", line_color="color", line_dash="solid", source=future_lines_cds
    )

    agent_rects = fig.patches(
        xs="rect_xs",
        ys="rect_ys",
        fill_color="color",
        line_color="black",
        # fill_alpha=0.7,
        source=agent_cds,
        view=curr_time_view,
    )

    agent_dir_patches = fig.patches(
        xs="dir_patch_xs",
        ys="dir_patch_ys",
        fill_color="color",
        line_color="black",
        # fill_alpha=0.7,
        source=agent_cds,
        view=curr_time_view,
    )

    time_slider = Slider(
        start=agent_cds.data["t"].min(),
        end=agent_cds.data["t"].max(),
        step=1,
        value=0,
        title=f"Current Timestep (scene timestep {scene_ts})",
    )

    def time_callback(attr, old, new) -> None:
        curr_time_view.filters = [BooleanFilter(agent_cds.data["t"] == new)]
        history_lines_cds.data = get_sliced_multi_line_data_df(
            history_line_data_df, slice(None, new + full_H), check_idx=-1
        )
        future_lines_cds.data = get_sliced_multi_line_data_df(
            future_line_data_df, slice(new + full_H, None), check_idx=0
        )

        if new == 0:
            time_slider.title = f"Current Timestep (scene timestep {scene_ts})"
        else:
            n_steps = abs(new)
            time_slider.title = f"{n_steps} timesteps ({n_steps * dt:.2f} s) into the {'future' if new > 0 else 'past'}"

    time_slider.on_change("value", time_callback)

    fig.add_tools(
        HoverTool(
            tooltips=[
                ("Class", "@type"),
                ("Position", "(@x, @y) m"),
                ("Speed", "@speed_mps m/s (@speed_kph km/h)"),
            ],
            renderers=[agent_rects],
        )
    )

    def button_callback():
        # Stop the server.
        io_loop.stop()

    exit_button = Button(label="Exit", button_type="danger", width=60)
    exit_button.on_click(button_callback)

    def animate_update():
        t = time_slider.value + 1

        if t > time_slider.end:
            # If slider value + 1 is above max, reset to 0.
            t = 0

        time_slider.value = t

    play_cb_manager = [None]

    def animate():
        if play_button.label.startswith("►"):
            play_button.label = "❚❚ Pause"

            play_cb_manager[0] = doc.add_periodic_callback(
                animate_update, period_milliseconds=int(dt * 1000)
            )
        else:
            play_button.label = "► Play"
            doc.remove_periodic_callback(play_cb_manager[0])

    play_button = Button(label="► Play", width=100)
    play_button.on_click(animate)

    agent_legend_elems = [
        fig.rect(
            fill_color=get_agent_type_color(x),
            line_color="black",
            name=str(x)[len("AgentType.") :],
        )
        for x in AgentType
    ]

    map_legend_elems = [
        LegendItem(label="Lane Center", renderers=[lane_centers])
    ]

    map_area_legend_elems = [
        LegendItem(label="Road Area", renderers=[road_areas]),
        LegendItem(label="Road Lanes", renderers=[road_lanes]),
        LegendItem(label="Pedestrian Crosswalks", renderers=[ped_crosswalks]),
        LegendItem(label="Pedestrian Walkways", renderers=[ped_walkways]),
    ]

    hist_future_legend_elems = [
        LegendItem(
            label="Past Motion",
            renderers=[
                fig.multi_line(line_color="black", line_dash="dashed"),
                history_lines,
            ],
        ),
        LegendItem(
            label="Future Motion",
            renderers=[
                fig.multi_line(line_color="black", line_dash="solid"),
                future_lines,
            ],
        ),
    ]

    legend = Legend(
        items=[
            LegendItem(label=legend_item.name, renderers=[legend_item])
            for legend_item in agent_legend_elems
        ]
        + hist_future_legend_elems
        + map_legend_elems
        + map_area_legend_elems,
        click_policy="hide",
    )
    fig.add_layout(legend, "right")

    video_button = Button(
        label="Render Animation",
        width=120,
    )

    def save_animation(file_path: Path) -> None:
        video_button.disabled = True
        video_button.label = "Implement me!"

    video_button.on_click(partial(save_animation, file_path=Path("video.avi")))

    doc.add_root(column(fig, row(play_button, time_slider, exit_button), video_button))

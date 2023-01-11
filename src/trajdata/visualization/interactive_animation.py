import sys
from collections import defaultdict
from decimal import Decimal
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from bokeh.document import Document
from bokeh.layouts import column, row
from bokeh.models import (
    BooleanFilter,
    Button,
    CDSView,
    ColumnDataSource,
    HoverTool,
    Slider,
)
from bokeh.plotting import figure
from bokeh.server.server import Server

from trajdata.data_structures.agent import AgentType
from trajdata.data_structures.batch import AgentBatch
from trajdata.data_structures.state import StateArray, StateTensor
from trajdata.maps.map_api import MapAPI
from trajdata.maps.vec_map import VectorMap
from trajdata.maps.vec_map_elements import RoadLane
from trajdata.utils.arr_utils import transform_coords_2d_np


class InteractiveAnimation:
    def __init__(self, main_func: Callable[[Document], None], **kwargs) -> None:
        self.main_func = main_func
        self.kwargs = kwargs

    def show(self) -> None:
        server = Server({"/": partial(self.main_func, **self.kwargs)})
        server.start()

        print("Opening Bokeh application on http://localhost:5006/")
        server.io_loop.add_callback(server.show, "/")
        server.io_loop.start()


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

    return palette[3]


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
    if agent_type == AgentType.PEDESTRIAN:
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


def get_map_cds(
    center_pt: StateTensor, vec_map: VectorMap, radius: float = 50.0, **kwargs
) -> ColumnDataSource:
    center_pt_np: StateArray = center_pt.cpu().numpy()

    lines_data = {
        "xs": [],
        "ys": [],
        "line_dash": [],
        "line_color": [],
        "line_alpha": [],
    }

    lanes = vec_map.get_lanes_within(center_pt_np.position3d, radius)
    lane: RoadLane
    for lane in lanes:
        if lane.left_edge is not None:
            lane_edge_pts: np.ndarray = transform_coords_2d_np(
                lane.left_edge.xy - center_pt_np.position,
                angle=-center_pt_np.heading,
            )

            lines_data["xs"].append(lane_edge_pts[:, 0])
            lines_data["ys"].append(lane_edge_pts[:, 1])
            lines_data["line_dash"].append("solid")
            lines_data["line_color"].append("red")
            lines_data["line_alpha"].append(0.7)

        if lane.right_edge is not None:
            lane_edge_pts: np.ndarray = transform_coords_2d_np(
                lane.right_edge.xy - center_pt_np.position,
                angle=-center_pt_np.heading,
            )
            lines_data["xs"].append(lane_edge_pts[:, 0])
            lines_data["ys"].append(lane_edge_pts[:, 1])
            lines_data["line_dash"].append("solid")
            lines_data["line_color"].append("red")
            lines_data["line_alpha"].append(0.7)

        lane_center_pts: np.ndarray = transform_coords_2d_np(
            lane.center.xy - center_pt_np.position, angle=-center_pt_np.heading
        )
        lines_data["xs"].append(lane_center_pts[:, 0])
        lines_data["ys"].append(lane_center_pts[:, 1])
        lines_data["line_dash"].append("solid")
        lines_data["line_color"].append("gray")
        lines_data["line_alpha"].append(0.5)

    return ColumnDataSource(data=lines_data)


def plot_full_agent_batch_interactive(
    doc: Document, batch: AgentBatch, batch_idx: int, cache_path: Path
) -> None:
    agent_data_df = extract_full_agent_data_df(batch, batch_idx)
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

    fig = figure(match_aspect=True, **kwargs)

    # No gridlines.
    fig.grid.visible = False

    # Set autohide to true to only show the toolbar when mouse is over plot.
    fig.toolbar.autohide = True

    # Setting the match_aspect property of bokeh's default BoxZoomTool.
    fig.tools[2].match_aspect = True

    if batch.map_names is not None:
        mapAPI = MapAPI(cache_path)

        map_cds = get_map_cds(
            batch.curr_agent_state[batch_idx],
            mapAPI.get_map(batch.map_names[batch_idx]),
            alpha=1.0,
        )

        fig.multi_line(
            source=map_cds,
            # This is to ensure that the columns given in the
            # ColumnDataSource are respected (e.g., "line_color").
            **{x: x for x in map_cds.column_names},
        )

    fig.multi_line(
        xs="xs",
        ys="ys",
        line_color="color",
        line_dash="dashed",
        source=history_lines_cds,
    )

    fig.multi_line(
        xs="xs", ys="ys", line_color="color", line_dash="solid", source=future_lines_cds
    )

    agent_rects = fig.patches(
        xs="rect_xs",
        ys="rect_ys",
        fill_color="color",
        line_color="black",
        fill_alpha=0.7,
        source=agent_cds,
        view=curr_time_view,
    )

    fig.patches(
        xs="dir_patch_xs",
        ys="dir_patch_ys",
        fill_color="color",
        line_color="black",
        fill_alpha=0.7,
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
        sys.exit()

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

    doc.add_root(column(fig, row(play_button, time_slider, exit_button)))

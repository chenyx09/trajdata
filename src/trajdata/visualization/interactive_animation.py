import socket
import warnings
from collections import defaultdict
from contextlib import closing
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
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
from tornado.ioloop import IOLoop

from trajdata.data_structures.agent import AgentType
from trajdata.data_structures.batch import AgentBatch
from trajdata.maps.map_api import MapAPI
from trajdata.maps.vec_map_elements import MapElementType
from trajdata.utils import vis_utils


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


def plot_full_agent_batch_interactive(
    doc: Document, io_loop: IOLoop, batch: AgentBatch, batch_idx: int, cache_path: Path
) -> None:
    agent_data_df = vis_utils.extract_full_agent_data_df(batch, batch_idx)

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

    # Map plotting.
    if batch.map_names is not None:
        mapAPI = MapAPI(cache_path)

        vec_map = mapAPI.get_map(
            batch.map_names[batch_idx],
            incl_road_lanes=True,
            incl_road_areas=True,
            incl_ped_crosswalks=True,
            incl_ped_walkways=True,
        )

        (
            road_areas,
            road_lanes,
            ped_crosswalks,
            ped_walkways,
            lane_centers,
        ) = vis_utils.draw_map_elems(fig, vec_map, batch.curr_agent_state[batch_idx])

    # Preparing agent information for fast slicing with the time_slider.
    agent_cds = ColumnDataSource(agent_data_df)
    curr_time_view = CDSView(
        source=agent_cds, filters=[BooleanFilter(agent_cds.data["t"] == 0)]
    )

    full_H = batch.agent_hist[batch_idx].shape[0]
    full_T = batch.agent_fut[batch_idx].shape[0]

    def create_multi_line_data(agents_df: pd.DataFrame) -> Dict[str, List]:
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

    def get_sliced_multi_line_data(
        multi_line_df: Dict[str, Any], slice_obj, check_idx: int
    ) -> Dict[str, List]:
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

    # Getting initial historical and future trajectory information ready for plotting.
    history_line_data_df = create_multi_line_data(agent_data_df)
    history_lines_cds = ColumnDataSource(
        get_sliced_multi_line_data(
            history_line_data_df, slice(None, full_H), check_idx=-1
        )
    )
    future_line_data_df = history_line_data_df.copy()
    future_lines_cds = ColumnDataSource(
        get_sliced_multi_line_data(
            future_line_data_df, slice(full_H, None), check_idx=0
        )
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

    # Agent rectangles/directional arrows at the current timestep.
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

    scene_ts: int = batch.scene_ts[batch_idx].item()

    # Controlling the timestep shown to users.
    time_slider = Slider(
        start=agent_cds.data["t"].min(),
        end=agent_cds.data["t"].max(),
        step=1,
        value=0,
        title=f"Current Timestep (scene timestep {scene_ts})",
    )

    dt: float = batch.dt[batch_idx].item()

    # Ensuring that information gets updated upon a cahnge in the slider value.
    def time_callback(attr, old, new) -> None:
        curr_time_view.filters = [BooleanFilter(agent_cds.data["t"] == new)]
        history_lines_cds.data = get_sliced_multi_line_data(
            history_line_data_df, slice(None, new + full_H), check_idx=-1
        )
        future_lines_cds.data = get_sliced_multi_line_data(
            future_line_data_df, slice(new + full_H, None), check_idx=0
        )

        if new == 0:
            time_slider.title = f"Current Timestep (scene timestep {scene_ts})"
        else:
            n_steps = abs(new)
            time_slider.title = f"{n_steps} timesteps ({n_steps * dt:.2f} s) into the {'future' if new > 0 else 'past'}"

    time_slider.on_change("value", time_callback)

    # Adding tooltips on mouse hover.
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

    # Writing animation callback functions so that the play/pause button animate the
    # data according to its native dt.
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

    # Creating the legend elements and connecting them to their original elements
    # (allows us to hide them on click later!)
    agent_legend_elems = [
        fig.rect(
            fill_color=vis_utils.get_agent_type_color(x),
            line_color="black",
            name=str(x)[len("AgentType.") :],
        )
        for x in AgentType
    ]

    map_legend_elems = [LegendItem(label="Lane Center", renderers=[lane_centers])]

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

    # Adding the legend to the figure.
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

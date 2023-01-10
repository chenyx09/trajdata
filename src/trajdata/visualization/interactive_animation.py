import sys
from collections import defaultdict
from decimal import Decimal
from functools import partial
from typing import Callable

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


def extract_full_agent_data_df(batch: AgentBatch, batch_idx: int) -> ColumnDataSource:
    main_data_dict = defaultdict(list)

    # Historical information
    ## Agent
    H = batch.agent_hist_len[batch_idx].item()
    agent_type = batch.agent_type[batch_idx].item()
    agent_extent: np.ndarray = batch.agent_hist_extent[batch_idx, -H:].cpu().numpy()
    agent_hist_np: StateArray = batch.agent_hist[batch_idx, -H:].cpu().numpy()

    speed_mps = np.linalg.norm(agent_hist_np.velocity, axis=1)

    main_data_dict["id"].extend([0] * H)
    main_data_dict["t"].extend(range(-H + 1, 1))
    main_data_dict["x"].extend(agent_hist_np.get_attr("x"))
    main_data_dict["y"].extend(agent_hist_np.get_attr("y"))
    main_data_dict["h"].extend(agent_hist_np.get_attr("h"))
    main_data_dict["speed_mps"].extend(speed_mps)
    main_data_dict["speed_kph"].extend(speed_mps * 3.6)
    main_data_dict["type"].extend([agent_type_to_str(agent_type)] * H)
    main_data_dict["length"].extend(agent_extent[:, 0])
    main_data_dict["width"].extend(agent_extent[:, 1])
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

        main_data_dict["id"].extend([n_neigh + 1] * H)
        main_data_dict["t"].extend(range(-H + 1, 1))
        main_data_dict["x"].extend(agent_hist_np.get_attr("x"))
        main_data_dict["y"].extend(agent_hist_np.get_attr("y"))
        main_data_dict["h"].extend(agent_hist_np.get_attr("h"))
        main_data_dict["speed_mps"].extend(speed_mps)
        main_data_dict["speed_kph"].extend(speed_mps * 3.6)
        main_data_dict["type"].extend([agent_type_to_str(agent_type)] * H)
        main_data_dict["length"].extend(agent_extent[:, 0])
        main_data_dict["width"].extend(agent_extent[:, 1])
        main_data_dict["pred_agent"].extend([False] * H)
        main_data_dict["color"].extend([get_agent_type_color(agent_type)] * H)

    # Future information
    ## Agent
    T = batch.agent_fut_len[batch_idx].item()
    agent_type = batch.agent_type[batch_idx].item()
    agent_extent: np.ndarray = batch.agent_fut_extent[batch_idx, :T].cpu().numpy()
    agent_fut_np: StateArray = batch.agent_fut[batch_idx, :T].cpu().numpy()

    speed_mps = np.linalg.norm(agent_fut_np.velocity, axis=1)

    main_data_dict["id"].extend([0] * T)
    main_data_dict["t"].extend(range(1, T + 1))
    main_data_dict["x"].extend(agent_fut_np.get_attr("x"))
    main_data_dict["y"].extend(agent_fut_np.get_attr("y"))
    main_data_dict["h"].extend(agent_fut_np.get_attr("h"))
    main_data_dict["speed_mps"].extend(speed_mps)
    main_data_dict["speed_kph"].extend(speed_mps * 3.6)
    main_data_dict["type"].extend([agent_type_to_str(agent_type)] * T)
    main_data_dict["length"].extend(agent_extent[:, 0])
    main_data_dict["width"].extend(agent_extent[:, 1])
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

        main_data_dict["id"].extend([n_neigh + 1] * T)
        main_data_dict["t"].extend(range(1, T + 1))
        main_data_dict["x"].extend(agent_fut_np.get_attr("x"))
        main_data_dict["y"].extend(agent_fut_np.get_attr("y"))
        main_data_dict["h"].extend(agent_fut_np.get_attr("h"))
        main_data_dict["speed_mps"].extend(speed_mps)
        main_data_dict["speed_kph"].extend(speed_mps * 3.6)
        main_data_dict["type"].extend([agent_type_to_str(agent_type)] * T)
        main_data_dict["length"].extend(agent_extent[:, 0])
        main_data_dict["width"].extend(agent_extent[:, 1])
        main_data_dict["pred_agent"].extend([False] * T)
        main_data_dict["color"].extend([get_agent_type_color(agent_type)] * T)

    return pd.DataFrame(main_data_dict)


def plot_full_agent_batch_interactive(
    doc: Document, batch: AgentBatch, batch_idx: int
) -> None:
    fig = figure(
        match_aspect=True,
    )
    fig.grid.visible = False

    # Setting the match_aspect property of bokeh's default BoxZoomTool
    fig.tools[2].match_aspect = True

    agent_data_df = extract_full_agent_data_df(batch, batch_idx)
    agent_cds = ColumnDataSource(agent_data_df)
    curr_time_view = CDSView(
        source=agent_cds, filters=[BooleanFilter(agent_cds.data["t"] == 0)]
    )

    full_H = batch.agent_hist[batch_idx].shape[0]
    full_T = batch.agent_fut[batch_idx].shape[0]

    def create_multi_line_data_df(agents_df: pd.DataFrame) -> pd.DataFrame:
        lines_data = defaultdict(list)
        for agent_id, agent_df in agents_df.groupby(by="id"):
            xs, ys, color = agent_df.x, agent_df.y, agent_df.color.iat[0]
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

        return pd.DataFrame(lines_data)

    def get_sliced_multi_line_data_df(
        multi_line_df: pd.DataFrame, slice_obj
    ) -> pd.DataFrame:
        lines_data = defaultdict(list)
        for row in multi_line_df.itertuples(index=False):
            lines_data["xs"].append(row.xs[slice_obj])
            lines_data["ys"].append(row.ys[slice_obj])
            lines_data["color"].append(row.color)

        return pd.DataFrame(lines_data)

    history_line_data_df = create_multi_line_data_df(agent_data_df)
    history_lines_cds = ColumnDataSource(
        get_sliced_multi_line_data_df(history_line_data_df, slice(None, full_H))
    )
    future_line_data_df = history_line_data_df.copy()
    future_lines_cds = ColumnDataSource(
        get_sliced_multi_line_data_df(future_line_data_df, slice(full_H, None))
    )

    dt: float = batch.dt[batch_idx].item()
    scene_ts: int = batch.scene_ts[batch_idx].item()

    fig = figure()
    s = fig.scatter(x="x", y="y", color="color", source=agent_cds, view=curr_time_view)
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
            history_line_data_df, slice(None, new + full_H)
        )
        future_lines_cds.data = get_sliced_multi_line_data_df(
            future_line_data_df, slice(new + full_H, None)
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
                ("Position", "($x, $y) m"),
                ("Speed", "@speed_mps m/s (@speed_kph km/h)"),
            ],
            renderers=[s],
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

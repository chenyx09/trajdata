from typing import Tuple

import bokeh.plotting as plt
import numpy as np
import torch
from bokeh.models import ColumnDataSource
from bokeh.models.renderers import GlyphRenderer
from torch import Tensor

from trajdata.data_structures.agent import AgentType
from trajdata.data_structures.state import StateArray, StateTensor
from trajdata.maps import VectorMap
from trajdata.maps.vec_map_elements import RoadLane
from trajdata.utils.arr_utils import transform_coords_2d_np


class InteractiveFigure:
    def __init__(self, **kwargs) -> None:
        self.fig = plt.figure(match_aspect=True, **kwargs)
        self.fig.grid.visible = False

        # Setting the match_aspect property of bokeh's default BoxZoomTool
        self.fig.tools[2].match_aspect = True

    def show(self) -> None:
        plt.show(self.fig)

    def add_line(self, past_states: StateTensor, **kwargs) -> GlyphRenderer:
        xy_pos = past_states.position.cpu().numpy()
        return self.fig.line(xy_pos[:, 0], xy_pos[:, 1], **kwargs)

    def add_lines(self, lines_data: ColumnDataSource, **kwargs) -> GlyphRenderer:
        return self.fig.multi_line(
            source=lines_data,
            # This is to ensure that the columns given in the
            # ColumnDataSource are respected (e.g., "line_color").
            **{x: x for x in lines_data.column_names},
            **kwargs,
        )

    def add_map_at(
        self, center_pt: StateTensor, vec_map: VectorMap, radius: float = 50.0, **kwargs
    ):
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

        return self.add_lines(ColumnDataSource(data=lines_data))

    def add_agent(
        self,
        agent_type: AgentType,
        agent_state: StateTensor,
        agent_extent: Tensor,
        **kwargs,
    ) -> Tuple[GlyphRenderer, GlyphRenderer]:
        """Draws an agent at the given location, heading, and dimensions.

        Args:
            agent_type (AgentType): _description_
            agent_state (Tensor): _description_
            agent_extent (Tensor): _description_
        """
        if torch.any(torch.isnan(agent_extent)):
            if agent_type == AgentType.VEHICLE:
                length = 4.3
                width = 1.8
            elif agent_type == AgentType.PEDESTRIAN:
                length = 0.5
                width = 0.5
            elif agent_type == AgentType.BICYCLE:
                length = 1.9
                width = 0.5
            else:
                length = 1.0
                width = 1.0
        else:
            length = agent_extent[0].item()
            width = agent_extent[1].item()

        x, y = agent_state.position.cpu().numpy()
        heading = agent_state.heading.cpu().numpy()

        source = {
            "x": [x],
            "y": [y],
            "angle": [-heading],
            "width": [length],
            "height": [width],
            "type": [str(AgentType(agent_type.item()))[len("AgentType.") :]],
            "speed": [torch.linalg.norm(agent_state.velocity).item()],
        }
        r = self.fig.rect(
            x="x",
            y="y",
            angle="angle",
            width="width",
            height="height",
            source=source,
            **kwargs,
        )

        size = 1.0
        if agent_type == AgentType.PEDESTRIAN:
            size = 0.25

        dir_patch_coords = transform_coords_2d_np(
            np.array(
                [
                    [0, np.sqrt(3) / 3],
                    [-1 / 2, -np.sqrt(3) / 6],
                    [1 / 2, -np.sqrt(3) / 6],
                ]
            )
            * size,
            angle=heading - np.pi / 2,
        )
        p = self.fig.patch(
            x=dir_patch_coords[:, 0] + x, y=dir_patch_coords[:, 1] + y, **kwargs
        )

        return r, p

    def add_agents(
        self,
        agent_rects_data: ColumnDataSource,
        dir_patches_data: ColumnDataSource,
        **kwargs,
    ) -> Tuple[GlyphRenderer, GlyphRenderer]:
        r = self.fig.patches(
            source=agent_rects_data,
            # This is to ensure that the columns given in the
            # ColumnDataSource are respected (e.g., "line_color").
            xs="xs",
            ys="ys",
            fill_alpha="fill_alpha",
            fill_color="fill_color",
            line_color="line_color",
            **kwargs,
        )

        p = self.fig.patches(
            source=dir_patches_data,
            # This is to ensure that the columns given in the
            # ColumnDataSource are respected (e.g., "line_color").
            **{x: x for x in dir_patches_data.column_names},
            **kwargs,
        )

        return r, p

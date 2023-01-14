from typing import Tuple

import bokeh.plotting as plt
import numpy as np
import torch
from bokeh.models import ColumnDataSource
from bokeh.models.renderers import GlyphRenderer
from torch import Tensor

from trajdata.data_structures.agent import AgentType
from trajdata.data_structures.state import StateTensor
from trajdata.maps import VectorMap
from trajdata.utils import vis_utils
from trajdata.utils.arr_utils import transform_coords_2d_np


class InteractiveFigure:
    def __init__(self, **kwargs) -> None:
        self.raw_figure = plt.figure(match_aspect=True, **kwargs)
        self.raw_figure.grid.visible = False

        # Setting the match_aspect property of bokeh's default BoxZoomTool
        self.raw_figure.tools[2].match_aspect = True

    def show(self) -> None:
        plt.show(self.raw_figure)

    def add_line(self, past_states: StateTensor, **kwargs) -> GlyphRenderer:
        xy_pos = past_states.position.cpu().numpy()
        return self.raw_figure.line(xy_pos[:, 0], xy_pos[:, 1], **kwargs)

    def add_lines(self, lines_data: ColumnDataSource, **kwargs) -> GlyphRenderer:
        return self.raw_figure.multi_line(
            source=lines_data,
            # This is to ensure that the columns given in the
            # ColumnDataSource are respected (e.g., "line_color").
            **{x: x for x in lines_data.column_names},
            **kwargs,
        )

    def add_map_at(
        self,
        map_from_world_tf: np.ndarray,
        vec_map: VectorMap,
        bbox: Tuple[float, float, float, float],
        **kwargs,
    ) -> Tuple[
        GlyphRenderer, GlyphRenderer, GlyphRenderer, GlyphRenderer, GlyphRenderer
    ]:
        """_summary_

        Args:
            map_from_world_tf (np.ndarray): _description_
            vec_map (VectorMap): _description_
            bbox (Tuple[float, float, float, float]): x_min, x_max, y_min, y_max

        Returns:
            Tuple[ GlyphRenderer, GlyphRenderer, GlyphRenderer, GlyphRenderer, GlyphRenderer ]: _description_
        """
        return vis_utils.draw_map_elems(
            self.raw_figure, vec_map, map_from_world_tf, bbox, **kwargs
        )

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
        r = self.raw_figure.rect(
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
        p = self.raw_figure.patch(
            x=dir_patch_coords[:, 0] + x, y=dir_patch_coords[:, 1] + y, **kwargs
        )

        return r, p

    def add_agents(
        self,
        agent_rects_data: ColumnDataSource,
        dir_patches_data: ColumnDataSource,
        **kwargs,
    ) -> Tuple[GlyphRenderer, GlyphRenderer]:
        r = self.raw_figure.patches(
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

        p = self.raw_figure.patches(
            source=dir_patches_data,
            # This is to ensure that the columns given in the
            # ColumnDataSource are respected (e.g., "line_color").
            **{x: x for x in dir_patches_data.column_names},
            **kwargs,
        )

        return r, p

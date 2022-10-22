from typing import Optional

import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm

from trajdata.maps.map_utils import densify_polyline, proto_to_np
from trajdata.proto.vectorized_map_pb2 import MapElement, VectorizedMap


class MapElementKDTree:
    """
    Constructs a KDTree of MapElements and exposes fast lookup functions.

    Inheriting classes need to implement the _extra_points function that defines for a MapElement
    the coordinates we want to store in the KDTree.
    """

    def __init__(self, vec_map: VectorizedMap) -> None:
        # Keep track of any map offsets in case the map was shifted in previous functions.
        self.bottom_left_pt = np.array(
            [vec_map.bottom_left_coords.x, vec_map.bottom_left_coords.y]
        )

        # Build kd-tree
        kdtree, polylines, polyline_inds, map_elem_inds = self._build_kdtree(vec_map)
        self.kdtree = kdtree

        # It would be sufficient to store the index of map elements, having the original VectorizedMap object
        # we could then look up the MapElement and extract the polyline.
        # Instead we store polylines as well, which is a bit wasteful but more convenient as we do not need to
        # keep VectorizedMap for most usecases.
        self.polylines = polylines
        self.polyline_inds = polyline_inds
        self.map_elem_inds = map_elem_inds

    def _build_kdtree(self, vec_map: VectorizedMap):
        polylines = []
        polyline_inds = []
        map_elem_inds = []

        map_elem: MapElement
        for map_elem_ind, map_elem in enumerate(
            tqdm(vec_map.elements, desc=f"Building K-D Trees", leave=False)
        ):
            points = self._extract_points(map_elem)
            if points is not None:
                polyline_inds.extend([len(polylines)] * points.shape[0])
                map_elem_inds.append(map_elem_ind)

                # Apply any map offsets to ensure we're in the same coordinate area as the
                # original world map.
                polylines.append(points + self.bottom_left_pt)

        points = np.concatenate(polylines, axis=0)
        polyline_inds = np.array(polyline_inds)
        map_elem_inds = np.array(map_elem_inds)

        kdtree = KDTree(points)
        return kdtree, polylines, polyline_inds, map_elem_inds

    def _extract_points(self, map_element: MapElement) -> Optional[np.ndarray]:
        """Defines the coordinates we want to store in the KDTree for a MapElement.
        Args:
            map_element (MapElement): the MapElement to store in the KDTree.
        Returns:
            Optional[np.ndarray]: coordinates based on which we can search the KDTree, or None.
                If None, the MapElement will not be stored.
        """
        raise NotImplementedError

    def closest_point(self, query_points: np.ndarray) -> np.ndarray:
        """Find the closest KDTree points to (a batch of) query points.

        Args:
            query_points: np.ndarray of shape (..., data_dim).

        Return:
            np.ndarray of shape (..., data_dim), the KDTree points closest to query_point.
        """
        _, data_inds = self.kdtree.query(query_points, k=1)
        pts = self.kdtree.data[data_inds]
        return pts

    def closest_polyline(self, point: np.ndarray) -> np.ndarray:
        """Return the polyline closest to point. Does not support batch."""
        ind = self.closest_polyline_ind(point)
        return self.polylines[ind]

    def closest_polyline_ind(self, query_points: np.ndarray) -> np.ndarray:
        """Find the index of the closest polyline(s) in self.polylines."""
        _, data_ind = self.kdtree.query(query_points, k=1)
        return self.polyline_inds[data_ind]

    def polyline_inds_in_range(self, point: np.ndarray, range: float) -> np.ndarray:
        """Find the index of polylines in self.polylines within 'range' distance to 'point'."""
        data_inds = self.kdtree.query_ball_point(point, range)
        return np.unique(self.polyline_inds[data_inds], axis=0)


class LaneCenterKDTree(MapElementKDTree):
    """KDTree for lane center polylines."""

    def __init__(
        self, vec_map: VectorizedMap, max_segment_len: Optional[float] = None
    ) -> None:
        """
        Args:
            vec_map: the VectorizedMap object to build the KDTree for
            max_segment_len (float, optional): if specified, we will insert extra points into the KDTree
                such that all polyline segments are shorter then max_segment_len.
        """
        self.max_segment_len = max_segment_len
        super().__init__(vec_map)

    def _extract_points(self, map_element: MapElement) -> Optional[np.ndarray]:
        if map_element.HasField("road_lane"):
            pts: np.ndarray = proto_to_np(map_element.road_lane.center)

            # Some datasets provide 2d and some provide 3d points. We will only use 2d.
            pts = pts[:, :2]
            if pts.shape[0] <= 1:
                # Reject polylines with less than two points because we will not be able to recover heading.
                return None

            if self.max_segment_len is not None:
                pts = densify_polyline(pts, self.max_segment_len)

            return pts
        else:
            return None

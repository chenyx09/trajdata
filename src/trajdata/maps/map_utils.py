import numpy as np
from trajdata.proto.vectorized_map_pb2 import Lane, MapElement, VectorizedMap


def populate_lane_polylines(new_lane: Lane, midlane_pts: np.ndarray, left_pts: np.ndarray, right_pts: np.ndarray) -> None:
    """Fill a Lane object's polyline attributes. All points should be in world coordinates and 

    Args:
        new_lane (Lane): _description_
        midlane_pts (np.ndarray): _description_
        left_pts (np.ndarray): _description_
        right_pts (np.ndarray): _description_
    """
    
    assert midlane_pts.shape == left_pts.shape == right_pts.shape
    
    new_lane.center.x.extend(midlane_pts[:, 0].tolist())
    new_lane.center.y.extend(midlane_pts[:, 1].tolist())
    
    new_lane.left_boundary.x.extend(left_pts[:, 0].tolist())
    new_lane.left_boundary.y.extend(left_pts[:, 1].tolist())

    new_lane.right_boundary.x.extend(right_pts[:, 0].tolist())
    new_lane.right_boundary.y.extend(right_pts[:, 1].tolist())

    if midlane_pts.shape[-1] == 3:
        new_lane.center.z.extend(midlane_pts[:, 2].tolist())
        new_lane.left_boundary.z.extend(left_pts[:, 2].tolist())
        new_lane.right_boundary.z.extend(right_pts[:, 2].tolist())

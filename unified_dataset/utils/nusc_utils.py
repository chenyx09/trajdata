from typing import Any, Dict, List, Optional, Set, Union
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

from unified_dataset.data_structures import Agent, AgentMetadata, SceneMetadata


def frame_iterator(nusc_obj: NuScenes, scene_metadata: SceneMetadata) -> Dict[str, Union[str, int]]:
    """Loops through all frames in a scene and yields them for the caller to deal with the information.
    """
    curr_scene_token: str = scene_metadata.data_access_info['first_sample_token']
    while curr_scene_token:
        frame = nusc_obj.get('sample', curr_scene_token)

        yield frame

        curr_scene_token = frame['next']


def agent_iterator(nusc_obj: NuScenes, frame_info: Dict[str, Any]) -> Dict[str, Any]:
    """Loops through all annotations (agents) in a frame and yields them for the caller to deal with the information.
    """
    ann_token: str
    for ann_token in frame_info['anns']:
        ann_record = nusc_obj.get('sample_annotation', ann_token)

        agent_category: str = ann_record['category_name']
        if agent_category.startswith('vehicle') or agent_category.startswith('human'):
            yield ann_record


def get_ego_pose(nusc_obj: NuScenes, frame_info: Dict[str, Any]) -> Dict[str, Any]:
    cam_front_data = nusc_obj.get('sample_data', frame_info['data']['CAM_FRONT'])
    ego_pose = nusc_obj.get('ego_pose', cam_front_data['ego_pose_token'])
    return ego_pose
    

def agg_agent_data(nusc_obj: NuScenes, agent_data: Dict[str, Any], curr_scene_index: int) -> Agent:
    """Loops through all annotations of a specific agent in a scene and aggregates their data into an Agent object.
    """
    if agent_data['past']:
        print('WARN: This is not the first frame of this agent!')

    translation_list = [agent_data['translation']]
    size_list = [agent_data['size']]
    yaw_list = [Quaternion(agent_data['rotation']).yaw_pitch_roll[0]]
    
    curr_sample_ann_token: str = agent_data['next']
    while curr_sample_ann_token:
        agent_data = nusc_obj.get('sample_annotation', curr_sample_ann_token)

        translation_list.append(agent_data['translation'])
        size_list.append(agent_data['size'])
        yaw_list.append(Quaternion(agent_data['rotation']).yaw_pitch_roll[0])

        curr_sample_ann_token = agent_data['next']

    category_mode = 0
    agent_metadata = AgentMetadata(agent_data['instance_token'], category_mode, curr_scene_index)
    return Agent(agent_metadata)

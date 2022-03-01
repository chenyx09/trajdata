import dill
from pathlib import Path
import numpy as np
import pandas as pd

from typing import Any, Dict, List, Optional, Set, Union
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

from unified_dataset.data_structures import Agent, AgentType, AgentMetadata, SceneMetadata


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
    if agent_data['prev']:
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

    translations_np = np.stack(translation_list, axis=0)
    sizes_np = np.stack(size_list, axis=0)
    yaws_np = np.expand_dims(np.stack(yaw_list, axis=0), axis=1)

    agent_data_np = np.concatenate([translations_np, yaws_np, sizes_np], axis=1)
    agent_data_df = pd.DataFrame(agent_data_np, columns=['x', 'y', 'z', 'heading', 'length', 'width', 'height'])

    agent_type = nusc_type_to_unified_type(agent_data['category_name'])
    agent_metadata = AgentMetadata(name=agent_data['instance_token'], agent_type=agent_type, first_timestep=curr_scene_index)
    return Agent(agent_metadata, agent_data_df)


def nusc_type_to_unified_type(nusc_type: str) -> AgentType:
    if nusc_type.startswith('human'):
        return AgentType.PEDESTRIAN
    elif nusc_type == 'vehicle.bicycle':
        return AgentType.BICYCLE
    elif nusc_type == 'vehicle.motorcycle':
        return AgentType.MOTORCYCLE
    elif nusc_type.startswith('vehicle'):
        return AgentType.VEHICLE
    else:
        return AgentType.UNKNOWN


def agg_ego_data(nusc_obj: NuScenes, scene_metadata: SceneMetadata) -> Agent:
    translation_list = list()
    size_list = list()
    yaw_list = list()
    for frame_info in frame_iterator(nusc_obj, scene_metadata):
        ego_pose = get_ego_pose(nusc_obj, frame_info)
        yaw_list.append(Quaternion(ego_pose['rotation']).yaw_pitch_roll[0])
        translation_list.append(ego_pose['translation'])
        size_list.append(np.array([4.084, 1.730, 1.562]))

    translations_np = np.stack(translation_list, axis=0)
    sizes_np = np.stack(size_list, axis=0)
    yaws_np = np.expand_dims(np.stack(yaw_list, axis=0), axis=1)

    ego_data_np = np.concatenate([translations_np, yaws_np, sizes_np], axis=1)
    ego_data_df = pd.DataFrame(ego_data_np, columns=['x', 'y', 'z', 'heading', 'length', 'width', 'height'])
    
    ego_metadata = AgentMetadata(name='ego', agent_type=AgentType.VEHICLE, first_timestep=0)
    return Agent(ego_metadata, ego_data_df)


def create_scene_timestep_metadata(scene_info: SceneMetadata, nusc_obj: NuScenes, cache_scene_dir: Path, rebuild_cache: bool) -> List[List[str]]:
    agent_presence: List[List[str]] = [[] for _ in range(scene_info.length_timesteps)]
    for frame_idx, frame_info in enumerate(frame_iterator(nusc_obj, scene_info)):
        agent_presence[frame_idx] = ['ego']
        for agent_info in agent_iterator(nusc_obj, frame_info):
            agent_id: str = agent_info['instance_token']
            agent_file: Path = cache_scene_dir / f"{agent_id}.dill"
            agent_presence[frame_idx].append(agent_id)
            if agent_file.is_file() and not rebuild_cache:
                continue
            
            agent: Agent = agg_agent_data(nusc_obj, agent_info, frame_idx)
            with open(agent_file, 'wb') as f:
                dill.dump(agent, f)

    ego_file: Path = cache_scene_dir / "ego.dill"
    if not ego_file.is_file() or rebuild_cache:
        ego_agent: Agent = agg_ego_data(nusc_obj, scene_info)
        with open(ego_file, 'wb') as f:
            dill.dump(ego_agent, f)

    return agent_presence

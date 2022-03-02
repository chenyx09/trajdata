import dill
import numpy as np
from scipy.stats import mode
import pandas as pd
from pathlib import Path
from typing import List, Tuple

from unified_dataset.data_structures import SceneMetadata, EnvMetadata, AgentMetadata, AgentType, Agent, FixedSize

from l5kit.data import ChunkedDataset, labels
from l5kit.geometry import rotation33_as_yaw


def get_matching_scenes(lyft_obj: ChunkedDataset, env_info: EnvMetadata, dataset_tuple: Tuple[str, ...]):
    scenes_list: List[SceneMetadata] = list()
    all_scene_frames = lyft_obj.scenes['frame_index_interval']
    for idx in range(all_scene_frames.shape[0]):
        scene_name: str = f'scene-{idx:04d}'
        scene_length: int = (all_scene_frames[idx, 1] - all_scene_frames[idx, 0]).item() # Doing .item() otherwise it'll be a numpy.int64
        scene_metadata = SceneMetadata(env_info, scene_name, 'palo_alto', env_info.scene_split_map[scene_name], scene_length, all_scene_frames[idx])
        scenes_list.append(scene_metadata)

    return scenes_list


def agg_ego_data(lyft_obj: ChunkedDataset, scene_metadata: SceneMetadata) -> Agent:
    scene_frame_start = scene_metadata.data_access_info[0]
    scene_frame_end = scene_metadata.data_access_info[1]

    ego_translations = lyft_obj.frames[scene_frame_start : scene_frame_end]['ego_translation']
    ego_rotations = lyft_obj.frames[scene_frame_start : scene_frame_end]['ego_rotation']
    ego_yaws = np.array([rotation33_as_yaw(ego_rotations[i]) for i in range(scene_metadata.length_timesteps)])

    ego_data_np = np.concatenate([ego_translations, np.expand_dims(ego_yaws, axis=1)], axis=1)
    ego_data_df = pd.DataFrame(ego_data_np, columns=['x', 'y', 'z', 'heading'])

    ego_metadata = AgentMetadata(name='ego', agent_type=AgentType.VEHICLE, first_timestep=0)
    return Agent(ego_metadata, ego_data_df, fixed_size=FixedSize(length=4.869, width=1.852, height=1.476))


def lyft_type_to_unified_type(lyft_type: int) -> AgentType:
    # TODO(bivanovic): Currently not handling TRAM or ANIMAL.
    if lyft_type in [0, 1, 2, 16]:
        return AgentType.UNKNOWN
    elif lyft_type in [3, 4, 6, 7, 8, 9]:
        return AgentType.VEHICLE
    elif lyft_type in [10, 12]:
        return AgentType.BICYCLE
    elif lyft_type in [11, 13]:
        return AgentType.MOTORCYCLE
    elif lyft_type == 14:
        return AgentType.PEDESTRIAN


def calc_agent_presence(scene_info: SceneMetadata, lyft_obj: ChunkedDataset, cache_scene_dir: Path, rebuild_cache: bool) -> List[List[str]]:
    agent_presence: List[List[str]] = [['ego'] for _ in range(scene_info.length_timesteps)]
    
    ego_file: Path = cache_scene_dir / "ego.dill"
    if not ego_file.is_file() or rebuild_cache:
        ego_agent: Agent = agg_ego_data(lyft_obj, scene_info)
        with open(ego_file, 'wb') as f:
            dill.dump(ego_agent, f)

    scene_frame_start = scene_info.data_access_info[0]
    scene_frame_end = scene_info.data_access_info[1]

    agent_indices = lyft_obj.frames[scene_frame_start : scene_frame_end]['agent_index_interval']
    agent_start_idx = agent_indices[0, 0]
    agent_end_idx = agent_indices[-1, 1]

    lyft_agents = lyft_obj.agents[agent_start_idx : agent_end_idx]
    agent_ids = lyft_agents['track_id']

    # This is so we later know what is the first scene timestep that an agent appears in the scene.
    num_agents_per_ts = agent_indices[:, 1] - agent_indices[:, 0]
    agent_frame_ids = np.repeat(np.arange(scene_info.length_timesteps), num_agents_per_ts)
    agent_frames_series = pd.Series(agent_frame_ids, index=[agent_ids])

    agent_translations = lyft_agents['centroid']
    agent_sizes = lyft_agents['extent']
    agent_yaws = lyft_agents['yaw']
    agent_velocities = lyft_agents['velocity']
    agent_probs = lyft_agents['label_probabilities']

    normal_cols = ['x', 'y', 'heading', 'length', 'width', 'height', 'vx', 'vy']
    class_start = len('PERCEPTION_LABEL')
    label_cols = ['prob' + label[class_start:] for label in labels.PERCEPTION_LABELS[:-1]]
    
    all_agent_data = np.concatenate([agent_translations, np.expand_dims(agent_yaws, axis=1), agent_sizes, agent_velocities, agent_probs[:, :-1]], axis=1)
    all_agent_data_df = pd.DataFrame(all_agent_data, columns=normal_cols + label_cols, index=[agent_ids])

    for agent_id in np.unique(agent_ids):
        agent_name: str = str(agent_id)
        scene_frames = agent_frames_series.loc[agent_id]

        for frame in scene_frames:
            agent_presence[frame].append(agent_name)

        agent_file: Path = cache_scene_dir / f"{agent_name}.dill"
        if agent_file.is_file() and not rebuild_cache:
            continue
        
        agent_data_df: pd.DataFrame = all_agent_data_df.loc[agent_id].reset_index(drop=True)
        start_frame = scene_frames.iloc[0]
        mode_type = mode(np.argmax(agent_data_df[label_cols].values, axis=1))[0].item()
        agent_type = lyft_type_to_unified_type(mode_type)
        
        agent_metadata = AgentMetadata(name=agent_name, agent_type=agent_type, first_timestep=start_frame.item())

        # For now only saving non-prob columns since Lyft is effectively one-hot (see https://arxiv.org/abs/2104.12446)
        agent = Agent(agent_metadata, agent_data_df[normal_cols])
        with open(agent_file, 'wb') as f:
            dill.dump(agent, f)

    return agent_presence

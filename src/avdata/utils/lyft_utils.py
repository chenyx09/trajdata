import contextlib
import sqlite3
from pathlib import Path
from typing import List, NamedTuple, Tuple

import dill
import numpy as np
import pandas as pd
from l5kit.data import ChunkedDataset, labels
from l5kit.geometry import rotation33_as_yaw
from scipy.stats import mode

from avdata.data_structures import (
    Agent,
    AgentMetadata,
    AgentType,
    EnvMetadata,
    FixedSize,
    SceneMetadata,
)

LYFT_DT = 0.1
LYFT_DB_SCHEMA = """
agent_id TEXT NOT NULL,
scene_ts INTEGER NOT NULL,
x REAL NOT NULL,
y REAL NOT NULL,
vx REAL NOT NULL,
vy REAL NOT NULL,
ax REAL NOT NULL,
ay REAL NOT NULL,
heading REAL NOT NULL
"""


class LyftSceneRecord(NamedTuple):
    name: str
    length: str


def _get_matching_scenes_from_obj(
    lyft_obj: ChunkedDataset,
    env_info: EnvMetadata,
    dataset_tuple: Tuple[str, ...],
    env_cache_dir: Path,
) -> List[SceneMetadata]:
    all_scenes_list: List[LyftSceneRecord] = list()

    scenes_list: List[SceneMetadata] = list()
    all_scene_frames = lyft_obj.scenes["frame_index_interval"]
    for idx in range(all_scene_frames.shape[0]):
        scene_name: str = f"scene-{idx:04d}"
        scene_length: int = (
            all_scene_frames[idx, 1] - all_scene_frames[idx, 0]
        ).item()  # Doing .item() otherwise it'll be a numpy.int64

        # Saving all scene records for later caching.
        all_scenes_list.append(LyftSceneRecord(scene_name, scene_length))

        scene_metadata = SceneMetadata(
            env_info,
            scene_name,
            "palo_alto",
            env_info.scene_split_map[scene_name],
            scene_length,
            all_scene_frames[idx],
        )
        scenes_list.append(scene_metadata)

    env_cache_dir.mkdir(parents=True, exist_ok=True)
    with open(env_cache_dir / "scenes_list.dill", "wb") as f:
        dill.dump(all_scenes_list, f)

    return scenes_list


def _get_matching_scenes_from_cache(
    env_info: EnvMetadata, dataset_tuple: Tuple[str, ...], env_cache_dir: Path
) -> List[SceneMetadata]:
    with open(env_cache_dir / "scenes_list.dill", "rb") as f:
        all_scenes_list: List[LyftSceneRecord] = dill.load(f)

    scenes_list: List[SceneMetadata] = list()
    for scene_record in all_scenes_list:
        scene_name, scene_length = scene_record

        scene_metadata = SceneMetadata(
            env_info,
            scene_name,
            "palo_alto",
            env_info.scene_split_map[scene_name],
            scene_length,
            None,  # This isn't used if everything is already cached.
        )
        scenes_list.append(scene_metadata)

    return scenes_list


def get_matching_scenes(
    lyft_obj: ChunkedDataset,
    env_info: EnvMetadata,
    dataset_tuple: Tuple[str, ...],
    env_cache_dir: Path,
    rebuild_cache: bool = False,
) -> List[SceneMetadata]:

    if lyft_obj is None and not rebuild_cache:
        return _get_matching_scenes_from_cache(env_info, dataset_tuple, env_cache_dir)
    else:
        return _get_matching_scenes_from_obj(
            lyft_obj, env_info, dataset_tuple, env_cache_dir
        )


def agg_ego_data(lyft_obj: ChunkedDataset, scene_metadata: SceneMetadata) -> Agent:
    scene_frame_start = scene_metadata.data_access_info[0]
    scene_frame_end = scene_metadata.data_access_info[1]

    ego_translations = lyft_obj.frames[scene_frame_start:scene_frame_end][
        "ego_translation"
    ][:, :2]

    # Doing this prepending so that the first velocity isn't zero (rather it's just the first actual velocity duplicated)
    prepend_pos = ego_translations[0] - (ego_translations[1] - ego_translations[0])
    ego_velocities = (
        np.diff(ego_translations, axis=0, prepend=np.expand_dims(prepend_pos, axis=0))
        / LYFT_DT
    )

    # Doing this prepending so that the first acceleration isn't zero (rather it's just the first actual acceleration duplicated)
    prepend_vel = ego_velocities[0] - (ego_velocities[1] - ego_velocities[0])
    ego_accelerations = (
        np.diff(ego_velocities, axis=0, prepend=np.expand_dims(prepend_vel, axis=0))
        / LYFT_DT
    )

    ego_rotations = lyft_obj.frames[scene_frame_start:scene_frame_end]["ego_rotation"]
    ego_yaws = np.array(
        [
            rotation33_as_yaw(ego_rotations[i])
            for i in range(scene_metadata.length_timesteps)
        ]
    )

    ego_data_np = np.concatenate(
        [
            ego_translations,
            ego_velocities,
            ego_accelerations,
            np.expand_dims(ego_yaws, axis=1),
        ],
        axis=1,
    )
    ego_data_df = pd.DataFrame(
        ego_data_np, columns=["x", "y", "vx", "vy", "ax", "ay", "heading"]
    )
    ego_data_df["agent_id"] = "ego"
    ego_data_df.index.name = "scene_ts"

    ego_metadata = AgentMetadata(
        name="ego",
        agent_type=AgentType.VEHICLE,
        first_timestep=0,
        last_timestep=ego_data_np.shape[0] - 1,
        fixed_size=FixedSize(length=4.869, width=1.852, height=1.476),
    )
    return Agent(
        metadata=ego_metadata,
        data=ego_data_df,
    )


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


# @profile
def calc_agent_presence(
    scene_info: SceneMetadata,
    lyft_obj: ChunkedDataset,
    cache_scene_dir: Path,
    rebuild_cache: bool,
) -> List[List[AgentMetadata]]:
    agent_presence: List[List[AgentMetadata]] = [
        [
            AgentMetadata(
                name="ego",
                agent_type=AgentType.VEHICLE,
                first_timestep=0,
                last_timestep=scene_info.length_timesteps - 1,
                fixed_size=FixedSize(length=4.869, width=1.852, height=1.476),
            )
        ]
        for _ in range(scene_info.length_timesteps)
    ]

    agent_data_list: List[pd.DataFrame] = list()

    ego_agent: Agent = agg_ego_data(lyft_obj, scene_info)
    agent_data_list.append(ego_agent.data)

    scene_frame_start = scene_info.data_access_info[0]
    scene_frame_end = scene_info.data_access_info[1]

    agent_indices = lyft_obj.frames[scene_frame_start:scene_frame_end][
        "agent_index_interval"
    ]
    agent_start_idx = agent_indices[0, 0]
    agent_end_idx = agent_indices[-1, 1]

    lyft_agents = lyft_obj.agents[agent_start_idx:agent_end_idx]
    agent_ids = lyft_agents["track_id"]

    # This is so we later know what is the first scene timestep that an agent appears in the scene.
    num_agents_per_ts = agent_indices[:, 1] - agent_indices[:, 0]
    agent_frame_ids = np.repeat(
        np.arange(scene_info.length_timesteps), num_agents_per_ts
    )

    agent_translations = lyft_agents["centroid"]
    agent_velocities = lyft_agents["velocity"]
    # agent_sizes = lyft_agents['extent']
    agent_yaws = lyft_agents["yaw"]
    agent_probs = lyft_agents["label_probabilities"]

    current_cols = ["x", "y", "vx", "vy", "heading"]
    final_cols = [
        "agent_id",
        "x",
        "y",
        "vx",
        "vy",
        "ax",
        "ay",
        "heading",
    ]  # Accelerations we have to do later per agent
    class_start = len("PERCEPTION_LABEL")
    label_cols = [
        "prob" + label[class_start:] for label in labels.PERCEPTION_LABELS[:-1]
    ]

    all_agent_data = np.concatenate(
        [
            agent_translations,
            agent_velocities,
            np.expand_dims(agent_yaws, axis=1),
            agent_probs[:, :-1],
        ],
        axis=1,
    )
    all_agent_data_df = pd.DataFrame(
        all_agent_data,
        columns=current_cols + label_cols,
        index=[agent_ids, agent_frame_ids],
    )
    all_agent_data_df.index.names = ["agent_id", "scene_ts"]

    for agent_id in np.unique(agent_ids):
        agent_data_df: pd.DataFrame = all_agent_data_df.loc[agent_id].copy()

        if len(agent_data_df) <= 1:
            # There are some agents with only a single detection to them, we don't care about these.
            continue

        start_frame: int = agent_data_df.index[0].item()
        last_frame: int = agent_data_df.index[-1].item()
        mode_type: int = mode(np.argmax(agent_data_df[label_cols].values, axis=1))[
            0
        ].item()
        agent_type: AgentType = lyft_type_to_unified_type(mode_type)

        # Doing this prepending so that the first acceleration isn't zero (rather it's just the first actual acceleration duplicated)
        prepend_vx = agent_data_df.at[start_frame, "vx"] - (
            agent_data_df.at[start_frame + 1, "vx"]
            - agent_data_df.at[start_frame, "vx"]
        )
        prepend_vy = agent_data_df.at[start_frame, "vy"] - (
            agent_data_df.at[start_frame + 1, "vy"]
            - agent_data_df.at[start_frame, "vy"]
        )
        agent_data_df[["ax", "ay"]] = (
            np.diff(
                agent_data_df[["vx", "vy"]],
                axis=0,
                prepend=np.array([[prepend_vx, prepend_vy]]),
            )
            / LYFT_DT
        )

        agent_metadata = AgentMetadata(
            name=str(agent_id),
            agent_type=agent_type,
            first_timestep=start_frame,
            last_timestep=last_frame,
        )

        for frame in agent_data_df.index:
            agent_presence[frame].append(agent_metadata)

        agent_data_df["agent_id"] = agent_metadata.name

        # For now only saving non-prob columns since Lyft is effectively one-hot (see https://arxiv.org/abs/2104.12446)
        agent = Agent(agent_metadata, agent_data_df[final_cols])
        agent_data_list.append(agent.data)

    with contextlib.closing(
        sqlite3.connect(cache_scene_dir / "agent_data.db")
    ) as connection:
        cursor = connection.cursor()
        cursor.execute(f"CREATE TABLE IF NOT EXISTS agent_data ({LYFT_DB_SCHEMA})")
        pd.concat(agent_data_list).to_sql(
            name="agent_data",
            con=connection,
            if_exists="replace",
            index=True,
            index_label="scene_ts",
        )

    return agent_presence

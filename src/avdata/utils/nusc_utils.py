import contextlib
import sqlite3
from pathlib import Path
from sqlite3 import Cursor
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

from avdata.data_structures import (
    Agent,
    AgentMetadata,
    AgentType,
    EnvMetadata,
    FixedSize,
    SceneMetadata,
)

NUSC_DT = 0.5
NUSC_DB_SCHEMA = """
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


def frame_iterator(
    nusc_obj: NuScenes, scene_metadata: SceneMetadata
) -> Dict[str, Union[str, int]]:
    """Loops through all frames in a scene and yields them for the caller to deal with the information."""
    curr_scene_token: str = scene_metadata.data_access_info["first_sample_token"]
    while curr_scene_token:
        frame = nusc_obj.get("sample", curr_scene_token)

        yield frame

        curr_scene_token = frame["next"]


def agent_iterator(nusc_obj: NuScenes, frame_info: Dict[str, Any]) -> Dict[str, Any]:
    """Loops through all annotations (agents) in a frame and yields them for the caller to deal with the information."""
    ann_token: str
    for ann_token in frame_info["anns"]:
        ann_record = nusc_obj.get("sample_annotation", ann_token)

        agent_category: str = ann_record["category_name"]
        if agent_category.startswith("vehicle") or agent_category.startswith("human"):
            yield ann_record


def get_matching_scenes(
    nusc_obj: NuScenes, env_info: EnvMetadata, dataset_tuple: Tuple[str, ...]
) -> List[SceneMetadata]:
    scenes_list: List[SceneMetadata] = list()
    for scene_record in nusc_obj.scene:
        scene_name: str = scene_record["name"]
        scene_location: str = nusc_obj.get("log", scene_record["log_token"])["location"]
        scene_split: str = env_info.scene_split_map[scene_name]
        scene_length: int = scene_record["nbr_samples"]

        if (
            scene_location.split("-")[0] in dataset_tuple
            and scene_split in dataset_tuple
        ):
            scene_metadata = SceneMetadata(
                env_info,
                scene_name,
                scene_location,
                scene_split,
                scene_length,
                scene_record,
            )
            scenes_list.append(scene_metadata)

    return scenes_list


def get_ego_pose(nusc_obj: NuScenes, frame_info: Dict[str, Any]) -> Dict[str, Any]:
    cam_front_data = nusc_obj.get("sample_data", frame_info["data"]["CAM_FRONT"])
    ego_pose = nusc_obj.get("ego_pose", cam_front_data["ego_pose_token"])
    return ego_pose


def agg_agent_data(
    nusc_obj: NuScenes, agent_data: Dict[str, Any], curr_scene_index: int
) -> Agent:
    """Loops through all annotations of a specific agent in a scene and aggregates their data into an Agent object."""
    if agent_data["prev"]:
        print("WARN: This is not the first frame of this agent!")

    translation_list = [agent_data["translation"]]
    agent_size = agent_data["size"]
    yaw_list = [Quaternion(agent_data["rotation"]).yaw_pitch_roll[0]]

    curr_sample_ann_token: str = agent_data["next"]
    while curr_sample_ann_token:
        agent_data = nusc_obj.get("sample_annotation", curr_sample_ann_token)

        translation_list.append(agent_data["translation"])
        # size_list.append(agent_data['size'])
        yaw_list.append(Quaternion(agent_data["rotation"]).yaw_pitch_roll[0])

        curr_sample_ann_token = agent_data["next"]

    translations_np = np.stack(translation_list, axis=0)[:, :2]

    # Doing this prepending so that the first velocity isn't zero (rather it's just the first actual velocity duplicated)
    prepend_pos = translations_np[0] - (translations_np[1] - translations_np[0])
    velocities_np = (
        np.diff(translations_np, axis=0, prepend=np.expand_dims(prepend_pos, axis=0))
        / NUSC_DT
    )

    # Doing this prepending so that the first acceleration isn't zero (rather it's just the first actual acceleration duplicated)
    prepend_vel = velocities_np[0] - (velocities_np[1] - velocities_np[0])
    accelerations_np = (
        np.diff(velocities_np, axis=0, prepend=np.expand_dims(prepend_vel, axis=0))
        / NUSC_DT
    )

    yaws_np = np.expand_dims(np.stack(yaw_list, axis=0), axis=1)
    # sizes_np = np.stack(size_list, axis=0)

    agent_data_np = np.concatenate(
        [translations_np, velocities_np, accelerations_np, yaws_np], axis=1
    )
    last_timestep = curr_scene_index + agent_data_np.shape[0] - 1
    agent_data_df = pd.DataFrame(
        agent_data_np,
        columns=["x", "y", "vx", "vy", "ax", "ay", "heading"],
        index=list(range(curr_scene_index, last_timestep + 1)),
    )
    agent_data_df.index.name = "scene_ts"
    agent_data_df["agent_id"] = agent_data["instance_token"]

    agent_type = nusc_type_to_unified_type(agent_data["category_name"])
    agent_metadata = AgentMetadata(
        name=agent_data["instance_token"],
        agent_type=agent_type,
        first_timestep=curr_scene_index,
        last_timestep=last_timestep,
        fixed_size=FixedSize(
            length=agent_size[0], width=agent_size[1], height=agent_size[2]
        ),
    )
    return Agent(
        metadata=agent_metadata,
        data=agent_data_df,
    )


def nusc_type_to_unified_type(nusc_type: str) -> AgentType:
    if nusc_type.startswith("human"):
        return AgentType.PEDESTRIAN
    elif nusc_type == "vehicle.bicycle":
        return AgentType.BICYCLE
    elif nusc_type == "vehicle.motorcycle":
        return AgentType.MOTORCYCLE
    elif nusc_type.startswith("vehicle"):
        return AgentType.VEHICLE
    else:
        return AgentType.UNKNOWN


def agg_ego_data(nusc_obj: NuScenes, scene_metadata: SceneMetadata) -> Agent:
    translation_list = list()
    yaw_list = list()
    for frame_info in frame_iterator(nusc_obj, scene_metadata):
        ego_pose = get_ego_pose(nusc_obj, frame_info)
        yaw_list.append(Quaternion(ego_pose["rotation"]).yaw_pitch_roll[0])
        translation_list.append(ego_pose["translation"][:2])

    translations_np = np.stack(translation_list, axis=0)

    # Doing this prepending so that the first velocity isn't zero (rather it's just the first actual velocity duplicated)
    prepend_pos = translations_np[0] - (translations_np[1] - translations_np[0])
    velocities_np = (
        np.diff(translations_np, axis=0, prepend=np.expand_dims(prepend_pos, axis=0))
        / NUSC_DT
    )

    # Doing this prepending so that the first acceleration isn't zero (rather it's just the first actual acceleration duplicated)
    prepend_vel = velocities_np[0] - (velocities_np[1] - velocities_np[0])
    accelerations_np = (
        np.diff(velocities_np, axis=0, prepend=np.expand_dims(prepend_vel, axis=0))
        / NUSC_DT
    )

    yaws_np = np.expand_dims(np.stack(yaw_list, axis=0), axis=1)

    ego_data_np = np.concatenate(
        [translations_np, velocities_np, accelerations_np, yaws_np], axis=1
    )
    ego_data_df = pd.DataFrame(
        ego_data_np, columns=["x", "y", "vx", "vy", "ax", "ay", "heading"]
    )
    ego_data_df.index.name = "scene_ts"
    ego_data_df["agent_id"] = "ego"

    ego_metadata = AgentMetadata(
        name="ego",
        agent_type=AgentType.VEHICLE,
        first_timestep=0,
        last_timestep=ego_data_np.shape[0] - 1,
        fixed_size=FixedSize(length=4.084, width=1.730, height=1.562),
    )
    return Agent(
        metadata=ego_metadata,
        data=ego_data_df,
    )


def agent_exists(cursor: Cursor, agent_id: str):
    cursor.execute(
        "SELECT EXISTS(SELECT 1 FROM agent_data WHERE agent_id=?)", (agent_id,)
    )
    return bool(cursor.fetchone()[0])


# @profile
def calc_agent_presence(
    scene_info: SceneMetadata,
    nusc_obj: NuScenes,
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
                fixed_size=FixedSize(length=4.084, width=1.730, height=1.562),
            )
        ]
        for _ in range(scene_info.length_timesteps)
    ]

    agent_data_list: List[pd.DataFrame] = list()
    existing_agents: Dict[str, AgentMetadata] = dict()
    for frame_idx, frame_info in enumerate(frame_iterator(nusc_obj, scene_info)):
        for agent_info in agent_iterator(nusc_obj, frame_info):
            if agent_info["instance_token"] in existing_agents:
                agent_presence[frame_idx].append(
                    existing_agents[agent_info["instance_token"]]
                )
                continue

            if not agent_info["next"]:
                # There are some agents with only a single detection to them, we don't care about these.
                continue

            agent: Agent = agg_agent_data(nusc_obj, agent_info, frame_idx)

            agent_presence[frame_idx].append(agent.metadata)
            existing_agents[agent.name] = agent.metadata

            agent_data_list.append(agent.data)

    ego_agent: Agent = agg_ego_data(nusc_obj, scene_info)
    agent_data_list.append(ego_agent.data)

    with contextlib.closing(
        sqlite3.connect(cache_scene_dir / "agent_data.db")
    ) as connection:
        cursor = connection.cursor()
        cursor.execute(f"CREATE TABLE IF NOT EXISTS agent_data ({NUSC_DB_SCHEMA})")
        pd.concat(agent_data_list).to_sql(
            name="agent_data",
            con=connection,
            if_exists="replace",
            index=True,
            index_label="scene_ts",
        )

    return agent_presence

import numpy as np
import pandas as pd
from l5kit.data import ChunkedDataset
from l5kit.geometry import rotation33_as_yaw

from avdata.data_structures import (
    Agent,
    AgentMetadata,
    AgentType,
    FixedSize,
    SceneMetadata,
)

LYFT_DT = 0.1


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
        ego_data_np,
        columns=["x", "y", "vx", "vy", "ax", "ay", "heading"],
        index=pd.MultiIndex.from_tuples(
            [("ego", idx) for idx in range(ego_data_np.shape[0])],
            names=["agent_id", "scene_ts"],
        ),
    )

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

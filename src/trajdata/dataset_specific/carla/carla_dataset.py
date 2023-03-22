import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import pandas as pd
from nuscenes.eval.prediction.splits import NUM_IN_TRAIN_VAL
from nuscenes.map_expansion.map_api import NuScenesMap, locations
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from tqdm import tqdm

from trajdata.caching import EnvCache, SceneCache
from trajdata.data_structures.agent import (
    Agent,
    AgentMetadata,
    AgentType,
    FixedExtent,
    VariableExtent,
)
from trajdata.data_structures.environment import EnvMetadata
from trajdata.data_structures.scene_metadata import Scene, SceneMetadata
from trajdata.data_structures.scene_tag import SceneTag
from trajdata.dataset_specific.nusc import nusc_utils
from trajdata.dataset_specific.raw_dataset import RawDataset
from trajdata.dataset_specific.scene_records import CarlaSceneRecord
from trajdata.maps import VectorMap
from trajdata.maps.map_api import MapAPI

from pdb import set_trace as st
import glob, os
import pickle
from collections import defaultdict
import re

import torch
import numpy as np

carla_to_trajdata_object_type = {
    0: AgentType.VEHICLE,
    1: AgentType.BICYCLE,  # Motorcycle
    2: AgentType.BICYCLE,
    3: AgentType.PEDESTRIAN,
    4: AgentType.UNKNOWN,
    # ?: AgentType.STATIC,
}

def create_splits_scenes(data_dir:str) -> Dict[str, List[str]]:
    all_scenes = {}
    all_scenes['train'] = [scene_path.split('/')[-1] for scene_path in glob.glob(data_dir+'/train/route*')]
    all_scenes['val'] = [scene_path.split('/')[-1] for scene_path in glob.glob(data_dir+'/val/route*')]
    return all_scenes

# TODO: (Yulong) format in object class
def tracklet_to_pred(tracklet_mem,ego=False):
    if ego:
        x, y, z = np.split(tracklet_mem['location'],3,axis=-1)
        hx, hy, hz = np.split(np.deg2rad(tracklet_mem['rotation']),3,axis=-1)
        vx, vy, _ = np.split(tracklet_mem['velocity'],3,axis=-1)
        ax, ay, _ = np.split(tracklet_mem['acceleration'],3,axis=-1)
    else:
        x, y, z = np.split(tracklet_mem['location'][0,:,-1,:],3,axis=-1)
        hx, hy, hz = np.split(np.deg2rad(tracklet_mem['rotation'])[0,:,-1,:],3,axis=-1)
        vx, vy, _ = np.split(tracklet_mem['velocity'][0,:,-1,:],3,axis=-1)
        ax, ay, _ = np.split(tracklet_mem['acceleration'][0,:,-1,:],3,axis=-1)
    pred_state = np.concatenate([
        x, -y, z, vx, -vy, ax, -ay, -hy
    ],axis=-1)
    return pred_state

def CarlaTracking(dataroot):
    dataset_obj = defaultdict(lambda: defaultdict(dict))
    frames = list(dataroot.glob('*/route*/metadata/tracking/*.pkl'))
    for frame in frames:
        frame = str(frame)
        with open(frame, 'rb') as f:
            track_mem = pickle.load(f)
        frame_idx = frame.split('/')[-1].split('.')[0]
        scene = frame.split('/')[-4]
        dataset_obj[scene]["all"][frame_idx] = track_mem

    frames = list(dataroot.glob('*/route*/metadata/ego/*.pkl'))
    for frame in frames:
        frame = str(frame)
        with open(frame, 'rb') as f:
            track_mem = pickle.load(f)
        frame_idx = frame.split('/')[-1].split('.')[0]
        scene = frame.split('/')[-4]
        dataset_obj[scene]["ego"][frame_idx] = track_mem
    
    return dataset_obj

def agg_ego_data(all_frames):
    agent_data = []
    agent_frame = []
    for frame_idx, frame_info in enumerate(all_frames):
        pred_state = tracklet_to_pred(frame_info, ego=True)
        agent_data.append(pred_state)
        agent_frame.append(frame_idx)

    agent_data_df = pd.DataFrame(
        agent_data,
        columns=["x", "y", "z", "vx", "vy", "ax", "ay", "heading"],
        index=pd.MultiIndex.from_tuples(
            [
                ("ego", idx)
                for idx in agent_frame
            ],
            names=["agent_id", "scene_ts"],
        ),
    )

    agent_metadata = AgentMetadata(
        name="ego",
        agent_type=AgentType.VEHICLE,
        first_timestep=agent_frame[0],
        last_timestep=agent_frame[-1],
        extent=FixedExtent(
            length=all_frames[-1]["size"][1], width=all_frames[-1]["size"][0], height=all_frames[-1]["size"][2]
        ),
    )
    return Agent(
        metadata=agent_metadata,
        data=agent_data_df,
    )

def agg_agent_data(all_frames, agent_info, frame_idx):
    agent_data = []
    agent_frame = []
    Agent_list = []
    for frame_idx, frame_info in enumerate(all_frames):
        for idx in range(frame_info['id'].shape[1]):
            if frame_info["id"][0,idx,0] == agent_info["id"]:
                # two segments of tracking
                if len(agent_frame) > 0 and frame_idx != agent_frame[-1] + 1:
                    agent_data_df = pd.DataFrame(
                        agent_data,
                        columns=["x", "y", "z", "vx", "vy", "ax", "ay", "heading"],
                        index=pd.MultiIndex.from_tuples(
                            [
                                (f'{int(agent_info["id"])}_{len(Agent_list)}', idx)
                                for idx in agent_frame
                            ],
                            names=["agent_id", "scene_ts"],
                        ),
                    )

                    agent_metadata = AgentMetadata(
                        name=f'{int(agent_info["id"])}_{len(Agent_list)}',
                        agent_type=carla_to_trajdata_object_type[agent_info["cls"]],
                        first_timestep=agent_frame[0],
                        last_timestep=agent_frame[-1],
                        extent=FixedExtent(
                            length=agent_info["size"][1], width=agent_info["size"][0], height=agent_info["size"][2]
                        ),
                    )
                    Agent_list.append(Agent(
                        metadata=agent_metadata,
                        data=agent_data_df,
                    ))

                    agent_data = []
                    agent_frame = []


                pred_state = tracklet_to_pred(frame_info)[idx]
                agent_data.append(pred_state)
                agent_frame.append(frame_idx)

    agent_data_df = pd.DataFrame(
        agent_data,
        columns=["x", "y", "z", "vx", "vy", "ax", "ay", "heading"],
        index=pd.MultiIndex.from_tuples(
            [
                (str(int(agent_info["id"])), idx)
                for idx in agent_frame
            ],
            names=["agent_id", "scene_ts"],
        ),
    )

    agent_metadata = AgentMetadata(
        name=str(int(agent_info["id"])),
        agent_type=carla_to_trajdata_object_type[agent_info["cls"]],
        first_timestep=agent_frame[0],
        last_timestep=agent_frame[-1],
        extent=FixedExtent(
            length=agent_info["size"][1], width=agent_info["size"][0], height=agent_info["size"][2]
        ),
    )
    
    Agent_list.append(Agent(
                            metadata=agent_metadata,
                            data=agent_data_df,
                        ))
    return Agent_list             



class CarlaDataset(RawDataset):
    def compute_metadata(self, env_name: str, data_dir: str) -> EnvMetadata:
        # We're using the nuScenes prediction challenge split here.
        # See https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/prediction/splits.py
        # for full details on how the splits are obtained below.
        all_scene_splits: Dict[str, List[str]] = create_splits_scenes(data_dir)

        train_scenes: List[str] = deepcopy(all_scene_splits["train"])
        NUM_IN_TRAIN_VAL = round(len(train_scenes)*0.25)
        all_scene_splits["train"] = train_scenes[NUM_IN_TRAIN_VAL:]
        all_scene_splits["train_val"] = train_scenes[:NUM_IN_TRAIN_VAL]

        if env_name == 'carla':
            carla_scene_splits: Dict[str, List[str]] = {
                k: all_scene_splits[k] for k in ["train", "train_val", "val"]
            }

            # nuScenes possibilities are the Cartesian product of these
            dataset_parts: List[Tuple[str, ...]] = [
                ("train", "train_val", "val"),
            ]
        else:
            raise ValueError(f"Unknown nuScenes environment name: {env_name}")

        self.scene_splits = carla_scene_splits

        # Inverting the dict from above, associating every scene with its data split.
        carla_scene_split_map: Dict[str, str] = {
            v_elem: k for k, v in carla_scene_splits.items() for v_elem in v
        }
        return EnvMetadata(
            name=env_name,
            data_dir=data_dir,
            dt=nusc_utils.NUSC_DT,
            parts=dataset_parts,
            scene_split_map=carla_scene_split_map,
            # The location names should match the map names used in
            # the unified data cache.
            map_locations=tuple([]),
        )

    def load_dataset_obj(self, verbose: bool = False) -> None:
        if verbose:
            print(f"Loading {self.name} dataset...", flush=True)

        self.dataset_obj = CarlaTracking(
            dataroot=self.metadata.data_dir
        )

    def _get_matching_scenes_from_obj(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[SceneMetadata]:
        all_scenes_list: List[CarlaSceneRecord] = list()

        scenes_list: List[SceneMetadata] = list()
        for idx, scene_record in enumerate(self.dataset_obj):
            scene_name: str = scene_record
            scene_location: str = re.match('.*(Town\d+)', scene_record)[1]
            scene_split: str = self.metadata.scene_split_map[scene_name]
            scene_length: int = len(self.dataset_obj[scene_record])

            # Saving all scene records for later caching.
            all_scenes_list.append(
                CarlaSceneRecord(
                    scene_name, scene_location, scene_length, idx
                )
            )

            if scene_split in scene_tag:

                scene_metadata = SceneMetadata(
                    env_name=self.metadata.name,
                    name=scene_name,
                    dt=self.metadata.dt,
                    raw_data_idx=idx,
                )
                scenes_list.append(scene_metadata)

        self.cache_all_scenes_list(env_cache, all_scenes_list)
        return scenes_list

    def _get_matching_scenes_from_cache(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[Scene]:
        all_scenes_list: List[CarlaSceneRecord] = env_cache.load_env_scenes_list(
            self.name
        )

        scenes_list: List[SceneMetadata] = list()
        for scene_record in all_scenes_list:
            (
                scene_name,
                scene_location,
                scene_length,
                data_idx,
            ) = scene_record
            scene_split: str = self.metadata.scene_split_map[scene_name]

            if scene_split in scene_tag:
                scene_metadata = Scene(
                    self.metadata,
                    scene_name,
                    scene_location,
                    scene_split,
                    scene_length,
                    data_idx,
                    None,  # This isn't used if everything is already cached.
                )
                scenes_list.append(scene_metadata)

        return scenes_list

    def get_scene(self, scene_info: SceneMetadata) -> Scene:
        _, route_name, _, data_idx = scene_info

        scene_record = sorted(self.dataset_obj[route_name])
        scene_name: str = route_name
        scene_location: str = re.match('.*(Town\d+)', route_name)[1]
        scene_split: str = self.metadata.scene_split_map[scene_name]
        scene_length: int = len(self.dataset_obj[route_name]['ego'])

        return Scene(
            self.metadata,
            scene_name,
            scene_location,
            scene_split,
            scene_length,
            data_idx,
            scene_record,
        )

    def get_agent_info(
        self, scene: Scene, cache_path: Path, cache_class: Type[SceneCache]
    ) -> Tuple[List[AgentMetadata], List[List[AgentMetadata]]]:
        ego_agent_info: AgentMetadata = AgentMetadata(
            name="ego",
            agent_type=AgentType.VEHICLE,
            first_timestep=0,
            last_timestep=scene.length_timesteps - 1,
            extent=FixedExtent(length=4.084, width=1.730, height=1.562), #TODO: (yulong) replace with carla ego
        )

        agent_presence: List[List[AgentMetadata]] = [
            [ego_agent_info] for _ in range(scene.length_timesteps)
        ]

        agent_data_list: List[pd.DataFrame] = list()
        existing_agents: Dict[str, AgentMetadata] = dict()

        all_frames = [self.dataset_obj[scene.name]["all"][key] for key in sorted(self.dataset_obj[scene.name]["all"])]
        
        # frame_idx_dict = {
        #     frame_dict: idx for idx, frame_dict in enumerate(all_frames)
        # }
        for frame_idx, frame_info in enumerate(all_frames):
            for idx in range(frame_info['id'].shape[1]):
                if str(int(frame_info["id"][0,idx,0])) in existing_agents:
                    continue

                agent_info = {"id": frame_info["id"][0,idx,0],
                              "cls": frame_info["cls"][0,idx,:].argmax(),
                               "size": frame_info["size"][0,idx,:] }
                # if not agent_info["next"]:
                #     # There are some agents with only a single detection to them, we don't care about these.
                #     continue

                agent_list: List[Agent] = agg_agent_data(
                    all_frames, agent_info, frame_idx
                )
                for agent in agent_list:
                    for scene_ts in range(
                        agent.metadata.first_timestep, agent.metadata.last_timestep + 1
                    ):
                        agent_presence[scene_ts].append(agent.metadata)

                    existing_agents[agent.name] = agent.metadata

                    agent_data_list.append(agent.data)

        ego_all_frames = [self.dataset_obj[scene.name]["ego"][key] for key in sorted(self.dataset_obj[scene.name]["ego"])]

        ego_agent: Agent = agg_ego_data(ego_all_frames)
        agent_data_list.append(ego_agent.data)

        agent_list: List[AgentMetadata] = [ego_agent_info] + list(
            existing_agents.values()
        )

        cache_class.save_agent_data(pd.concat(agent_data_list), cache_path, scene)

        return agent_list, agent_presence


    def cache_maps(
        self,
        cache_path: Path,
        map_cache_class: Type[SceneCache],
        map_params: Dict[str, Any],
    ) -> None:
        
        map_api = MapAPI(cache_path)
        for carla_town in [f"Town0{x}" for x in range(1, 8)] + ["Town10", "Town10HD"]: # ["main"]:
            vec_map = map_api.get_map(
                f"drivesim:main" if carla_town == "main" else f"carla:{carla_town}",
                incl_road_lanes=True,
                incl_road_areas=True,
                incl_ped_crosswalks=True,
                incl_ped_walkways=True,
            )
            map_cache_class.finalize_and_cache_map(cache_path, vec_map, map_params)

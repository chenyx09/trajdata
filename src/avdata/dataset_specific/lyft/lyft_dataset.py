from math import ceil
from pathlib import Path
from random import Random
from typing import List, Optional, Tuple, Type

import numpy as np
import pandas as pd
from l5kit.data import ChunkedDataset, labels, LocalDataManager
from l5kit.configs.config import load_metadata
from l5kit.rasterization import RenderContext
from scipy.stats import mode

from avdata.caching import EnvCache, SceneCache
from avdata.data_structures import AgentMetadata, EnvMetadata, SceneMetadata, SceneTag
from avdata.data_structures.agent import Agent, AgentType, FixedExtent
from avdata.data_structures.map import Map, MapMetadata
from avdata.dataset_specific.lyft import lyft_utils
from avdata.dataset_specific.lyft.rasterizer import MapSemanticRasterizer
from avdata.dataset_specific.raw_dataset import RawDataset
from avdata.dataset_specific.scene_records import LyftSceneRecord


class LyftDataset(RawDataset):
    def compute_metadata(self, env_name: str, data_dir: str) -> EnvMetadata:
        if env_name == "lyft_sample":
            dataset_parts: List[Tuple[str, ...]] = [
                ("mini_train", "mini_val"),
                ("palo_alto",),
            ]
            # Using seeded randomness to assign 80% of scenes to "mini_train" and 20% to "mini_val"
            rng = Random(0)
            scene_split = ["mini_train"] * 80 + ["mini_val"] * 20
            rng.shuffle(scene_split)

            scene_split_map = {
                f"scene-{idx:04d}": scene_split[idx] for idx in range(len(scene_split))
            }
        elif env_name == "lyft":
            pass

        return EnvMetadata(
            name=env_name,
            data_dir=data_dir,
            dt=lyft_utils.LYFT_DT,
            parts=dataset_parts,
            scene_split_map=scene_split_map,
        )

    def load_dataset_obj(self) -> None:
        print(f"Loading {self.name} dataset...", flush=True)

        self.dataset_obj = ChunkedDataset(str(self.metadata.data_dir)).open()

    def _get_matching_scenes_from_obj(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[SceneMetadata]:
        all_scenes_list: List[LyftSceneRecord] = list()

        scenes_list: List[SceneMetadata] = list()
        all_scene_frames = self.dataset_obj.scenes["frame_index_interval"]
        for idx in range(all_scene_frames.shape[0]):
            scene_name: str = f"scene-{idx:04d}"
            scene_split: str = self.metadata.scene_split_map[scene_name]
            scene_length: int = (
                all_scene_frames[idx, 1] - all_scene_frames[idx, 0]
            ).item()  # Doing .item() otherwise it'll be a numpy.int64

            # Saving all scene records for later caching.
            all_scenes_list.append(LyftSceneRecord(scene_name, scene_length))

            if scene_split in scene_tag and scene_desc_contains is None:
                scene_metadata = SceneMetadata(
                    self.metadata,
                    scene_name,
                    "palo_alto",
                    scene_split,
                    scene_length,
                    all_scene_frames[idx],
                )
                scenes_list.append(scene_metadata)

        self.cache_all_scenes_list(env_cache, all_scenes_list)
        return scenes_list

    def _get_matching_scenes_from_cache(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[SceneMetadata]:
        all_scenes_list: List[LyftSceneRecord] = env_cache.load_env_scenes_list(
            self.name
        )

        scenes_list: List[SceneMetadata] = list()
        for scene_record in all_scenes_list:
            scene_name, scene_length = scene_record
            scene_split: str = self.metadata.scene_split_map[scene_name]

            if scene_split in scene_tag and scene_desc_contains is None:
                scene_metadata = SceneMetadata(
                    self.metadata,
                    scene_name,
                    "palo_alto",
                    scene_split,
                    scene_length,
                    None,  # This isn't used if everything is already cached.
                )
                scenes_list.append(scene_metadata)

        return scenes_list

    def get_agent_info(
        self, scene_info: SceneMetadata, cache_path: Path, cache_class: Type[SceneCache]
    ) -> Tuple[List[AgentMetadata], List[List[AgentMetadata]]]:
        ego_agent_info: AgentMetadata = AgentMetadata(
            name="ego",
            agent_type=AgentType.VEHICLE,
            first_timestep=0,
            last_timestep=scene_info.length_timesteps - 1,
            extent=FixedExtent(length=4.869, width=1.852, height=1.476),
        )

        agent_list: List[AgentMetadata] = [ego_agent_info]
        agent_presence: List[List[AgentMetadata]] = [
            [ego_agent_info] for _ in range(scene_info.length_timesteps)
        ]
        # TODO(bivanovic): Handle missing timesteps via linear interpolation
        agent_data_list: List[pd.DataFrame] = list()

        ego_agent: Agent = lyft_utils.agg_ego_data(self.dataset_obj, scene_info)
        agent_data_list.append(ego_agent.data)

        scene_frame_start = scene_info.data_access_info[0]
        scene_frame_end = scene_info.data_access_info[1]

        agent_indices = self.dataset_obj.frames[scene_frame_start:scene_frame_end][
            "agent_index_interval"
        ]
        agent_start_idx = agent_indices[0, 0]
        agent_end_idx = agent_indices[-1, 1]

        lyft_agents = self.dataset_obj.agents[agent_start_idx:agent_end_idx]
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
            agent_type: AgentType = lyft_utils.lyft_type_to_unified_type(mode_type)

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
                / lyft_utils.LYFT_DT
            )

            agent_metadata = AgentMetadata(
                name=str(agent_id),
                agent_type=agent_type,
                first_timestep=start_frame,
                last_timestep=last_frame,
            )

            agent_list.append(agent_metadata)
            for frame in agent_data_df.index:
                agent_presence[frame].append(agent_metadata)

            agent_data_df["agent_id"] = agent_metadata.name
            agent_data_df.set_index("agent_id", append=True, inplace=True)

            # For now only saving non-prob columns since Lyft is effectively one-hot (see https://arxiv.org/abs/2104.12446)
            agent = Agent(agent_metadata, agent_data_df.loc[:, final_cols].swaplevel())
            agent_data_list.append(agent.data)

        cache_class.save_agent_data(pd.concat(agent_data_list), cache_path, scene_info)

        return agent_list, agent_presence

    def cache_maps(self, cache_path: Path, map_cache_class: Type[SceneCache], resolution: int = 2) -> None:
        # We have to do this ../.. stuff because the data_dir for lyft is scenes/sample.zarr
        dm = LocalDataManager((self.metadata.data_dir / ".." / "..").resolve())
        
        world_right, world_top = self.dataset_obj.agents["centroid"].max(axis=0) * 1.15
        world_left, world_bottom = self.dataset_obj.agents["centroid"].min(axis=0) * 1.15
        
        world_center: np.ndarray = np.array([(world_left + world_right)/2, (world_bottom + world_top)/2])
        raster_size_px: np.ndarray = np.array([ceil((world_right - world_left) * resolution), ceil((world_top - world_bottom) * resolution)])

        dataset_meta = load_metadata(dm.require("meta.json"))
        world_to_ecef = np.array(dataset_meta["world_to_ecef"], dtype=np.float64)
        semantic_map_filepath = dm.require("semantic_map/semantic_map.pb")
        render_context = RenderContext(
            raster_size_px=raster_size_px,
            pixel_size_m=np.array([1/resolution, 1/resolution]),
            center_in_raster_ratio=np.array([0.5, 0.5]),
            set_origin_to_bottom=True,
        )

        rasterizer = MapSemanticRasterizer(render_context,
                                           semantic_map_filepath,
                                           world_to_ecef)
        
        map_data: np.ndarray = rasterizer.render_semantic_map(world_center)
        
        map_info: MapMetadata = MapMetadata(
            name='palo_alto',
            shape=map_data.shape,
            layers=["lane_area", "lane_lines", "ped_walkways"],
            resolution=resolution,
        )
        
        map_obj: Map = Map(map_info, map_data)
        map_cache_class.cache_map(cache_path, map_obj, self.name)

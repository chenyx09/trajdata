from copy import deepcopy
from typing import Any, Dict, List, Union

import numpy as np

from avdata import filtering
from avdata.caching.df_cache import DataFrameCache
from avdata.data_structures.agent import AgentMetadata
from avdata.data_structures.batch import AgentBatch, agent_collate_fn
from avdata.data_structures.batch_element import AgentBatchElement
from avdata.data_structures.scene import SceneTimeAgent
from avdata.data_structures.scene_metadata import SceneMetadata
from avdata.dataset import UnifiedDataset
from avdata.simulation.sim_cache import SimulationCache
from avdata.simulation.sim_df_cache import SimulationDataFrameCache


class SimulationScene:
    def __init__(
        self,
        env_name: str,
        scene_name: str,
        scene_info: SceneMetadata,
        dataset: UnifiedDataset,
        init_timestep: int = 0,
        freeze_agents: bool = True,
        return_dict: bool = False,
    ) -> None:
        self.env_name: str = env_name
        self.scene_name: str = scene_name
        self.scene_info: SceneMetadata = deepcopy(scene_info)
        self.dataset: UnifiedDataset = dataset
        self.init_scene_ts: int = init_timestep
        self.freeze_agents: bool = freeze_agents
        self.return_dict: bool = return_dict

        self.scene_ts: int = self.init_scene_ts

        agents_present: List[AgentMetadata] = self.scene_info.agent_presence[
            self.scene_ts
        ]
        self.agents: List[AgentMetadata] = filtering.agent_types(
            agents_present, self.dataset.no_types, self.dataset.only_types
        )
        self.scene_info.agents = self.agents

        # Note this order of operations is important, we first instantiate
        # the cache with the copied scene_info + modified agents list.
        # Then, we change the env_name and etc later during finalization
        # (if we did it earlier then the cache would go looking inside
        # the sim folder for scene data rather than the original scene
        # data location).
        if self.dataset.cache_class == DataFrameCache:
            self.cache: SimulationCache = SimulationDataFrameCache(
                dataset.cache_path, self.scene_info, init_timestep
            )

        if self.freeze_agents:
            self.scene_info.agent_presence = self.scene_info.agent_presence[
                : self.init_scene_ts + 1
            ]

    def reset(self) -> Union[AgentBatch, Dict[str, Any]]:
        self.scene_ts: int = self.init_scene_ts
        return self.get_obs()

    def step(
        self, new_xyh_dict: Dict[str, np.ndarray]
    ) -> Union[AgentBatch, Dict[str, Any]]:
        self.scene_ts += 1

        self.cache.append_state(new_xyh_dict)

        if not self.freeze_agents:
            agents_present: List[AgentMetadata] = self.scene_info.agent_presence[
                self.scene_ts
            ]
            self.agents: List[AgentMetadata] = filtering.agent_types(
                agents_present, self.dataset.no_types, self.dataset.only_types
            )

            self.scene_info.agent_presence[self.scene_ts] = self.agents
        else:
            self.scene_info.agent_presence.append(self.agents)

        return self.get_obs()

    def get_obs(self) -> Union[AgentBatch, Dict[str, Any]]:
        agent_data_list: List[AgentBatchElement] = list()
        for agent in self.agents:
            scene_time_agent = SceneTimeAgent(
                self.scene_info, self.scene_ts, self.agents, agent, self.cache
            )

            agent_data_list.append(
                AgentBatchElement(
                    self.cache,
                    -1,  # Not used
                    scene_time_agent,
                    history_sec=self.dataset.history_sec,
                    future_sec=self.dataset.future_sec,
                    agent_interaction_distances=self.dataset.agent_interaction_distances,
                    incl_robot_future=False,
                    incl_map=self.dataset.incl_map,
                    map_params=self.dataset.map_params,
                    standardize_data=self.dataset.standardize_data,
                )
            )

            # Need to do reset for each agent since each
            # AgentBatchElement transforms (standardizes) the cache.
            self.cache.reset()

        return agent_collate_fn(agent_data_list, return_dict=self.return_dict)

    def finalize(self) -> None:
        for agent in self.agents:
            agent.last_timestep = self.scene_ts

        self.scene_info.length_timesteps = self.scene_ts + 1

        self.scene_info.agent_presence = self.scene_info.agent_presence[
            : self.scene_ts + 1
        ]

        self.scene_info.env_metadata.name = self.env_name
        self.scene_info.env_name = self.env_name
        self.scene_info.name = self.scene_name

    def save(self) -> None:
        self.dataset.env_cache.save_scene_metadata(self.scene_info)
        self.cache.save_sim_scene(self.scene_info)

from typing import List

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
        self, scene_info: SceneMetadata, dataset: UnifiedDataset, init_timestep: int = 0
    ) -> None:
        self.scene_info: SceneMetadata = scene_info
        self.dataset: UnifiedDataset = dataset
        self.init_scene_ts: int = init_timestep

        if self.dataset.cache_class == DataFrameCache:
            self.cache: SimulationCache = SimulationDataFrameCache(
                dataset.cache_path, scene_info, init_timestep
            )

        self.scene_ts: int = self.init_scene_ts

    def reset(self) -> AgentBatch:
        self.scene_ts: int = self.init_scene_ts

        agent_data_list: List[AgentBatchElement] = list()

        agents_present: List[AgentMetadata] = self.scene_info.agent_presence[
            self.scene_ts
        ]
        filtered_agents: List[AgentMetadata] = filtering.agent_types(
            agents_present, self.dataset.no_types, self.dataset.only_types
        )
        for agent in filtered_agents:
            scene_time_agent = SceneTimeAgent.from_cache(
                self.scene_info,
                self.scene_ts,
                agent.name,
                self.cache,
                only_types=self.dataset.only_types,
                no_types=self.dataset.no_types,
                incl_robot_future=False,
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

        return agent_collate_fn(agent_data_list)

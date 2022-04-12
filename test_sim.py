from collections import defaultdict
from typing import Dict, List

import numpy as np
from tqdm import trange

from avdata import AgentBatch, AgentType, UnifiedDataset
from avdata.caching.df_cache import DataFrameCache
from avdata.data_structures.scene_metadata import SceneMetadata
from avdata.simulation import SimulationScene
from avdata.visualization.vis import plot_agent_batch


# @profile
def main():
    dataset = UnifiedDataset(
        desired_data=["nusc_mini"],
        only_types=[AgentType.VEHICLE],
        agent_interaction_distances=defaultdict(lambda: 50.0),
        incl_map=True,
        map_params={"px_per_m": 2, "map_size_px": 224},
    )

    sim_env_name = "nusc_mini_sim"
    all_sim_scenes: List[SceneMetadata] = list()
    desired_scene: SceneMetadata
    for idx, desired_scene in enumerate(dataset.scene_index):
        sim_scene: SimulationScene = SimulationScene(
            env_name=sim_env_name,
            scene_name=f"sim_scene-{idx:04d}",
            scene_info=desired_scene,
            dataset=dataset,
            init_timestep=10,
            freeze_agents=True,
        )

        obs: AgentBatch = sim_scene.reset()
        for t in trange(1, 11):
            new_xyh_dict: Dict[str, np.ndarray] = {
                agent.name: obs.curr_agent_state[idx, [0, 1, -1]].numpy()
                + np.array([t, 0, t / 100])
                for idx, agent in enumerate(sim_scene.agents)
            }
            obs = sim_scene.step(new_xyh_dict)

        plot_agent_batch(obs, 0, show=False, close=False)
        plot_agent_batch(obs, 1, show=False, close=False)
        plot_agent_batch(obs, 2, show=False, close=False)
        plot_agent_batch(obs, 3, show=True, close=True)

        sim_scene.finalize()
        sim_scene.save()

        all_sim_scenes.append(sim_scene.scene_info)

    dataset.env_cache.save_env_scenes_list(sim_env_name, all_sim_scenes)


if __name__ == "__main__":
    main()

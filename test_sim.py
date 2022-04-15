from collections import defaultdict
from tabnanny import verbose
from typing import Dict, List

import numpy as np
from tqdm import trange

from avdata import AgentBatch, AgentType, UnifiedDataset
from avdata.data_structures.scene_metadata import SceneMetadata
from avdata.simulation import SimulationScene
from avdata.visualization.vis import plot_agent_batch


# @profile
def main():
    dataset = UnifiedDataset(
        desired_data=["nusc_mini-mini_val"],
        only_types=[AgentType.VEHICLE],
        agent_interaction_distances=defaultdict(lambda: 50.0),
        incl_map=True,
        map_params={
            "px_per_m": 2,
            "map_size_px": 224,
            "offset_frac_xy": (0.0, 0.0),
            "return_rgb": True,
        },
        verbose=True,
        desired_dt=0.05,
    )

    sim_env_name = "lyft_sample_sim"
    all_sim_scenes: List[SceneMetadata] = list()
    desired_scene: SceneMetadata
    for idx, desired_scene in enumerate(dataset.scene_index):
        sim_scene: SimulationScene = SimulationScene(
            env_name=sim_env_name,
            scene_name=f"sim_scene-{idx:04d}",
            scene_info=desired_scene,
            dataset=dataset,
            init_timestep=0,
            freeze_agents=True,
        )

        obs: AgentBatch = sim_scene.reset()
        for t in trange(1, 401):
            new_xyh_dict: Dict[str, np.ndarray] = dict()
            for idx, agent_name in enumerate(obs.agent_name):
                curr_yaw = obs.curr_agent_state[idx, -1]
                curr_pos = obs.curr_agent_state[idx, :2]
                world_from_agent = np.array(
                    [
                        [np.cos(curr_yaw), np.sin(curr_yaw)],
                        [-np.sin(curr_yaw), np.cos(curr_yaw)],
                    ]
                )
                next_state = np.zeros((3,))
                if obs.agent_fut_len[idx] < 1:
                    next_state[:2] = curr_pos
                    yaw_ac = 0
                else:
                    next_state[:2] = (
                        obs.agent_fut[idx, 0, :2] @ world_from_agent + curr_pos
                    )
                    yaw_ac = np.arctan2(
                        obs.agent_fut[idx, 0, -2], obs.agent_fut[idx, 0, -1]
                    )

                next_state[2] = curr_yaw + yaw_ac
                new_xyh_dict[agent_name] = next_state

            obs = sim_scene.step(new_xyh_dict)

        plot_agent_batch(
            obs, 0, dataset.map_params["offset_frac_xy"], show=False, close=False
        )
        plot_agent_batch(
            obs, 1, dataset.map_params["offset_frac_xy"], show=False, close=False
        )
        plot_agent_batch(
            obs, 2, dataset.map_params["offset_frac_xy"], show=False, close=False
        )
        plot_agent_batch(
            obs, 3, dataset.map_params["offset_frac_xy"], show=True, close=True
        )

        sim_scene.finalize()
        sim_scene.save()

        all_sim_scenes.append(sim_scene.scene_info)

    dataset.env_cache.save_env_scenes_list(sim_env_name, all_sim_scenes)


if __name__ == "__main__":
    main()

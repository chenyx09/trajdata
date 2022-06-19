from typing import Dict # Just for type annotations

import numpy as np

from avdata import AgentBatch, UnifiedDataset
from avdata.data_structures.scene_metadata import Scene # Just for type annotations
from avdata.simulation import SimulationScene

dataset = UnifiedDataset(desired_data=["nusc_mini"])

desired_scene: Scene = dataset.get_scene(scene_idx=0)
sim_scene = SimulationScene(
    env_name="nusc_mini_sim",
    scene_name="sim_scene",
    scene=desired_scene,
    dataset=dataset,
    init_timestep=0,
    freeze_agents=True,
)

obs: AgentBatch = sim_scene.reset()
for t in range(1, sim_scene.scene_info.length_timesteps):
    new_xyh_dict: Dict[str, np.ndarray] = dict()

    # Everything inside the forloop just sets
    # agents' next states to their current ones.
    for idx, agent_name in enumerate(obs.agent_name):
        curr_yaw = obs.curr_agent_state[idx, -1]
        curr_pos = obs.curr_agent_state[idx, :2]

        next_state = np.zeros((3,))
        next_state[:2] = curr_pos
        next_state[2] = curr_yaw
        new_xyh_dict[agent_name] = next_state

    obs = sim_scene.step(new_xyh_dict)
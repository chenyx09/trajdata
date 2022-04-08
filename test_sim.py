from collections import defaultdict
from typing import Dict

import numpy as np
from tqdm import trange

from avdata import AgentBatch, AgentType, UnifiedDataset
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

    desired_scene: SceneMetadata = dataset.scene_index[0]
    sim_scene: SimulationScene = SimulationScene(
        desired_scene, dataset, init_timestep=0
    )

    obs: AgentBatch = sim_scene.reset()
    plot_agent_batch(obs, 0, show=False, close=False)
    plot_agent_batch(obs, 1, show=False, close=False)
    plot_agent_batch(obs, 2, show=False, close=False)
    plot_agent_batch(obs, 3, show=True, close=True)
    for _ in trange(10):
        new_pos_dict: Dict[str, np.ndarray] = dict()
        obs = sim_scene.step(new_pos_dict)

    sim_scene.save("sim_scene-0001")


if __name__ == "__main__":
    main()

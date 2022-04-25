from typing import List, Tuple

import numpy as np


class DataIndex:
    """The data index is effectively a big list of tuples taking the form:

    (env_name: str, scene_name: str, timestep: int, agent_name: str)
    """

    def __init__(self, data_index: List[Tuple]) -> None:
        self.len = len(data_index)

        # TODO(bivanovic): Handle scene data index too (which doesn't have agent_names)
        # TODO: Something is very slow here... Might be better to just numpy array all
        #       of this and then astype it?
        env_names, scene_names, timesteps, agent_names = zip(*data_index)

        self.env_names = np.array(env_names).astype(np.string_)
        self.scene_names = np.array(scene_names).astype(np.string_)
        self.timesteps = np.array(timesteps)
        self.agent_names = np.array(agent_names).astype(np.string_)

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index: int) -> Tuple:
        env_name: str = str(self.env_names[index], encoding="utf-8")
        scene_name: str = str(self.scene_names[index], encoding="utf-8")
        timestep: int = self.timesteps[index].item()
        agent_name: str = str(self.agent_names[index], encoding="utf-8")

        return (env_name, scene_name, timestep, agent_name)

import pandas as pd


class Augmentation:
    def __init__(self) -> None:
        raise NotImplementedError()
    
    def apply(self, agent_data: pd.DataFrame) -> None:
        raise NotImplementedError()
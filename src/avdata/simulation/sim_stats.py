from typing import List, Tuple

import numpy as np
import pandas as pd


class SimStatistic:
    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, scene_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()


class VelocityHistogram(SimStatistic):
    def __init__(self, bins: List[int]) -> None:
        super().__init__("vel_hist")
        self.bins = bins

    def __call__(self, scene_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        velocities: np.ndarray = np.linalg.norm(scene_df[["vx", "vy"]], axis=1)

        return np.histogram(velocities, bins=self.bins)


class LongitudinalAccHistogram(SimStatistic):
    def __init__(self, bins: List[int]) -> None:
        super().__init__("lon_acc_hist")
        self.bins = bins

    def __call__(self, scene_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        accels: np.ndarray = np.linalg.norm(scene_df[["ax", "ay"]], axis=1)
        lon_accels: np.ndarray = accels * np.cos(scene_df["heading"])

        return np.histogram(lon_accels, bins=self.bins)


class LateralAccHistogram(SimStatistic):
    def __init__(self, bins: List[int]) -> None:
        super().__init__("lat_acc_hist")
        self.bins = bins

    def __call__(self, scene_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        accels: np.ndarray = np.linalg.norm(scene_df[["ax", "ay"]], axis=1)
        lat_accels: np.ndarray = accels * np.sin(scene_df["heading"])

        return np.histogram(lat_accels, bins=self.bins)


class JerkHistogram(SimStatistic):
    def __init__(self, bins: List[int], dt: float) -> None:
        super().__init__("jerk_hist")
        self.bins = bins
        self.dt = dt

    def __call__(self, scene_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        accels: np.ndarray = np.linalg.norm(scene_df[["ax", "ay"]], axis=1)
        jerk: np.ndarray = np.gradient(accels, self.dt)

        return np.histogram(jerk, bins=self.bins)

import contextlib
import sqlite3
from typing import Iterable

import pandas as pd

from avdata.caching.base_cache import BaseCache
from avdata.data_structures.scene_metadata import SceneMetadata


class SQLiteCache(BaseCache):
    def __init__(self, cache_location: str) -> None:
        super().__init__(cache_location)

    def load_single_agent_data(
        self, agent_id: str, scene_info: SceneMetadata
    ) -> pd.DataFrame:
        with contextlib.closing(
            sqlite3.connect(
                self.path / scene_info.env_name / scene_info.name / "agent_data.db"
            )
        ) as conn:
            data_df = pd.read_sql_query(
                f"SELECT * FROM agent_data WHERE agent_id=?",
                conn,
                params=(agent_id,),
                index_col="scene_ts",
            )

        return data_df

    def load_multiple_agent_data(
        self, agent_ids: Iterable[str], scene_info: SceneMetadata
    ) -> pd.DataFrame:
        with contextlib.closing(
            sqlite3.connect(
                self.path / scene_info.env_name / scene_info.name / "agent_data.db"
            )
        ) as conn:
            data_df = pd.read_sql_query(
                f"SELECT * FROM agent_data WHERE agent_id IN ({','.join('?'*len(agent_ids))})",
                conn,
                params=tuple(agent_ids),
                index_col=["agent_id", "scene_ts"],
            )

        return data_df

    def load_all_agent_data(
        self, scene_info: SceneMetadata
    ) -> pd.DataFrame:
        with contextlib.closing(
            sqlite3.connect(
                self.path / scene_info.env_name / scene_info.name / "agent_data.db"
            )
        ) as conn:
            data_df = pd.read_sql_table(
                "agent_data",
                conn,
                index_col=["agent_id", "scene_ts"],
            )

        return data_df

    def load_agent_xy_at_time(
        self, scene_ts: int, scene_info: SceneMetadata
    ) -> pd.DataFrame:
        with contextlib.closing(
            sqlite3.connect(
                self.path / scene_info.env_name / scene_info.name / "agent_data.db"
            )
        ) as conn:
            data_df = pd.read_sql_query(
                "SELECT agent_id,x,y FROM agent_data WHERE scene_ts=?",
                conn,
                params=(scene_ts,),
                index_col="agent_id",
            )

        return data_df

    def load_data_between_times(
        self, from_ts: int, to_ts: int, scene_info: SceneMetadata
    ) -> pd.DataFrame:
        with contextlib.closing(
            sqlite3.connect(
                self.path / scene_info.env_name / scene_info.name / "agent_data.db"
            )
        ) as conn:
            all_agents_df = pd.read_sql_query(
                "SELECT * FROM agent_data WHERE scene_ts BETWEEN ? AND ?",
                conn,
                params=(from_ts, to_ts),
                index_col=["agent_id", "scene_ts"],
            )

        return all_agents_df

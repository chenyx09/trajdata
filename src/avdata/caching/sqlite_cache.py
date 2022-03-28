import contextlib
import sqlite3
from typing import Iterable
from pathlib import Path
import pandas as pd

from avdata.caching import EnvCache, SceneCache
from avdata.data_structures.scene_metadata import SceneMetadata

DB_SCHEMA = """
agent_id TEXT NOT NULL,
scene_ts INTEGER NOT NULL,
x REAL NOT NULL,
y REAL NOT NULL,
vx REAL NOT NULL,
vy REAL NOT NULL,
ax REAL NOT NULL,
ay REAL NOT NULL,
heading REAL NOT NULL
"""


class SQLiteCache(SceneCache):
    def __init__(self, cache_path: Path, scene_info: SceneMetadata) -> None:
        super().__init__(cache_path, scene_info)

        self.agent_data_path: Path = self.scene_dir / "agent_data.db"

    def save_agent_data(
        self, agent_data: pd.DataFrame
    ) -> None:
        with contextlib.closing(
            sqlite3.connect(
                self.agent_data_path
            )
        ) as connection:
            cursor = connection.cursor()
            cursor.execute(f"CREATE TABLE IF NOT EXISTS agent_data ({DB_SCHEMA})")
            agent_data.to_sql(
                name="agent_data",
                con=connection,
                if_exists="replace",
                index=True,
                index_label="scene_ts",
            )

    def load_single_agent_data(
        self, agent_id: str
    ) -> pd.DataFrame:
        with contextlib.closing(
            sqlite3.connect(
                self.agent_data_path
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
        self, agent_ids: Iterable[str]
    ) -> pd.DataFrame:
        with contextlib.closing(
            sqlite3.connect(
                self.agent_data_path
            )
        ) as conn:
            data_df = pd.read_sql_query(
                f"SELECT * FROM agent_data WHERE agent_id IN ({','.join('?'*len(agent_ids))})",
                conn,
                params=tuple(agent_ids),
                index_col=["agent_id", "scene_ts"],
            )

        return data_df

    def load_all_agent_data(self) -> pd.DataFrame:
        with contextlib.closing(
            sqlite3.connect(
                self.agent_data_path
            )
        ) as conn:
            data_df = pd.read_sql_table(
                "agent_data",
                conn,
                index_col=["agent_id", "scene_ts"],
            )

        return data_df

    def load_agent_xy_at_time(
        self, scene_ts: int
    ) -> pd.DataFrame:
        with contextlib.closing(
            sqlite3.connect(
                self.agent_data_path
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
        self, from_ts: int, to_ts: int
    ) -> pd.DataFrame:
        with contextlib.closing(
            sqlite3.connect(
                self.agent_data_path
            )
        ) as conn:
            all_agents_df = pd.read_sql_query(
                "SELECT * FROM agent_data WHERE scene_ts BETWEEN ? AND ?",
                conn,
                params=(from_ts, to_ts),
                index_col=["agent_id", "scene_ts"],
            )

        return all_agents_df

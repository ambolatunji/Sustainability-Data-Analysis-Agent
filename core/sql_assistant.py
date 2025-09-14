from __future__ import annotations
from typing import Dict
import pandas as pd
import duckdb
from .utils import safe_name


class SQLAssistant:
    def __init__(self, tables: Dict[str, pd.DataFrame]):
        self.tables = tables
        self.conn = duckdb.connect(database=':memory:')
        for name, df in tables.items():
            self.conn.register(safe_name(name), df)

    def refresh(self):
        self.conn = duckdb.connect(database=':memory:')
        for name, df in self.tables.items():
            self.conn.register(safe_name(name), df)

    def describe(self) -> str:
        lines = ["Available tables and columns:"]
        for name, df in self.tables.items():
            cols = ", ".join([f"{c} ({str(t)})" for c, t in zip(df.columns, df.dtypes)])
            lines.append(f"- {safe_name(name)}: {cols}")
        return "\n".join(lines)

    def run(self, sql: str) -> pd.DataFrame:
        return self.conn.execute(sql).df()
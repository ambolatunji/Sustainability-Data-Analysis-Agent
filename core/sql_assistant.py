from __future__ import annotations
import re
from typing import Dict

import pandas as pd
import duckdb


class SQLAssistant:
    def __init__(self, tables: Dict[str, pd.DataFrame]):
        self.tables = tables
        self.conn = duckdb.connect(database=':memory:')
        for name, df in tables.items():
            safe = self._safe_name(name)
            self.conn.register(safe, df)

    def _safe_name(self, name: str) -> str:
        s = re.sub(r"[^A-Za-z0-9_]+", "_", name)
        if re.match(r"^\d", s):
            s = "T_" + s
        return s

    def describe(self) -> str:
        lines = ["Available tables and columns:"]
        for name, df in self.tables.items():
            safe = self._safe_name(name)
            cols = ", ".join([f"{c} ({str(t)})" for c, t in zip(df.columns, df.dtypes)])
            lines.append(f"- {safe}: {cols}")
        return "\n".join(lines)

    def run(self, sql: str) -> pd.DataFrame:
        return self.conn.execute(sql).df()
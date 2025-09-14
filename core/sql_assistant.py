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
            self.conn.register(self._safe_name(name), df)

    def _safe_name(self, name: str) -> str:
        s = re.sub(r"[^A-Za-z0-9_]+", "_", name)
        if re.match(r"^\d", s):
            s = "T_" + s
        return s

    def refresh(self):
        # call this after mutating tables/dtypes
        self.conn = duckdb.connect(database=':memory:')
        for name, df in self.tables.items():
            self.conn.register(self._safe_name(name), df)

    def describe(self) -> str:
        lines = ["Available tables and columns:"]
        for name, df in self.tables.items():
            cols = ", ".join([f"{c} ({str(t)})" for c, t in zip(df.columns, df.dtypes)])
            lines.append(f"- {self._safe_name(name)}: {cols}")
        return "\n".join(lines)

    def _normalize_sql(self, sql: str) -> str:
        # Replace [original table name] with safe names; also tolerate backticks/quotes
        s = sql or ""
        for orig in self.tables.keys():
            safe = self._safe_name(orig)
            pattern_brackets = rf"\[\s*{re.escape(orig)}\s*\]"
            s = re.sub(pattern_brackets, safe, s, flags=re.I)
            s = re.sub(rf"`\s*{re.escape(orig)}\s*`", safe, s, flags=re.I)
            s = re.sub(rf"\"{re.escape(orig)}\"", safe, s, flags=re.I)
        return s

    def run(self, sql: str) -> pd.DataFrame:
        s = (sql or "").strip().rstrip(";")
        if not s or "NO_SQL" in s.upper():
            raise ValueError("Empty/unsupported SQL.")
        s = self._normalize_sql(s)
        # Use .sql(...).df() which always returns a relation
        rel = self.conn.sql(s)
        return rel.df()

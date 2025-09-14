from __future__ import annotations
import re
import pandas as pd
import numpy as np
from functools import lru_cache
from typing import Optional
from sentence_transformers import SentenceTransformer


def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def safe_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]+", "_", name)
    if re.match(r"^\d", s):
        s = "T_" + s
    return s


@lru_cache(maxsize=2)
def load_sbert(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    return SentenceTransformer(model_name)


def resolve_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """If duplicate column names exist, merge them:
    - numeric dups → row-wise sum
    - non-numeric dups → first non-null
    Returns a new DF with unique columns.
    """
    cols = list(df.columns)
    if len(cols) == len(set(cols)):
        return df

    out = {}
    for col in dict.fromkeys(cols):  # preserves order
        dup_idx = [i for i, c in enumerate(cols) if c == col]
        if len(dup_idx) == 1:
            out[col] = df.iloc[:, dup_idx[0]]
            continue
        parts = [df.iloc[:, i] for i in dup_idx]
        # numeric?
        if all(pd.api.types.is_numeric_dtype(p) for p in parts):
            merged = sum(p.fillna(0) for p in parts)
        else:
            merged = parts[0]
            for p in parts[1:]:
                merged = merged.where(merged.notna(), p)
        out[col] = merged
    res = pd.DataFrame(out)
    return res


# Robust date parsing: supports "September 2024", "2024-09", "2024/10", "2024/09/19", DD/MM/YYYY, datetimes
DATE_FORMATS = [
    "%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y",
    "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y",
    "%Y-%m", "%Y/%m",
    "%b %Y", "%B %Y",  # Sep 2024, September 2024
]

def parse_date_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s)
    # Try pandas' inference first
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    # If too many NaT, try explicit formats
    if dt.notna().mean() < 0.7:
        s_str = s.astype(str).str.strip()
        dt = pd.to_datetime(s_str, errors="coerce", dayfirst=True)
        if dt.notna().mean() < 0.7:
            # try formats
            vals = []
            for v in s_str:
                got = pd.NaT
                for fmt in DATE_FORMATS:
                    try:
                        got = pd.to_datetime(v, format=fmt)
                        break
                    except Exception:
                        pass
                vals.append(got)
            dt = pd.to_datetime(vals)
    return dt
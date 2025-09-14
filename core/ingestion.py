from __future__ import annotations
import io
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
from pypdf import PdfReader

from .utils import clean_text, resolve_duplicate_columns, safe_name


@dataclass
class IngestedData:
    tables: Dict[str, pd.DataFrame]
    corpus_docs: List[Tuple[str, str]]  # (doc_id, text)


def _read_pdf(file: io.BytesIO, name: str):
    reader = PdfReader(file)
    docs = []
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        if txt.strip():
            docs.append((f"{name}#page={i+1}", clean_text(txt)))
    return docs, []


def _read_csv(file: io.BytesIO, name: str):
    df = pd.read_csv(file)
    df.columns = [str(c).strip() for c in df.columns]
    df = resolve_duplicate_columns(df)
    return [(f"{name}#table", clean_text(df.to_csv(index=False)))], [df]


def _read_excel(file: io.BytesIO, name: str):
    xl = pd.ExcelFile(file)
    docs, tables = [], []
    for sheet in xl.sheet_names:
        try:
            df = xl.parse(sheet)
        except Exception:
            continue
        df.columns = [str(c).strip() for c in df.columns]
        df = resolve_duplicate_columns(df)
        df.name = f"{name}:{sheet}"
        tables.append(df)
        docs.append((f"{name}#{sheet}", clean_text(df.to_csv(index=False))))
    return docs, tables


def ingest_files(uploaded_files) -> IngestedData:
    tables: Dict[str, pd.DataFrame] = {}
    docs: List[Tuple[str, str]] = []

    for uf in uploaded_files:
        name = uf.name
        ext = (name.split('.')[-1] or "").lower()
        data = uf.read()
        bio = io.BytesIO(data)
        if ext == "pdf":
            corp, tbs = _read_pdf(bio, name)
        elif ext == "csv":
            corp, tbs = _read_csv(bio, name)
        else:  # xlsx/xls/xlsm
            corp, tbs = _read_excel(bio, name)
        docs.extend(corp)
        for i, df in enumerate(tbs):
            key = getattr(df, 'name', f"{name}:table{i+1}")
            # If table key already exists → append and drop exact duplicate rows
            if key in tables:
                base = tables[key]
                # align columns union
                new_cols = list({*base.columns, *df.columns})
                base = base.reindex(columns=new_cols)
                df = df.reindex(columns=new_cols)
                merged = pd.concat([base, df], ignore_index=True)
                merged = merged.drop_duplicates()
                tables[key] = merged
            else:
                if 'Date' in df.columns:
                    try:
                        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                    except Exception:
                        pass
                tables[key] = df
    return IngestedData(tables=tables, corpus_docs=docs)
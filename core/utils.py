from __future__ import annotations
import re
from functools import lru_cache
from typing import Optional

from sentence_transformers import SentenceTransformer


def clean_text(s: str) -> str:
    """Normalize whitespace for robust embedding chunks."""
    return re.sub(r"\s+", " ", s or "").strip()


@lru_cache(maxsize=2)
def load_sbert(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    """Lazy-load and cache the SentenceTransformer model."""
    return SentenceTransformer(model_name)
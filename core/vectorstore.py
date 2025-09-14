from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import faiss

from .utils import load_sbert


@dataclass
class VectorIndex:
    index: faiss.IndexFlatIP
    ids: List[str]
    embeddings: np.ndarray
    model_name: str

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        sbert = load_sbert(self.model_name)
        q = sbert.encode([query], normalize_embeddings=True).astype('float32')
        D, I = self.index.search(q, k)
        out: List[Tuple[str, float]] = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            out.append((self.ids[idx], float(score)))
        return out


def build_faiss_index(docs: List[Tuple[str, str]], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> VectorIndex:
    sbert = load_sbert(model_name)
    texts = [t for _, t in docs]
    ids = [i for i, _ in docs]
    dim = sbert.get_sentence_embedding_dimension()

    if not texts:
        return VectorIndex(index=faiss.IndexFlatIP(dim), ids=[], embeddings=np.zeros((0, dim), dtype='float32'), model_name=model_name)

    X = sbert.encode(texts, normalize_embeddings=True).astype('float32')
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    return VectorIndex(index=index, ids=ids, embeddings=X, model_name=model_name)
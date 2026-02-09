from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np


@dataclass(frozen=True)
class SearchHit:
    doc_id: str
    score: float


class VectorIndex(Protocol):
    dim: int

    def add(self, doc_ids: list[str], vectors: np.ndarray) -> None: ...

    def search(self, query: np.ndarray, top_k: int) -> list[SearchHit]: ...


class BruteForceIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self._doc_ids: list[str] = []
        self._vectors: list[np.ndarray] = []

    def add(self, doc_ids: list[str], vectors: np.ndarray) -> None:
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"Expected vectors shape (n,{self.dim}), got {vectors.shape}")
        for doc_id, vec in zip(doc_ids, vectors, strict=True):
            self._doc_ids.append(doc_id)
            self._vectors.append(vec.astype(np.float32, copy=False))

    def search(self, query: np.ndarray, top_k: int) -> list[SearchHit]:
        if query.shape != (self.dim,):
            raise ValueError(f"Expected query shape ({self.dim},), got {query.shape}")
        if not self._vectors:
            return []
        mat = np.stack(self._vectors, axis=0)
        # Cosine similarity (assumes vectors are already normalized)
        scores = mat @ query.astype(np.float32, copy=False)
        idx = np.argsort(-scores)[:top_k]
        return [SearchHit(doc_id=self._doc_ids[i], score=float(scores[i])) for i in idx]


class FaissIndex:
    def __init__(self, dim: int):
        self.dim = dim
        try:
            import faiss  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("faiss not installed; install extras: pip install -e .[faiss]") from e

        self._faiss = faiss
        self._index = faiss.IndexFlatIP(dim)
        self._doc_ids: list[str] = []

    def add(self, doc_ids: list[str], vectors: np.ndarray) -> None:
        vectors = vectors.astype(np.float32, copy=False)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"Expected vectors shape (n,{self.dim}), got {vectors.shape}")
        self._index.add(vectors)
        self._doc_ids.extend(doc_ids)

    def search(self, query: np.ndarray, top_k: int) -> list[SearchHit]:
        query = query.astype(np.float32, copy=False).reshape(1, -1)
        scores, indices = self._index.search(query, top_k)
        hits: list[SearchHit] = []
        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx < 0 or idx >= len(self._doc_ids):
                continue
            hits.append(SearchHit(doc_id=self._doc_ids[idx], score=float(score)))
        return hits

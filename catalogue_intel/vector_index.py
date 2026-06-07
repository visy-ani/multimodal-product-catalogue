"""Vector index behind a tiny interface, with two interchangeable backends:

- `VectorIndex`        — in-memory numpy cosine (L2-normalise once, dot product),
                          persisted to .npz. Zero extra dependencies.
- `ChromaVectorIndex`  — a persistent Chroma DB via `langchain-chroma`. We embed
                          once (batched) with the official OpenAI SDK and hand
                          Chroma the PRECOMPUTED vectors, so it never re-embeds
                          (keeps the RPM budget intact).

Both expose: add / add_many / search(vector, top_k) -> [(id, score)] / __len__ /
ids / save / load. `make_index()` / `load_index()` pick the backend from config.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from . import config


def _normalise(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    return v / norm


class VectorIndex:
    """Cosine-similarity index over fixed-dim vectors keyed by product id."""

    def __init__(self, dim: int = config.EMBED_DIM) -> None:
        self.dim = dim
        self._ids: list[str] = []
        self._matrix: Optional[np.ndarray] = None  # (N, dim) normalised

    # -- mutation --------------------------------------------------------- #
    def add(self, id: str, vector: list[float]) -> None:
        vec = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if vec.shape[1] != self.dim:
            raise ValueError(f"vector dim {vec.shape[1]} != index dim {self.dim}")
        vec = _normalise(vec)
        if self._matrix is None:
            self._matrix = vec
        else:
            self._matrix = np.vstack([self._matrix, vec])
        self._ids.append(id)

    def add_many(self, items: list[tuple[str, list[float]]]) -> None:
        for id, vec in items:
            self.add(id, vec)

    # -- query ------------------------------------------------------------ #
    def search(self, vector: list[float], top_k: int = 5) -> list[tuple[str, float]]:
        """Return [(id, cosine_score)] sorted high→low."""
        if self._matrix is None or len(self._ids) == 0:
            return []
        q = _normalise(np.asarray(vector, dtype=np.float32).reshape(1, -1))
        scores = (self._matrix @ q.T).ravel()  # cosine, both sides normalised
        k = min(top_k, len(self._ids))
        idx = np.argsort(-scores)[:k]
        return [(self._ids[i], float(scores[i])) for i in idx]

    def __len__(self) -> int:
        return len(self._ids)

    @property
    def ids(self) -> list[str]:
        return list(self._ids)

    # -- persistence ------------------------------------------------------ #
    def save(self, path: Path = config.INDEX_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        matrix = self._matrix if self._matrix is not None else np.zeros((0, self.dim), np.float32)
        np.savez(path, ids=np.array(self._ids, dtype=object), matrix=matrix, dim=self.dim)

    @classmethod
    def load(cls, path: Path = config.INDEX_PATH) -> "VectorIndex":
        path = Path(path)
        data = np.load(path, allow_pickle=True)
        idx = cls(dim=int(data["dim"]))
        idx._ids = list(data["ids"])
        matrix = data["matrix"]
        idx._matrix = matrix if matrix.shape[0] > 0 else None
        return idx


class _NoEmbed:
    """A langchain Embeddings that refuses to embed.

    We supply Chroma with precomputed vectors and query by vector, so Chroma must
    never call out to an embedding model. If it tries, fail loudly.
    """

    def embed_documents(self, texts):  # pragma: no cover - defensive
        raise RuntimeError("ChromaVectorIndex uses precomputed vectors only")

    def embed_query(self, text):  # pragma: no cover - defensive
        raise RuntimeError("ChromaVectorIndex uses precomputed vectors only")


class ChromaVectorIndex:
    """Persistent Chroma backend via `langchain-chroma`, same interface as VectorIndex.

    Cosine space; scores are similarity = 1 - cosine_distance (so higher = closer,
    matching the numpy backend).
    """

    def __init__(
        self,
        dim: int = config.EMBED_DIM,
        persist_dir: Path = config.CHROMA_DIR,
        collection: str = config.CHROMA_COLLECTION,
        reset: bool = False,
    ) -> None:
        from langchain_chroma import Chroma

        self.dim = dim
        self._persist_dir = str(persist_dir)
        self._collection_name = collection
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._store = Chroma(
            collection_name=collection,
            embedding_function=_NoEmbed(),
            persist_directory=self._persist_dir,
            collection_metadata={"hnsw:space": "cosine"},
        )
        if reset:
            existing = self._store._collection.get()["ids"]
            if existing:
                self._store._collection.delete(ids=existing)

    @property
    def _coll(self):
        return self._store._collection

    def add(self, id: str, vector: list[float]) -> None:
        self.add_many([(id, vector)])

    def add_many(self, items: list[tuple[str, list[float]]]) -> None:
        if not items:
            return
        ids = [i for i, _ in items]
        embeddings = [list(map(float, v)) for _, v in items]
        # documents are required by chroma; store the id as a placeholder doc
        self._coll.add(ids=ids, embeddings=embeddings, documents=ids)

    def search(self, vector: list[float], top_k: int = 5) -> list[tuple[str, float]]:
        n = self._coll.count()
        if n == 0:
            return []
        res = self._coll.query(
            query_embeddings=[list(map(float, vector))],
            n_results=min(top_k, n),
            include=["distances"],
        )
        ids = res["ids"][0]
        dists = res["distances"][0]
        return [(i, 1.0 - float(d)) for i, d in zip(ids, dists)]  # cosine similarity

    def __len__(self) -> int:
        return self._coll.count()

    @property
    def ids(self) -> list[str]:
        return list(self._coll.get()["ids"])

    def save(self, path=None) -> None:
        """No-op: Chroma persists to `persist_directory` on every write."""

    @classmethod
    def load(cls, persist_dir: Path = config.CHROMA_DIR) -> "ChromaVectorIndex":
        return cls(persist_dir=persist_dir, reset=False)


# --------------------------------------------------------------------------- #
# Backend factory                                                             #
# --------------------------------------------------------------------------- #
def make_index(reset: bool = False, backend: Optional[str] = None):
    """Create a fresh index of the configured backend (for ingest)."""
    backend = backend or config.VECTOR_BACKEND
    if backend == "chroma":
        return ChromaVectorIndex(reset=reset)
    return VectorIndex()


def load_index(backend: Optional[str] = None):
    """Reconnect to / load a previously-persisted index."""
    backend = backend or config.VECTOR_BACKEND
    if backend == "chroma":
        return ChromaVectorIndex.load()
    return VectorIndex.load()

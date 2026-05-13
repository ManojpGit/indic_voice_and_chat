"""Embedding + reranking interfaces.

Three implementations of each:

- ``IEmbedder`` Protocol — ``embed_documents`` and ``embed_query`` returning
  unit-norm float vectors.
- ``LocalEmbedder`` — wraps ``sentence-transformers``; lazy-loads on first use.
  Default model: ``paraphrase-multilingual-MiniLM-L12-v2`` (384-dim).
- ``HashEmbedder`` — deterministic, dependency-free, used by tests so the
  unit suite stays under a second and never downloads model weights.

Same pattern for ``IReranker``:
- ``LocalReranker`` lazy-loads a cross-encoder.
- ``IdentityReranker`` returns scores derived from substring overlap; works
  for tests and provides a graceful no-model fallback.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Optional, Protocol


# --- Embedders ----------------------------------------------------------


class IEmbedder(Protocol):
    dim: int
    model_name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class HashEmbedder:
    """Deterministic feature-hashing embedder. NOT semantic — just makes
    similar strings produce somewhat-similar unit vectors. Useful for tests.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.model_name = "test/hash-embedder"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        if not text:
            vec[0] = 1.0
            return vec
        for token in _tokenize(text):
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 1) & 1 else -1.0
            vec[idx] += sign
        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class LocalEmbedder:
    """sentence-transformers wrapper with lazy model load."""

    DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        dim: int = 384,
    ) -> None:
        self.model_name = model_name
        self.dim = dim
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "LocalEmbedder requires the 'sentence-transformers' package. "
                "Install with: pip install sentence-transformers"
            ) from e
        self._model = SentenceTransformer(self.model_name)
        actual_dim = self._model.get_sentence_embedding_dimension()
        if actual_dim != self.dim:
            self.dim = actual_dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._ensure_model()
        if not texts:
            return []
        # normalize_embeddings=True so cosine == inner product (matches our FAISS index).
        vecs = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return [v.tolist() for v in vecs]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


# --- Rerankers ----------------------------------------------------------


@dataclass
class RerankItem:
    text: str
    score: float


class IReranker(Protocol):
    model_name: str

    def rerank(self, query: str, documents: list[str]) -> list[float]: ...


class IdentityReranker:
    """No-op reranker that scores by token overlap. Useful as a fallback
    and as a deterministic test double."""

    model_name = "test/identity-reranker"

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not query or not documents:
            return [0.0] * len(documents)
        q_tokens = set(_tokenize(query))
        scores: list[float] = []
        for d in documents:
            d_tokens = set(_tokenize(d))
            overlap = len(q_tokens & d_tokens)
            denom = math.sqrt(len(q_tokens) * len(d_tokens)) or 1.0
            scores.append(overlap / denom)
        return scores


class LocalReranker:
    """Cross-encoder reranker, lazy-loaded."""

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise RuntimeError(
                "LocalReranker requires 'sentence-transformers'."
            ) from e
        self._model = CrossEncoder(self.model_name)

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        self._ensure_model()
        if not documents:
            return []
        pairs = [(query, d) for d in documents]
        scores = self._model.predict(pairs)
        return [float(s) for s in scores]


# --- helpers -----------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Cheap tokenizer for hash embedding / identity reranker.

    Lowercases, splits on whitespace + a few punctuation chars. Devanagari
    and Latin both flow through unchanged.
    """
    out: list[str] = []
    cur: list[str] = []
    for ch in text.lower():
        if ch.isalnum() or ch in "ँंः़":  # keep Devanagari modifiers attached
            cur.append(ch)
        else:
            if cur:
                out.append("".join(cur))
                cur = []
    if cur:
        out.append("".join(cur))
    return [t for t in out if t]

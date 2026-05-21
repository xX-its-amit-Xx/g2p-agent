"""Embedding backends.

Default is BAAI/bge-small-en-v1.5 via sentence-transformers. When torch /
sentence-transformers are not installed (e.g. a lightweight CI box or this
build environment), we fall back to a deterministic ``HashingEmbedder`` so the
retrieval pipeline still runs end-to-end and tests stay reproducible.

The hashing embedder is *not* semantically meaningful in the BGE sense, but it
is a stable, normalized bag-of-character-ngrams projection: identical text maps
to identical vectors and lexically similar text maps to nearby vectors, which
is enough to exercise dense scoring and fusion deterministically. Production
deployments should install the ``embeddings`` extra to get real BGE vectors.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

from .config import settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: list[str], *, is_query: bool = False) -> np.ndarray: ...


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class HashingEmbedder:
    """Deterministic hashed character-n-gram embedder (offline fallback).

    Tokenizes to lowercase word + 3-gram features and hashes each into a fixed
    dimensional space with a signed hash, then L2-normalizes. Pure-numpy, no
    model download, fully deterministic.
    """

    name = "hash"

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim or settings.embedding_dim

    def _features(self, text: str) -> list[str]:
        text = text.lower()
        words = _TOKEN_RE.findall(text)
        feats = list(words)
        for w in words:
            padded = f"#{w}#"
            feats.extend(padded[i : i + 3] for i in range(len(padded) - 2))
        return feats

    def encode(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for feat in self._features(text):
                h = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(h[:4], "little") % self.dim
                sign = 1.0 if h[4] & 1 else -1.0
                out[i, idx] += sign
        return _normalize(out)


class BGEEmbedder:
    """Real BAAI/bge-small-en-v1.5 embeddings via sentence-transformers.

    BGE recommends prefixing queries (not documents) with a short instruction;
    we follow the official recipe for the retrieval use-case.
    """

    name = "bge"
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer  # lazy import

        self.model_name = model_name or settings.embedding_model
        self._model = SentenceTransformer(self.model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        payload = [self.QUERY_PREFIX + t for t in texts] if is_query else texts
        vecs = self._model.encode(payload, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


def get_embedder(kind: str | None = None) -> Embedder:
    """Factory honoring config; ``None`` resolves via ``settings.resolve_embedder()``."""
    kind = kind or settings.resolve_embedder()
    if kind == "bge":
        return BGEEmbedder()
    if kind == "hash":
        return HashingEmbedder()
    raise ValueError(f"Unknown embedder kind: {kind!r}")

"""Hybrid retrieval: dense (BGE/hash) + sparse (BM25), fused with RRF.

Reciprocal Rank Fusion (Cormack et al., 2009) combines the dense and sparse
rankings without needing to calibrate their score scales:

    score(d) = Σ_r  1 / (rrf_k + rank_r(d))

An optional, cheap rerank pass then nudges chunks that match a requested gene
and that physically cover a queried residue — the signals that matter most for
variant questions. A heavier cross-encoder reranker can be dropped in here (see
AGENTS.md) without touching callers.
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

from .config import settings
from .embeddings import Embedder, get_embedder
from .ingest import get_collection
from .schemas import Chunk, ScoredChunk

_TOK = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOK.findall(text.lower())


class Retriever:
    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder or get_embedder()
        self.collection = get_collection(reset=False)
        data = self.collection.get(include=["documents", "metadatas"])
        self.ids: list[str] = data["ids"]
        self.documents: list[str] = data["documents"] or []
        self.metadatas: list[dict] = data["metadatas"] or []
        self._id_to_idx = {cid: i for i, cid in enumerate(self.ids)}
        self._bm25 = BM25Okapi([_tokenize(d) for d in self.documents]) if self.documents else None

    # --- public API --------------------------------------------------------
    @property
    def size(self) -> int:
        return len(self.ids)

    def _chunk_from_idx(self, idx: int) -> Chunk:
        md = self.metadatas[idx] or {}
        return Chunk(
            id=self.ids[idx],
            text=self.documents[idx],
            gene=md.get("gene", ""),
            uniprot_id=md.get("uniprot_id", ""),
            start=int(md.get("start", 0)),
            end=int(md.get("end", 0)),
            domain=md.get("domain"),
            chunk_kind=md.get("chunk_kind", "domain"),
            metadata=md,
        )

    def _dense_ranks(self, query: str, gene: str | None, k: int) -> dict[str, int]:
        qv = self.embedder.encode([query], is_query=True)[0]
        where = {"gene": gene.upper()} if gene else None
        res = self.collection.query(
            query_embeddings=[qv.tolist()],
            n_results=min(k, max(self.size, 1)),
            where=where,
        )
        ids = res["ids"][0] if res["ids"] else []
        return {cid: rank for rank, cid in enumerate(ids)}

    def _sparse_ranks(self, query: str, gene: str | None, k: int) -> dict[str, int]:
        if self._bm25 is None:
            return {}
        scores = self._bm25.get_scores(_tokenize(query))
        order = np.argsort(scores)[::-1]
        ranks: dict[str, int] = {}
        rank = 0
        for idx in order:
            if scores[idx] <= 0:
                break
            if gene and (self.metadatas[idx] or {}).get("gene") != gene.upper():
                continue
            ranks[self.ids[idx]] = rank
            rank += 1
            if rank >= k:
                break
        return ranks

    def search(
        self,
        query: str,
        *,
        gene: str | None = None,
        position: int | None = None,
        top_k: int | None = None,
        candidate_k: int | None = None,
        rerank: bool | None = None,
    ) -> list[ScoredChunk]:
        if self.size == 0:
            return []
        top_k = top_k or settings.top_k
        candidate_k = candidate_k or settings.candidate_k
        rerank = settings.rerank if rerank is None else rerank

        dense = self._dense_ranks(query, gene, candidate_k)
        sparse = self._sparse_ranks(query, gene, candidate_k)

        rrf_k = settings.rrf_k
        fused: dict[str, dict] = {}
        for cid, r in dense.items():
            fused.setdefault(cid, {"dense": None, "sparse": None})["dense"] = r
        for cid, r in sparse.items():
            fused.setdefault(cid, {"dense": None, "sparse": None})["sparse"] = r

        scored: list[ScoredChunk] = []
        for cid, ranks in fused.items():
            if cid not in self._id_to_idx:
                continue
            idx = self._id_to_idx[cid]
            d_component = 1.0 / (rrf_k + ranks["dense"]) if ranks["dense"] is not None else 0.0
            s_component = 1.0 / (rrf_k + ranks["sparse"]) if ranks["sparse"] is not None else 0.0
            score = d_component + s_component
            chunk = self._chunk_from_idx(idx)
            components = {"dense_rrf": d_component, "sparse_rrf": s_component}

            if rerank:
                bonus = 0.0
                if gene and chunk.gene == gene.upper():
                    bonus += 0.05
                if position is not None and chunk.covers(position):
                    bonus += 0.15  # residue-level hit is the strongest signal
                components["rerank_bonus"] = bonus
                score += bonus

            scored.append(ScoredChunk(
                chunk=chunk, score=score,
                dense_rank=ranks["dense"], sparse_rank=ranks["sparse"],
                components=components,
            ))

        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    """Process-wide cached retriever (BM25 index build is not free)."""
    return Retriever()

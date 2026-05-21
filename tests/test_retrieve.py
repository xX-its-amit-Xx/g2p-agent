"""Retrieval tests: embeddings, RRF fusion, and end-to-end search.

These run against the offline HashingEmbedder and a small in-memory-ish Chroma
collection built from synthetic chunks, so they need no network or API key.
"""

from __future__ import annotations

import numpy as np
import pytest

from g2p_agent.embeddings import HashingEmbedder
from g2p_agent.schemas import Chunk


def test_hashing_embedder_is_deterministic_and_normalized():
    emb = HashingEmbedder(dim=128)
    a = emb.encode(["TP53 R175H DNA-binding domain"])
    b = emb.encode(["TP53 R175H DNA-binding domain"])
    assert a.shape == (1, 128)
    np.testing.assert_allclose(a, b)  # deterministic
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)  # normalized


def test_hashing_embedder_similar_text_closer_than_unrelated():
    emb = HashingEmbedder(dim=256)
    v = emb.encode([
        "TP53 DNA-binding domain zinc coordination",
        "TP53 DNA-binding domain zinc coordination site",
        "CFTR chloride channel ATP binding",
    ])
    sim_related = float(v[0] @ v[1])
    sim_unrelated = float(v[0] @ v[2])
    assert sim_related > sim_unrelated


def test_chunk_covers():
    c = Chunk(id="x", text="t", gene="TP53", uniprot_id="P04637", start=94, end=312)
    assert c.covers(175)
    assert not c.covers(400)
    assert not c.covers(None)


@pytest.fixture()
def populated_collection(tmp_path, monkeypatch):
    """Build a tiny persistent Chroma collection with synthetic chunks."""
    monkeypatch.setenv("G2P_CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("G2P_EMBEDDER", "hash")
    monkeypatch.setenv("G2P_COLLECTION", "test_collection")
    # rebuild settings with patched env
    import importlib

    import g2p_agent.config as cfg
    importlib.reload(cfg)
    import g2p_agent.ingest as ingest
    importlib.reload(ingest)

    emb = HashingEmbedder()
    coll = ingest.get_collection(reset=True)
    chunks = [
        ("TP53:P04637:161-200", "TP53 P04637 residues 161-200 DNA-binding region buried R175 zinc", "TP53", 161, 200),
        ("TP53:P04637:1-40", "TP53 P04637 residues 1-40 transactivation phosphorylation S15", "TP53", 1, 40),
        ("CFTR:P13569:423-646", "CFTR P13569 residues 423-646 ABC transporter NBD1 F508 buried", "CFTR", 423, 646),
    ]
    docs = [c[1] for c in chunks]
    vecs = emb.encode(docs)
    coll.upsert(
        ids=[c[0] for c in chunks],
        documents=docs,
        embeddings=[v.tolist() for v in vecs],
        metadatas=[{"gene": c[2], "uniprot_id": "X", "start": c[3], "end": c[4],
                    "domain": "d", "chunk_kind": "cluster"} for c in chunks],
    )
    import g2p_agent.retrieve as retrieve
    importlib.reload(retrieve)
    return retrieve


def test_search_returns_gene_filtered_results(populated_collection):
    retrieve = populated_collection
    r = retrieve.Retriever()
    results = r.search("DNA-binding domain stability", gene="TP53", position=175)
    assert results, "expected at least one result"
    assert all(s.chunk.gene == "TP53" for s in results)
    # the chunk covering residue 175 should rank first thanks to the rerank bonus
    assert results[0].chunk.covers(175)


def test_rrf_combines_dense_and_sparse(populated_collection):
    retrieve = populated_collection
    r = retrieve.Retriever()
    results = r.search("ABC transporter NBD1 F508", gene="CFTR")
    assert results
    top = results[0]
    # fused score should include contributions and the components recorded
    assert "dense_rrf" in top.components and "sparse_rrf" in top.components
    assert top.score > 0


def test_empty_query_safe(populated_collection):
    retrieve = populated_collection
    r = retrieve.Retriever()
    assert r.search("zzzzz nonexistent token qqqq", gene="TP53") is not None

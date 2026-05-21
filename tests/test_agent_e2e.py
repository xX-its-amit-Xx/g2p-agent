"""End-to-end agent + eval tests against the offline (mock) backends.

Assumes the baseline genes have been ingested (`g2p-agent ingest`). If the
Chroma collection is empty, the ingest-dependent tests are skipped so the suite
still passes on a fresh checkout / CI box without network.
"""

from __future__ import annotations

import pytest

from g2p_agent.agent import Agent
from g2p_agent.llm import MockLLM, _parse_variant
from g2p_agent.retrieve import get_retriever
from g2p_agent.schemas import Confidence


@pytest.fixture(scope="module")
def has_index() -> bool:
    try:
        return get_retriever().size > 0
    except Exception:
        return False


def test_parse_variant_forms():
    assert _parse_variant("TP53 R175H stability")["position"] == 175
    assert _parse_variant("the R213X nonsense")["alt_aa"] == "*"
    assert _parse_variant("CFTR F508del folding")["position"] == 508
    assert _parse_variant("no variant here") == {}


def test_mock_llm_emits_search_tool_call_first():
    llm = MockLLM()
    reply = llm.message(
        system="s",
        messages=[{"role": "user", "content": "What does R175H in TP53 do?"}],
        tools=[{"name": "search_variants"}],
    )
    assert reply.stop_reason == "tool_use"
    assert reply.tool_calls[0]["name"] == "search_variants"
    assert reply.tool_calls[0]["input"]["gene"] == "TP53"


def test_agent_grounds_answer_for_known_variant(has_index):
    if not has_index:
        pytest.skip("no Chroma index; run `g2p-agent ingest` first")
    resp = Agent().ask("What does the missense variant R175H in TP53 do to protein stability?")
    assert not resp.insufficient_evidence
    assert resp.claims, "expected grounded claims"
    # every claim must cite a chunk that was actually retrieved
    retrieved = set(resp.debug["retrieved_ids"])
    for claim in resp.claims:
        assert claim.citations, "claim must be cited"
        for cit in claim.citations:
            assert cit.chunk_id in retrieved
    assert resp.confidence in (Confidence.high, Confidence.medium)


def test_agent_refuses_when_gene_not_indexed(has_index):
    if not has_index:
        pytest.skip("no Chroma index; run `g2p-agent ingest` first")
    resp = Agent().ask("What does the EGFR T790M variant do to drug binding?")
    assert resp.insufficient_evidence
    assert "don't know" in resp.answer.lower()
    assert resp.confidence == Confidence.low
    assert not resp.cited_chunk_ids


def test_agent_never_cites_unretrieved_chunks(has_index):
    if not has_index:
        pytest.skip("no Chroma index; run `g2p-agent ingest` first")
    resp = Agent().ask("How does C124S affect the catalytic activity of PTEN?")
    retrieved = set(resp.debug["retrieved_ids"])
    assert set(resp.cited_chunk_ids).issubset(retrieved)

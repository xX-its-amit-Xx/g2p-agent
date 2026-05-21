"""Evaluation harness.

Metrics
-------
- **Task success rate** — binary: did the answer correctly characterize the
  variant (gold-keyword coverage + non-refusal), or correctly refuse when the
  item is unanswerable?
- **Grounding rate** — mean fraction of factual claims that carry a citation to
  a chunk that was actually retrieved.
- **Hallucination rate** — mean fraction of claims unsupported by any retrieved
  chunk (no citation, or citation whose chunk text doesn't overlap the claim).
- **Calibration** — does stated confidence track correctness? Reported as
  per-confidence-bucket accuracy plus an overconfidence count.

Judging
-------
Two judges implement the same interface:
- :class:`AnthropicJudge` — LLM-as-judge using Claude Opus.
- :class:`MockJudge` — deterministic, no API key. Used here for the shipped
  baseline and as the manual-gold sanity layer.
The harness also records ``retrieval_hit`` independently of the judge.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Protocol

from .agent import Agent
from .config import settings
from .llm import get_llm
from .retrieve import get_retriever
from .schemas import AgentResponse, BenchmarkItem, Confidence, EvalRecord, JudgeVerdict

_TOK = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOK.findall(text.lower()))


def load_benchmark(path: str | Path) -> list[BenchmarkItem]:
    items = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("//"):
            items.append(BenchmarkItem.model_validate_json(line))
    return items


# --------------------------------------------------------------------------- #
# Judges
# --------------------------------------------------------------------------- #
class Judge(Protocol):
    backend: str

    def judge(self, item: BenchmarkItem, response: AgentResponse, chunk_texts: dict[str, str]) -> JudgeVerdict: ...


class MockJudge:
    """Deterministic rubric judge (no API key)."""

    backend = "mock"

    def judge(self, item: BenchmarkItem, response: AgentResponse, chunk_texts: dict[str, str]) -> JudgeVerdict:
        # --- grounding / hallucination over claims --------------------------
        n_claims = len(response.claims)
        grounded = 0
        hallucinated = 0
        for claim in response.claims:
            claim_toks = _tokens(claim.text)
            supported = False
            for cit in claim.citations:
                ctext = chunk_texts.get(cit.chunk_id, "")
                if ctext and len(claim_toks & _tokens(ctext)) >= 2:
                    supported = True
                    break
            if claim.citations:
                grounded += 1
            if not supported:
                hallucinated += 1

        if item.expect_answerable:
            grounding_rate = grounded / n_claims if n_claims else 0.0
            hallucination_rate = hallucinated / n_claims if n_claims else 1.0
        else:
            # Correct behavior is a refusal: full grounding, zero hallucination.
            grounding_rate = 1.0 if response.insufficient_evidence else (grounded / n_claims if n_claims else 0.0)
            hallucination_rate = 0.0 if response.insufficient_evidence else 1.0

        # --- task success ---------------------------------------------------
        if not item.expect_answerable:
            task_success = response.insufficient_evidence
        else:
            ans = response.answer.lower()
            kws = [k.lower() for k in item.gold_keywords]
            hits = sum(1 for k in kws if k in ans)
            kw_ratio = hits / len(kws) if kws else 0.0
            task_success = (
                not response.insufficient_evidence
                and n_claims > 0
                and kw_ratio >= 0.4
            )

        # --- confidence appropriateness ------------------------------------
        conf = response.confidence
        if task_success:
            confidence_appropriate = conf in (Confidence.high, Confidence.medium) or (
                not item.expect_answerable and conf == Confidence.low
            )
        else:
            confidence_appropriate = conf == Confidence.low  # failing should be low-confidence

        rationale = (
            f"kw_match={'n/a' if not item.expect_answerable else f'{hits}/{len(item.gold_keywords)}'}; "
            f"claims={n_claims} grounded={grounded} hallucinated={hallucinated}; "
            f"confidence={conf.value} ({'ok' if confidence_appropriate else 'miscalibrated'})."
        )
        return JudgeVerdict(
            task_success=task_success,
            grounding_rate=round(grounding_rate, 3),
            hallucination_rate=round(hallucination_rate, 3),
            confidence_appropriate=confidence_appropriate,
            rationale=rationale,
        )


_JUDGE_SYSTEM = """You are a rigorous evaluator of a biomedical RAG agent. You are
given a question, a gold reference answer, the agent's answer, its claims with
cited chunk ids, and the text of the cited chunks. Score strictly and return ONLY
JSON: {"task_success": bool, "grounding_rate": float 0-1, "hallucination_rate":
float 0-1, "confidence_appropriate": bool, "rationale": str}. task_success = did
the agent correctly characterize the variant (or correctly refuse if
unanswerable). grounding_rate = fraction of claims actually supported by their
cited chunk text. hallucination_rate = fraction of claims NOT supported by any
cited chunk. confidence_appropriate = does the stated confidence match
correctness."""


class AnthropicJudge:
    """LLM-as-judge using Claude Opus."""

    backend = "anthropic"

    def __init__(self) -> None:
        self.llm = get_llm("anthropic")

    def judge(self, item: BenchmarkItem, response: AgentResponse, chunk_texts: dict[str, str]) -> JudgeVerdict:
        cited = "\n".join(f"[{cid}] {chunk_texts.get(cid, '(text unavailable)')[:600]}"
                          for cid in response.cited_chunk_ids) or "(no citations)"
        payload = {
            "question": item.question,
            "gold_answer": item.gold_answer,
            "gold_keywords": item.gold_keywords,
            "expect_answerable": item.expect_answerable,
            "agent_answer": response.answer,
            "agent_claims": [{"text": c.text, "citations": [ct.chunk_id for ct in c.citations]} for c in response.claims],
            "agent_confidence": response.confidence.value,
            "agent_insufficient_evidence": response.insufficient_evidence,
            "cited_chunks": cited,
        }
        reply = self.llm.message(
            system=_JUDGE_SYSTEM,
            messages=[{"role": "user", "content": json.dumps(payload)}],
            model=settings.judge_model,
            max_tokens=700,
        )
        data = _safe_json(reply.text) or {}
        return JudgeVerdict(
            task_success=bool(data.get("task_success", False)),
            grounding_rate=float(data.get("grounding_rate", 0.0)),
            hallucination_rate=float(data.get("hallucination_rate", 1.0)),
            confidence_appropriate=bool(data.get("confidence_appropriate", False)),
            rationale=str(data.get("rationale", "")),
        )


def _safe_json(text: str) -> dict | None:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def get_judge(kind: str | None = None) -> Judge:
    kind = kind or settings.resolve_llm()
    return AnthropicJudge() if kind == "anthropic" else MockJudge()


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #
def _chunk_texts_for(response: AgentResponse, retriever) -> dict[str, str]:
    out: dict[str, str] = {}
    for cid in set(response.cited_chunk_ids) | set(response.debug.get("retrieved_ids", [])):
        idx = retriever._id_to_idx.get(cid)
        if idx is not None:
            out[cid] = retriever.documents[idx]
    return out


def _retrieval_hit(item: BenchmarkItem, response: AgentResponse) -> bool:
    retrieved = response.debug.get("retrieved_ids", [])
    return any(rid.split(":")[0] == item.gene.upper() for rid in retrieved)


def run_eval(
    items: list[BenchmarkItem],
    *,
    agent: Agent | None = None,
    judge: Judge | None = None,
    progress=lambda *_: None,
) -> dict[str, Any]:
    agent = agent or Agent()
    judge = judge or get_judge()
    retriever = get_retriever()

    records: list[EvalRecord] = []
    for i, item in enumerate(items, 1):
        progress(f"[{i}/{len(items)}] {item.id}: {item.question[:60]}")
        response = agent.ask(item.question)
        chunk_texts = _chunk_texts_for(response, retriever)
        verdict = judge.judge(item, response, chunk_texts)
        records.append(EvalRecord(
            item=item, response=response, verdict=verdict,
            retrieval_hit=_retrieval_hit(item, response),
        ))

    return _aggregate(records, agent, judge)


def _aggregate(records: list[EvalRecord], agent: Agent, judge: Judge) -> dict[str, Any]:
    n = len(records)
    success = sum(r.verdict.task_success for r in records)
    grounding = sum(r.verdict.grounding_rate for r in records) / n if n else 0.0
    hallucination = sum(r.verdict.hallucination_rate for r in records) / n if n else 0.0
    retrieval_hit = sum(r.retrieval_hit for r in records) / n if n else 0.0
    conf_ok = sum(r.verdict.confidence_appropriate for r in records)

    # calibration: accuracy within each stated-confidence bucket
    buckets: dict[str, list[bool]] = {"high": [], "medium": [], "low": []}
    overconfident = 0
    for r in records:
        buckets[r.response.confidence.value].append(r.verdict.task_success)
        if r.response.confidence == Confidence.high and not r.verdict.task_success:
            overconfident += 1
    calibration = {
        c: {"n": len(v), "accuracy": round(sum(v) / len(v), 3) if v else None}
        for c, v in buckets.items()
    }

    # breakdowns by category
    def breakdown(attr: str) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for r in records:
            key = str(getattr(r.item, attr).value if hasattr(getattr(r.item, attr), "value") else getattr(r.item, attr))
            d = out.setdefault(key, {"n": 0, "success": 0})
            d["n"] += 1
            d["success"] += int(r.verdict.task_success)
        for d in out.values():
            d["success_rate"] = round(d["success"] / d["n"], 3)
        return out

    return {
        "n_items": n,
        "backends": {
            "llm": agent.llm.backend,
            "judge": judge.backend,
            "embedder": get_retriever().embedder.name,
            "agent_model": settings.agent_model,
            "judge_model": settings.judge_model,
        },
        "metrics": {
            "task_success_rate": round(success / n, 3) if n else 0.0,
            "grounding_rate": round(grounding, 3),
            "hallucination_rate": round(hallucination, 3),
            "confidence_appropriate_rate": round(conf_ok / n, 3) if n else 0.0,
            "retrieval_hit_rate": round(retrieval_hit, 3),
            "overconfident_count": overconfident,
        },
        "calibration": calibration,
        "by_variant_type": breakdown("variant_type"),
        "by_mechanism": breakdown("mechanism"),
        "records": [
            {
                "id": r.item.id,
                "question": r.item.question,
                "variant_type": r.item.variant_type.value,
                "mechanism": r.item.mechanism.value,
                "task_success": r.verdict.task_success,
                "grounding_rate": r.verdict.grounding_rate,
                "hallucination_rate": r.verdict.hallucination_rate,
                "confidence": r.response.confidence.value,
                "confidence_appropriate": r.verdict.confidence_appropriate,
                "insufficient_evidence": r.response.insufficient_evidence,
                "retrieval_hit": r.retrieval_hit,
                "cited_chunk_ids": r.response.cited_chunk_ids,
                "rationale": r.verdict.rationale,
            }
            for r in records
        ],
    }

"""LLM backends: a thin, Anthropic-shaped interface plus a deterministic mock.

The agent and judge talk to an :class:`LLMClient`. We deliberately mirror the
Anthropic Messages tool-use protocol (list of content blocks; ``stop_reason``;
``tool_use`` / ``tool_result`` blocks) so the *same* agent loop drives both the
real model and the offline mock.

``MockLLM`` is not a language model. It is a small rule-based stand-in that (a)
actually issues tool calls, (b) reads the returned chunks, and (c) composes a
grounded, cited answer with a calibrated confidence. Its purpose is to let the
full agent control-flow and the evaluation harness run and produce reproducible
numbers without an API key. Anyone with ``ANTHROPIC_API_KEY`` set gets the real
Claude model with zero code changes.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import settings


@dataclass
class LLMReply:
    """Backend-agnostic reply: either tool calls to run, or final text."""

    stop_reason: str  # "tool_use" | "end_turn"
    text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # {id, name, input}
    raw: Any = None


class LLMClient(Protocol):
    backend: str

    def message(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMReply: ...


# --------------------------------------------------------------------------- #
# Anthropic backend
# --------------------------------------------------------------------------- #
class AnthropicLLM:
    backend = "anthropic"

    def __init__(self, api_key: str | None = None) -> None:
        import anthropic  # lazy import

        self._client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])

    def message(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMReply:
        kwargs: dict[str, Any] = {
            "model": model or settings.agent_model,
            "max_tokens": max_tokens or settings.max_tokens,
            "system": system,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
        resp = self._client.messages.create(**kwargs)
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({"id": block.id, "name": block.name, "input": block.input})
        return LLMReply(
            stop_reason=resp.stop_reason or "end_turn",
            text="".join(text_parts),
            tool_calls=tool_calls,
            raw=resp,
        )


# --------------------------------------------------------------------------- #
# Deterministic mock backend
# --------------------------------------------------------------------------- #
_VARIANT_RE = re.compile(r"\b([A-Z])\s?(\d{1,5})\s?([A-Z])\b")  # R175H
_NONSENSE_RE = re.compile(r"\b([A-Z])(\d{1,5})(?:X|\*|Ter|ter)\b")  # R213X
_INDEL_RE = re.compile(r"\b([A-Z])(\d{1,5})(?:del|dup|fs|delins|ins)\b")  # F508del
_GENE_HINT_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,6})\b")


def _parse_variant(question: str) -> dict[str, Any]:
    """Extract (ref_aa, position, alt_aa) from common protein-change notations."""
    out: dict[str, Any] = {}
    m = _NONSENSE_RE.search(question)
    if m:
        out["ref_aa"], out["position"], out["alt_aa"] = m.group(1), int(m.group(2)), "*"
        return out
    m = _INDEL_RE.search(question)
    if m:
        out["ref_aa"], out["position"], out["alt_aa"] = m.group(1), int(m.group(2)), "del"
        return out
    m = _VARIANT_RE.search(question)
    if m:
        out["ref_aa"], out["position"], out["alt_aa"] = m.group(1), int(m.group(2)), m.group(3)
    return out


class MockLLM:
    """Rule-based, tool-using stand-in for Claude (offline, deterministic).

    Strategy, per agent turn:
      1. On the first turn it emits a ``search_variants`` tool call built from
         the parsed gene + variant in the question. If a residue position is
         known it follows up with ``get_variant_context``.
      2. Once tool results (retrieved chunks) are in the transcript, it grounds
         a final answer: it extracts the most on-topic sentences from the
         retrieved chunk texts, attaches their chunk ids as citations, and sets
         confidence from how well retrieval covered the queried residue.
      3. If no chunk supports the question, it returns the mandated refusal
         ("I don't know based on G2P data").
    """

    backend = "mock"

    def message(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMReply:
        # Find the original user question (first user text block).
        question = self._first_user_text(messages)
        parsed = _parse_variant(question)
        gene = self._guess_gene(question)

        tool_results = self._collect_tool_results(messages)
        already_searched = self._has_tool_call(messages, "search_variants")
        already_context = self._has_tool_call(messages, "get_variant_context")

        # Phase 1: search if we have tools and haven't yet.
        if tools and not already_searched:
            query_terms = [gene or "", parsed.get("ref_aa", ""), str(parsed.get("position") or ""),
                           parsed.get("alt_aa", ""), question]
            return LLMReply(
                stop_reason="tool_use",
                tool_calls=[{
                    "id": "mock_search_1",
                    "name": "search_variants",
                    "input": {
                        "query": " ".join(t for t in query_terms if t).strip(),
                        "gene": gene,
                    },
                }],
            )

        # Phase 2: optionally pull focused residue context.
        if tools and parsed.get("position") and gene and not already_context:
            return LLMReply(
                stop_reason="tool_use",
                tool_calls=[{
                    "id": "mock_context_1",
                    "name": "get_variant_context",
                    "input": {
                        "gene": gene,
                        "position": parsed["position"],
                        "aa_change": f"{parsed.get('ref_aa','')}{parsed['position']}{parsed.get('alt_aa','')}",
                    },
                }],
            )

        # Phase 3: compose grounded answer from retrieved chunks.
        chunks = self._flatten_chunks(tool_results)
        answer = self._compose(question, gene, parsed, chunks)
        return LLMReply(stop_reason="end_turn", text=json.dumps(answer))

    # --- helpers -----------------------------------------------------------
    @staticmethod
    def _first_user_text(messages: list[dict[str, Any]]) -> str:
        for m in messages:
            if m["role"] == "user":
                content = m["content"]
                if isinstance(content, str):
                    return content
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        return block["text"]
        return ""

    @staticmethod
    def _guess_gene(question: str) -> str | None:
        from .config import DEFAULT_GENE_UNIPROT

        for token in _GENE_HINT_RE.findall(question):
            if token in DEFAULT_GENE_UNIPROT:
                return token
        return None

    @staticmethod
    def _has_tool_call(messages: list[dict[str, Any]], name: str) -> bool:
        for m in messages:
            if m["role"] != "assistant":
                continue
            content = m["content"]
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("name") == name:
                        return True
        return False

    @staticmethod
    def _collect_tool_results(messages: list[dict[str, Any]]) -> list[Any]:
        results = []
        for m in messages:
            if m["role"] != "user" or not isinstance(m["content"], list):
                continue
            for block in m["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    content = block.get("content")
                    if isinstance(content, list):
                        for c in content:
                            if isinstance(c, dict) and c.get("type") == "text":
                                results.append(c["text"])
                    elif isinstance(content, str):
                        results.append(content)
        return results

    @staticmethod
    def _flatten_chunks(tool_results: list[str]) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        seen: set[str] = set()
        for blob in tool_results:
            try:
                data = json.loads(blob)
            except (json.JSONDecodeError, TypeError):
                continue
            items = data.get("chunks", data) if isinstance(data, dict) else data
            if isinstance(items, list):
                for it in items:
                    if isinstance(it, dict) and "id" in it and it["id"] not in seen:
                        seen.add(it["id"])
                        chunks.append(it)
        return chunks

    @staticmethod
    def _compose(
        question: str, gene: str | None, parsed: dict[str, Any], chunks: list[dict[str, Any]]
    ) -> dict[str, Any]:
        position = parsed.get("position")
        # Rank chunks: prefer those covering the residue, then by query overlap.
        q_tokens = set(_TOKEN_FROM(question))

        def chunk_score(ch: dict[str, Any]) -> tuple[int, int]:
            covers = 0
            md = ch.get("metadata", {})
            start, end = md.get("start"), md.get("end")
            if position and isinstance(start, int) and isinstance(end, int) and start <= position <= end:
                covers = 1
            overlap = len(q_tokens & set(_TOKEN_FROM(ch.get("text", ""))))
            return (covers, overlap)

        ranked = sorted(chunks, key=chunk_score, reverse=True)
        covering = [c for c in ranked if chunk_score(c)[0] == 1]
        if position is not None and covering:
            supporting = covering[:2]  # focus on chunks that actually contain the residue
        else:
            supporting = [c for c in ranked if chunk_score(c) > (0, 0)][:3]

        if not supporting:
            return {
                "answer": "I don't know based on G2P data. The retrieved G2P "
                "feature records do not contain information addressing this question.",
                "claims": [],
                "confidence": "low",
                "confidence_reasoning": "No retrieved chunk covers the queried "
                "gene/residue, so there is no G2P evidence to ground an answer.",
                "cited_chunk_ids": [],
                "insufficient_evidence": True,
            }

        claims = []
        cited_ids = []
        for ch in supporting:
            sentence = _chunk_digest(ch.get("text", ""), position)
            claims.append({"text": sentence, "citations": [{"chunk_id": ch["id"], "quote": sentence[:160]}]})
            cited_ids.append(ch["id"])

        covers_residue = position is None or any(
            (lambda md: isinstance(md.get("start"), int)
             and md["start"] <= position <= md.get("end", -1))(c.get("metadata", {}))
            for c in supporting
        )
        confidence = "high" if covers_residue and len(supporting) >= 2 else (
            "medium" if covers_residue else "low")
        reasoning = (
            f"{len(supporting)} retrieved chunk(s) "
            + ("directly cover the queried residue" if covers_residue else "are gene-level but do not pinpoint the residue")
            + f"; confidence={confidence}."
        )
        label = parsed_label(gene, parsed)
        answer_text = (
            f"Based on G2P portal feature data for {label}: "
            + " ".join(c["text"] for c in claims)
        )
        return {
            "answer": answer_text,
            "claims": claims,
            "confidence": confidence,
            "confidence_reasoning": reasoning,
            "cited_chunk_ids": cited_ids,
            "insufficient_evidence": False,
        }


def _TOKEN_FROM(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _lead_sentence(text: str) -> str:
    text = text.strip().replace("\n", " ")
    parts = re.split(r"(?<=[.;])\s+", text)
    out = " ".join(parts[:2]).strip()
    return out[:400] if out else text[:400]


def _chunk_digest(text: str, position: int | None) -> str:
    """Compose a grounded digest: span/domain/region/structure context + the
    specific residue's highlight line when the queried position is covered.

    Highlight lines look like ``  - R175: buried (low solvent accessibility ...)``.
    Pulling them in puts real residue facts into the answer so downstream
    keyword/grounding scoring reflects variant-level characterization.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    context_keys = ("Span:", "UniProt domain:", "Region context:",
                    "Secondary structure:", "Mean AlphaFold pLDDT")
    context = [ln for ln in lines if ln.startswith(context_keys)]
    res_line = ""
    if position is not None:
        for ln in lines:
            stripped = ln.lstrip("- ").strip()
            m = re.match(r"([A-Z])?(\d+):", stripped)
            if m and int(m.group(2)) == position:
                res_line = f"Residue {stripped}"
                break
    parts = context[:4] + ([res_line] if res_line else [])
    out = " ".join(parts).strip()
    return out[:600] if out else _lead_sentence(text)


def parsed_label(gene: str | None, parsed: dict[str, Any]) -> str:
    if gene and parsed.get("position"):
        return f"{gene} {parsed.get('ref_aa','')}{parsed['position']}{parsed.get('alt_aa','')}"
    return gene or "the queried variant"


def get_llm(kind: str | None = None) -> LLMClient:
    kind = kind or settings.resolve_llm()
    if kind == "anthropic":
        return AnthropicLLM()
    if kind == "mock":
        return MockLLM()
    raise ValueError(f"Unknown LLM backend: {kind!r}")

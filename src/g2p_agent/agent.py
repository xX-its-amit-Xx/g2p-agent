"""The Claude-powered (or mock) agent loop with tool use.

Drives a multi-turn tool-use conversation against an :class:`LLMClient`,
executes the requested tools, and parses the final message into a validated
:class:`AgentResponse`. Grounding is enforced *after* generation: any citation
to a chunk id the agent never actually retrieved is flagged in ``debug`` and
stripped, so the response can't claim support it doesn't have.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import settings
from .llm import LLMClient, get_llm
from .schemas import AgentResponse, Citation, Claim, Confidence
from .tools import TOOL_SCHEMAS, execute_tool

_PROMPT_DIR = Path(__file__).parent / "prompts"


@lru_cache(maxsize=1)
def _system_prompt() -> str:
    system = (_PROMPT_DIR / "system.md").read_text(encoding="utf-8")
    few_shot = (_PROMPT_DIR / "few_shot.md").read_text(encoding="utf-8")
    return f"{system}\n\n---\n{few_shot}"


def _extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fence.group(1) if fence else None
    if candidate is None:
        start = text.find("{")
        end = text.rfind("}")
        candidate = text[start : end + 1] if start != -1 and end > start else None
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


class Agent:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or get_llm()

    def ask(self, question: str, *, max_turns: int | None = None) -> AgentResponse:
        max_turns = max_turns or settings.max_tool_turns
        messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
        retrieved_ids: set[str] = set()
        tool_trace: list[dict[str, Any]] = []

        turns = 0
        reply = None
        while turns < max_turns:
            turns += 1
            reply = self.llm.message(
                system=_system_prompt(),
                messages=messages,
                tools=TOOL_SCHEMAS,
                model=settings.agent_model,
            )
            if reply.stop_reason != "tool_use" or not reply.tool_calls:
                break

            # record assistant turn (text + tool_use blocks)
            assistant_content: list[dict[str, Any]] = []
            if reply.text:
                assistant_content.append({"type": "text", "text": reply.text})
            for tc in reply.tool_calls:
                assistant_content.append(
                    {"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["input"]}
                )
            messages.append({"role": "assistant", "content": assistant_content})

            # execute tools, gather results
            tool_results: list[dict[str, Any]] = []
            for tc in reply.tool_calls:
                result_json = execute_tool(tc["name"], tc["input"])
                tool_trace.append({"name": tc["name"], "input": tc["input"]})
                retrieved_ids.update(_ids_in(result_json))
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc["id"],
                    "content": [{"type": "text", "text": result_json}],
                })
            messages.append({"role": "user", "content": tool_results})

        final_text = reply.text if reply else ""
        response = self._parse(final_text, retrieved_ids)
        response.debug.update({
            "llm_backend": self.llm.backend,
            "agent_model": settings.agent_model,
            "turns": turns,
            "tool_calls": tool_trace,
            "n_retrieved": len(retrieved_ids),
            "retrieved_ids": sorted(retrieved_ids),
        })
        return response

    def _parse(self, text: str, retrieved_ids: set[str]) -> AgentResponse:
        data = _extract_json(text)
        if data is None:
            return AgentResponse(
                answer=text or "I don't know based on G2P data.",
                confidence=Confidence.low,
                confidence_reasoning="Model did not return parseable structured output.",
                insufficient_evidence=True,
                debug={"parse_error": True, "raw": text[:500]},
            )

        claims_in = data.get("claims", []) or []
        claims: list[Claim] = []
        ungrounded: list[str] = []
        for c in claims_in:
            cits = []
            for cit in c.get("citations", []) or []:
                cid = cit.get("chunk_id")
                if cid and cid in retrieved_ids:
                    cits.append(Citation(chunk_id=cid, quote=cit.get("quote")))
                elif cid:
                    ungrounded.append(cid)  # cited but never retrieved -> drop
            claims.append(Claim(text=c.get("text", ""), citations=cits))

        cited = sorted({cit.chunk_id for cl in claims for cit in cl.citations})
        try:
            conf = Confidence(data.get("confidence", "low"))
        except ValueError:
            conf = Confidence.low

        resp = AgentResponse(
            answer=data.get("answer", ""),
            claims=claims,
            confidence=conf,
            confidence_reasoning=data.get("confidence_reasoning", ""),
            cited_chunk_ids=cited or data.get("cited_chunk_ids", []),
            insufficient_evidence=bool(data.get("insufficient_evidence", False)),
        )
        if ungrounded:
            resp.debug["dropped_ungrounded_citations"] = ungrounded
        # Safety net: a confident answer with zero grounded claims is downgraded.
        if not resp.insufficient_evidence and not claims and conf != Confidence.low:
            resp.confidence = Confidence.low
            resp.confidence_reasoning += " (downgraded: no grounded claims)."
        return resp


def ask(question: str) -> AgentResponse:
    """Convenience one-shot entry point."""
    return Agent().ask(question)


def _ids_in(result_json: str) -> set[str]:
    try:
        data = json.loads(result_json)
    except json.JSONDecodeError:
        return set()
    ids: set[str] = set()
    for ch in data.get("chunks", []) if isinstance(data, dict) else []:
        if isinstance(ch, dict) and "id" in ch:
            ids.add(ch["id"])
    return ids

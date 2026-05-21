"""Pydantic models for structured I/O across g2p-agent.

These models are the contract between the retriever, the agent, the tools, and
the evaluation harness. The agent is required to emit an :class:`AgentResponse`
whose every :class:`Claim` cites at least one retrieved :class:`Chunk` by id.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Confidence(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class VariantType(str, Enum):
    missense = "missense"
    nonsense = "nonsense"
    indel = "indel"
    splicing = "splicing"
    other = "other"


class Mechanism(str, Enum):
    stability = "stability"
    binding_site = "binding-site"
    ptm_site = "post-translational-modification site"
    splicing = "splicing"
    other = "other"


class VariantQuery(BaseModel):
    """A parsed natural-language question about a variant.

    Most fields are optional because users ask incomplete questions; the agent
    fills in what it can and the tools degrade gracefully.
    """

    question: str
    gene: str | None = None
    uniprot_id: str | None = None
    position: int | None = None
    ref_aa: str | None = Field(default=None, description="reference (wild-type) amino acid, 1-letter")
    alt_aa: str | None = Field(default=None, description="alternate (variant) amino acid, 1-letter")
    variant_type: VariantType | None = None

    @field_validator("gene")
    @classmethod
    def _upper_gene(cls, v: str | None) -> str | None:
        return v.upper() if v else v

    @property
    def hgvs_p(self) -> str | None:
        """Best-effort short protein change label, e.g. 'R175H'."""
        if self.ref_aa and self.position and self.alt_aa:
            return f"{self.ref_aa}{self.position}{self.alt_aa}"
        return None


class Chunk(BaseModel):
    """A retrievable unit of G2P knowledge with rich metadata."""

    id: str
    text: str
    gene: str
    uniprot_id: str
    start: int = Field(description="1-based residue start of the chunk")
    end: int = Field(description="1-based residue end of the chunk")
    domain: str | None = None
    chunk_kind: str = Field(default="domain", description="domain | cluster | summary")
    metadata: dict[str, Any] = Field(default_factory=dict)

    def covers(self, position: int | None) -> bool:
        return position is not None and self.start <= position <= self.end


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    components: dict[str, float] = Field(default_factory=dict)


class Citation(BaseModel):
    chunk_id: str
    quote: str | None = Field(default=None, description="short supporting span from the chunk")


class Claim(BaseModel):
    """A single factual assertion in the answer, with its supporting citations."""

    text: str
    citations: list[Citation] = Field(default_factory=list)

    @property
    def is_grounded(self) -> bool:
        return len(self.citations) > 0


class AgentResponse(BaseModel):
    """The structured answer the agent returns."""

    answer: str
    claims: list[Claim] = Field(default_factory=list)
    confidence: Confidence = Confidence.low
    confidence_reasoning: str = ""
    cited_chunk_ids: list[str] = Field(default_factory=list)
    insufficient_evidence: bool = Field(
        default=False,
        description="True when the agent declined to answer for lack of G2P support.",
    )
    debug: dict[str, Any] = Field(default_factory=dict)

    @property
    def grounding_rate(self) -> float:
        if not self.claims:
            return 0.0
        return sum(c.is_grounded for c in self.claims) / len(self.claims)


# --- evaluation models -------------------------------------------------------
class BenchmarkItem(BaseModel):
    id: str
    question: str
    gene: str
    variant: str | None = None
    variant_type: VariantType
    mechanism: Mechanism
    gold_answer: str
    gold_keywords: list[str] = Field(default_factory=list)
    expect_answerable: bool = True
    notes: str | None = None


class JudgeVerdict(BaseModel):
    task_success: bool
    grounding_rate: float
    hallucination_rate: float
    confidence_appropriate: bool
    rationale: str


class EvalRecord(BaseModel):
    item: BenchmarkItem
    response: AgentResponse
    verdict: JudgeVerdict
    retrieval_hit: bool = Field(
        default=False, description="did retrieval return any chunk for the target gene/position"
    )

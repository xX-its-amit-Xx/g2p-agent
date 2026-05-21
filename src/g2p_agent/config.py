"""Central configuration for g2p-agent.

Everything that a user might reasonably want to tune lives here and can be
overridden by environment variables, so the rest of the codebase never reads
``os.environ`` directly.

Backend selection is deliberately *graceful*: if neither an Anthropic API key
nor sentence-transformers/torch are available, the agent transparently falls
back to deterministic offline backends (``MockLLM`` / ``HashingEmbedder``) so
the full pipeline still runs and produces reproducible numbers. Which backend
was actually used is always surfaced in :class:`AgentResponse.debug`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --- canonical gene -> UniProt mapping --------------------------------------
# g2papi requires both a gene symbol and a canonical UniProt accession. These
# are the well-characterized disease genes we ship as the default ingest set.
# (UniProt canonical accessions, verified against uniprot.org.)
DEFAULT_GENE_UNIPROT: dict[str, str] = {
    "TP53": "P04637",
    "BRCA1": "P38398",
    "BRCA2": "P51587",
    "CFTR": "P13569",
    "KCNQ1": "P51787",
    "SERPING1": "P05155",
    "UMOD": "P07911",
    "MUC1": "P15941",
    "PTEN": "P60484",
    "MLH1": "P40692",
    "MSH2": "P43246",
    "VHL": "P40337",
    "RB1": "P06400",
    "HBB": "P68871",
    "LDLR": "P01130",
    "SCN5A": "Q14524",
    "KCNH2": "Q12809",
    "RYR1": "P21817",
    "G6PD": "P11413",
    "F8": "P00451",
    "F9": "P00740",
    "DMD": "P11532",
    "FBN1": "P35555",
    "COL1A1": "P02452",
    "APC": "P25054",
    "NF1": "P21359",
    "RET": "P07949",
    "KRAS": "P01116",
    "EGFR": "P00533",
    "PIK3CA": "P42336",
    "PCSK9": "Q8NBP7",
    "GBA": "P04062",
    "HEXA": "P06865",
    "PAH": "P00439",
    "ATP7B": "P35670",
    "SOD1": "P00441",
    "HTT": "P42858",
    "MECP2": "P51608",
    "GJB2": "P29033",
    "TTR": "P02766",
    "APOE": "P02649",
    "INS": "P01308",
    "GCK": "P35557",
    "ABCA4": "P78363",
    "USH2A": "O75445",
    "MYH7": "P12883",
    "MYBPC3": "Q14896",
    "LMNA": "P02545",
    "DSP": "P15924",
    "RYR2": "Q92736",
    "CACNA1C": "Q13936",
}

# Genes used for the shipped baseline evaluation. Kept smaller than the full
# default set so a cold ``ingest`` + ``eval`` finishes quickly and the benchmark
# questions all have backing data. Override with G2P_GENES.
BASELINE_GENES: tuple[str, ...] = (
    "TP53",
    "BRCA1",
    "CFTR",
    "KCNQ1",
    "SERPING1",
    "UMOD",
    "MUC1",
    "PTEN",
    "VHL",
    "LDLR",
)


def _project_root() -> Path:
    # src/g2p_agent/config.py -> project root is three parents up.
    return Path(__file__).resolve().parents[2]


@dataclass
class Settings:
    """Runtime settings, populated from environment variables with defaults."""

    # --- data / storage -----------------------------------------------------
    root: Path = field(default_factory=_project_root)
    raw_dir: Path = field(init=False)
    chroma_dir: Path = field(init=False)
    collection: str = os.environ.get("G2P_COLLECTION", "g2p_variants")

    # --- embeddings ---------------------------------------------------------
    # "auto" picks BGE if sentence-transformers is importable, else "hash".
    embedder: str = os.environ.get("G2P_EMBEDDER", "auto")
    embedding_model: str = os.environ.get("G2P_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    embedding_dim: int = int(os.environ.get("G2P_EMBEDDING_DIM", "384"))

    # --- retrieval ----------------------------------------------------------
    top_k: int = int(os.environ.get("G2P_TOP_K", "8"))
    candidate_k: int = int(os.environ.get("G2P_CANDIDATE_K", "30"))
    rrf_k: int = int(os.environ.get("G2P_RRF_K", "60"))
    rerank: bool = os.environ.get("G2P_RERANK", "1") not in ("0", "false", "False")

    # --- LLM ----------------------------------------------------------------
    # "auto" uses Anthropic if ANTHROPIC_API_KEY is set, else the mock backend.
    llm: str = os.environ.get("G2P_LLM", "auto")
    agent_model: str = os.environ.get("G2P_AGENT_MODEL", "claude-sonnet-4-5")
    judge_model: str = os.environ.get("G2P_JUDGE_MODEL", "claude-opus-4-1")
    max_tokens: int = int(os.environ.get("G2P_MAX_TOKENS", "2048"))
    max_tool_turns: int = int(os.environ.get("G2P_MAX_TOOL_TURNS", "6"))

    # --- chunking -----------------------------------------------------------
    cluster_gap: int = int(os.environ.get("G2P_CLUSTER_GAP", "8"))

    def __post_init__(self) -> None:
        self.raw_dir = Path(os.environ.get("G2P_RAW_DIR", self.root / "data" / "raw"))
        self.chroma_dir = Path(os.environ.get("G2P_CHROMA_DIR", self.root / "data" / "chroma"))
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)

    # --- resolved backend choices ------------------------------------------
    def resolve_embedder(self) -> str:
        if self.embedder != "auto":
            return self.embedder
        try:
            import sentence_transformers  # noqa: F401

            return "bge"
        except Exception:
            return "hash"

    def resolve_llm(self) -> str:
        if self.llm != "auto":
            return self.llm
        return "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "mock"

    def gene_uniprot(self, genes: list[str] | None = None) -> dict[str, str]:
        """Return gene->uniprot for the requested genes (default: baseline set)."""
        env_genes = os.environ.get("G2P_GENES")
        if genes is None and env_genes:
            genes = [g.strip().upper() for g in env_genes.split(",") if g.strip()]
        if genes is None:
            genes = list(BASELINE_GENES)
        out: dict[str, str] = {}
        for g in genes:
            g = g.upper()
            if g not in DEFAULT_GENE_UNIPROT:
                raise KeyError(
                    f"No UniProt accession known for gene {g!r}. "
                    "Add it to DEFAULT_GENE_UNIPROT in config.py."
                )
            out[g] = DEFAULT_GENE_UNIPROT[g]
        return out


settings = Settings()

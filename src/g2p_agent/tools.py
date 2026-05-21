"""Tools the agent can call.

Each tool returns a JSON string whose top-level ``chunks`` list contains
retrieved units with stable ``id``s — these ids are what the agent must cite.
Residue- and structure-level tools add extra grounded facts alongside.

The same schemas drive the real Anthropic tool-use API and the offline mock.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .config import settings
from .retrieve import get_retriever
from .schemas import ScoredChunk

# --------------------------------------------------------------------------- #
# Anthropic tool schemas
# --------------------------------------------------------------------------- #
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "search_variants",
        "description": (
            "Hybrid semantic + keyword search over G2P protein-feature chunks. "
            "Use this first for any question. Returns chunks with ids you MUST "
            "cite. Optionally filter by gene symbol."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language or keyword query."},
                "gene": {"type": "string", "description": "Optional gene symbol filter, e.g. TP53."},
                "top_k": {"type": "integer", "description": "Max chunks to return (default 8)."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_variant_context",
        "description": (
            "Fetch focused context for a specific residue/variant: the G2P chunk(s) "
            "covering that position plus per-residue facts (wild-type AA, domain, "
            "annotated sites, AlphaFold pLDDT, solvent accessibility). Cite the "
            "returned chunk ids."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "gene": {"type": "string"},
                "position": {"type": "integer", "description": "1-based residue number."},
                "aa_change": {"type": "string", "description": "e.g. R175H or R213* (optional)."},
            },
            "required": ["gene", "position"],
        },
    },
    {
        "name": "get_protein_structure_annotations",
        "description": (
            "Per-residue structural annotations (AlphaFold pLDDT, secondary "
            "structure, solvent accessibility, druggable pockets) for a UniProt "
            "accession over a residue range. Useful for stability/folding questions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "uniprot_id": {"type": "string"},
                "residue_range": {
                    "type": "string",
                    "description": "Inclusive range 'start-end', e.g. '170-180'.",
                },
            },
            "required": ["uniprot_id", "residue_range"],
        },
    },
]


def _clean(v: Any) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    s = str(v).strip()
    return "" if s.lower() in ("nan", "-", "") else s


def _scored_to_payload(scored: list[ScoredChunk]) -> list[dict[str, Any]]:
    return [
        {
            "id": s.chunk.id,
            "gene": s.chunk.gene,
            "score": round(s.score, 5),
            "text": s.chunk.text,
            "metadata": {
                "gene": s.chunk.gene,
                "uniprot_id": s.chunk.uniprot_id,
                "start": s.chunk.start,
                "end": s.chunk.end,
                "domain": s.chunk.domain,
                "chunk_kind": s.chunk.chunk_kind,
            },
        }
        for s in scored
    ]


# --------------------------------------------------------------------------- #
# Tool implementations
# --------------------------------------------------------------------------- #
def search_variants(query: str, gene: str | None = None, top_k: int | None = None) -> dict[str, Any]:
    retriever = get_retriever()
    scored = retriever.search(query, gene=gene, top_k=top_k or settings.top_k)
    return {"query": query, "gene": gene, "n": len(scored), "chunks": _scored_to_payload(scored)}


def get_variant_context(gene: str, position: int, aa_change: str | None = None) -> dict[str, Any]:
    retriever = get_retriever()
    scored = retriever.search(
        f"{gene} residue {position} {aa_change or ''}".strip(),
        gene=gene, position=position, top_k=5,
    )
    covering = [s for s in scored if s.chunk.covers(position)]
    facts = _residue_facts(gene, position)
    return {
        "gene": gene,
        "position": position,
        "aa_change": aa_change,
        "residue_facts": facts,
        "n": len(scored),
        "chunks": _scored_to_payload(covering or scored),
    }


def get_protein_structure_annotations(uniprot_id: str, residue_range: str) -> dict[str, Any]:
    df = _load_by_uniprot(uniprot_id)
    if df is None:
        return {"error": f"No ingested data for UniProt {uniprot_id}.", "chunks": []}
    try:
        lo, hi = (int(x) for x in residue_range.split("-"))
    except ValueError:
        return {"error": f"Bad residue_range {residue_range!r}; expected 'start-end'.", "chunks": []}
    span = df[(df["residueId"] >= lo) & (df["residueId"] <= hi)]
    rows = []
    for _, r in span.iterrows():
        rows.append({
            "residue": int(r["residueId"]),
            "aa": _clean(r.get("AA")),
            "plddt": _clean(r.get("AlphaFold confidence (pLDDT)")),
            "secondary_structure": _clean(r.get("Secondary structure (DSSP 3-state)*")),
            "asa": _clean(r.get("Accessible surface area (Å²)*")),
            "pocket": _clean(r.get("p2rank: pocket probability*")) or _clean(r.get("fpocket: druggability score*")),
        })
    # also surface the covering retrieval chunk ids so claims stay citeable
    gene = _gene_for_uniprot(uniprot_id)
    scored = get_retriever().search(f"{gene} residues {lo}-{hi} structure", gene=gene, top_k=3)
    return {
        "uniprot_id": uniprot_id,
        "residue_range": residue_range,
        "annotations": rows,
        "chunks": _scored_to_payload(scored),
    }


# --------------------------------------------------------------------------- #
# Raw-table helpers (read the cached G2P TSVs)
# --------------------------------------------------------------------------- #
def _gene_for_uniprot(uniprot_id: str) -> str | None:
    for g, u in settings.gene_uniprot(list(settings.gene_uniprot().keys())).items():
        if u == uniprot_id:
            return g
    # fall back to scanning the full default map
    from .config import DEFAULT_GENE_UNIPROT

    for g, u in DEFAULT_GENE_UNIPROT.items():
        if u == uniprot_id:
            return g
    return None


def _load_by_uniprot(uniprot_id: str) -> pd.DataFrame | None:
    matches = list(Path(settings.raw_dir).glob(f"*_{uniprot_id}.tsv"))
    if not matches:
        return None
    return pd.read_csv(matches[0], sep="\t")


def _residue_facts(gene: str, position: int) -> dict[str, Any]:
    from .config import DEFAULT_GENE_UNIPROT

    uniprot = DEFAULT_GENE_UNIPROT.get(gene.upper())
    df = _load_by_uniprot(uniprot) if uniprot else None
    if df is None:
        return {}
    row = df[df["residueId"] == position]
    if row.empty:
        return {}
    r = row.iloc[0]
    facts: dict[str, Any] = {
        "wild_type_aa": _clean(r.get("AA")),
        "domain": _clean(r.get("Domain (UniProt)")) or _clean(r.get("Region (UniProt)")),
        "plddt": _clean(r.get("AlphaFold confidence (pLDDT)")),
        "asa": _clean(r.get("Accessible surface area (Å²)*")),
        "secondary_structure": _clean(r.get("Secondary structure (DSSP 3-state)*")),
    }
    for col in ("Active site (UniProt)", "Binding site (UniProt)", "DNA binding (UniProt)",
                "Disulfide bond (UniProt)", "Modified residue (UniProt)"):
        v = _clean(r.get(col))
        if v:
            facts[col] = v
    return {k: v for k, v in facts.items() if v}


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
_DISPATCH = {
    "search_variants": search_variants,
    "get_variant_context": get_variant_context,
    "get_protein_structure_annotations": get_protein_structure_annotations,
}


def execute_tool(name: str, tool_input: dict[str, Any]) -> str:
    fn = _DISPATCH.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool {name!r}", "chunks": []})
    try:
        return json.dumps(fn(**tool_input))
    except Exception as e:  # surface tool errors to the model rather than crashing
        return json.dumps({"error": f"{type(e).__name__}: {e}", "chunks": []})

"""Ingestion: G2P portal -> chunks -> Chroma.

Pipeline
--------
1. Pull the per-residue protein feature table for each (gene, UniProt) via
   ``g2papi.get_protein_features`` and cache the raw TSV under ``data/raw/``
   (so re-ingestion is offline and reproducible).
2. Segment each protein into chunks. Two chunk kinds:
     * ``domain``  – a contiguous span sharing the same UniProt Domain/Region.
     * ``cluster`` – fixed-width windows over the regions with no named domain,
                     which also act as "variant clusters" of nearby residues.
   Plus one ``summary`` chunk per protein.
3. For each chunk, synthesize an information-dense, *grounded* text from the
   actual feature columns (active/binding sites, DNA binding, disulfide bonds,
   PTMs, AlphaFold pLDDT, solvent accessibility, secondary structure).
4. Embed with the configured embedder and upsert into a persistent Chroma
   collection with rich scalar metadata for filtered retrieval.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import settings
from .embeddings import get_embedder

# Per-residue site columns: a residue gets its own highlight line if any are set.
SITE_COLUMNS = {
    "Active site (UniProt)": "active site",
    "Binding site (UniProt)": "binding site",
    "Disulfide bond (UniProt)": "disulfide bond",
    "Zinc finger (UniProt)": "zinc finger residue",
    "Modified residue (UniProt)": "modified residue",
    "Glycosylation (UniProt)": "glycosylation site",
    "Lipidation (UniProt)": "lipidation site",
    "Motif (UniProt)": "sequence motif",
    "Site (UniProt)": "functional site",
    "Mutagenesis (UniProt)": "characterized mutagenesis",
}
PTM_COLUMNS = {
    "Acetylation": "acetylation",
    "Phosphorylation": "phosphorylation",
    "Ubiquitination": "ubiquitination",
    "SUMOylation": "SUMOylation",
    "Methylation": "methylation",
    "O-GlcNAc": "O-GlcNAc",
    "O-GalNAc": "O-GalNAc",
    "Disease-associated PTMs": "disease-associated PTM",
}
# Span-level context: region/topology annotations summarized once per chunk
# (not per residue) so chunk text stays clean and informative.
REGION_CONTEXT_COLUMNS = {
    "DNA binding (UniProt)": "within UniProt DNA-binding region",
    "Transmembrane (UniProt)": "transmembrane segment",
    "Intramembrane (UniProt)": "intramembrane segment",
    "Signal (UniProt)": "signal peptide",
    "Transit peptide (UniProt)": "transit peptide",
    "Coiled coil (UniProt)": "coiled-coil",
    "Propeptide (UniProt)": "propeptide",
}
RAW_REGION_COLUMNS = ["Region (UniProt)", "Topological domain (UniProt)"]
# Segmentation key uses only true structural domains; everything else windows.
DOMAIN_COLS = ["Domain (UniProt)"]
MAX_WINDOW = 40  # residues per non-domain cluster chunk


@dataclass
class ChunkRow:
    id: str
    text: str
    gene: str
    uniprot_id: str
    start: int
    end: int
    domain: str
    chunk_kind: str
    length: int
    mean_plddt: float
    has_active_site: bool
    has_binding_site: bool
    has_dna_binding: bool
    has_disulfide: bool
    has_ptm: bool


# --------------------------------------------------------------------------- #
# Fetching / caching
# --------------------------------------------------------------------------- #
def _cache_path(gene: str, uniprot: str) -> Path:
    return settings.raw_dir / f"{gene}_{uniprot}.tsv"


def fetch_protein_features(gene: str, uniprot: str, *, use_cache: bool = True) -> pd.DataFrame:
    """Return per-residue feature table, fetching from G2P or local cache."""
    path = _cache_path(gene, uniprot)
    if use_cache and path.exists():
        return pd.read_csv(path, sep="\t")
    import g2papi  # lazy: only needed for live fetch

    df = g2papi.get_protein_features(gene, uniprot)
    df.to_csv(path, sep="\t", index=False)
    return df


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def _clean(val: Any) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return ""
    s = str(val).strip()
    return "" if s.lower() in ("nan", "-", "") else s


def _domain_of_row(row: pd.Series) -> str:
    for col in DOMAIN_COLS:
        if col in row and _clean(row[col]):
            return _clean(row[col])
    return ""


def _segments(df: pd.DataFrame) -> list[tuple[int, int, str, str]]:
    """Yield (start, end, domain_label, kind) spans.

    Contiguous residues sharing a domain label form one ``domain`` span;
    label-less stretches are cut into ``cluster`` windows of <= MAX_WINDOW.
    """
    res_ids = df["residueId"].tolist()
    labels = [_domain_of_row(r) for _, r in df.iterrows()]
    segments: list[tuple[int, int, str, str]] = []

    i = 0
    n = len(res_ids)
    while i < n:
        label = labels[i]
        j = i
        while j + 1 < n and labels[j + 1] == label and res_ids[j + 1] == res_ids[j] + 1:
            j += 1
        start, end = res_ids[i], res_ids[j]
        if label:
            segments.append((start, end, label, "domain"))
        else:
            # split label-less span into windows
            s = start
            while s <= end:
                e = min(s + MAX_WINDOW - 1, end)
                segments.append((s, e, "", "cluster"))
                s = e + 1
        i = j + 1
    return segments


def _highlights(span_df: pd.DataFrame) -> list[str]:
    """Per-residue highlight lines for residues with a specific site/PTM, or buried.

    Region/topology membership is handled separately (see _region_context) so it
    is not repeated on every residue. No cap: a queried residue's line is always
    present in its chunk.
    """
    out: list[str] = []
    for _, row in span_df.iterrows():
        res = int(row["residueId"])
        aa = _clean(row.get("AA"))
        notes: list[str] = []
        for col, label in SITE_COLUMNS.items():
            v = _clean(row.get(col)) if col in row else ""
            if v:
                notes.append(f"{label} ({v})" if v.lower() not in label.lower() else label)
        ptms = [label for col, label in PTM_COLUMNS.items() if col in row and _clean(row.get(col))]
        if ptms:
            notes.append("PTM: " + ", ".join(sorted(set(ptms))))
        asa = _clean(row.get("Accessible surface area (Å²)*"))
        if asa:
            try:
                if float(asa) <= 10:
                    notes.append(f"buried (low solvent accessibility, ASA {asa} Å²)")
            except ValueError:
                pass
        if notes:
            out.append(f"  - {aa}{res}: " + "; ".join(notes))
    return out


def _region_context(span_df: pd.DataFrame) -> list[str]:
    """Span-level region/topology annotations, summarized once."""
    ctx: list[str] = []
    for col, label in REGION_CONTEXT_COLUMNS.items():
        if col in span_df and span_df[col].map(_clean).ne("").any():
            ctx.append(label)
    for col in RAW_REGION_COLUMNS:
        if col not in span_df:
            continue
        # Region values are often long ';'-joined annotations that vary slightly
        # per residue; split, dedupe atomic annotations, and summarize.
        atoms: list[str] = []
        for v in span_df[col]:
            cv = _clean(v)
            if not cv:
                continue
            for atom in cv.split(";"):
                atom = atom.strip()
                if atom and atom not in atoms:
                    atoms.append(atom)
        ctx.extend(f"Region: {a}" for a in atoms[:8])
    return ctx


def _col_present(span_df: pd.DataFrame, colnames: list[str]) -> bool:
    return any(c in span_df and span_df[c].map(_clean).ne("").any() for c in colnames)


def _mean_plddt(span_df: pd.DataFrame) -> float:
    col = "AlphaFold confidence (pLDDT)"
    if col not in span_df:
        return float("nan")
    vals = pd.to_numeric(span_df[col], errors="coerce").dropna()
    return round(float(vals.mean()), 2) if len(vals) else float("nan")


def _secondary_structure_summary(span_df: pd.DataFrame) -> str:
    col = "Secondary structure (DSSP 3-state)*"
    if col not in span_df:
        return ""
    vals = span_df[col].map(_clean)
    counts = vals[vals != ""].value_counts()
    if counts.empty:
        return ""
    top = counts.idxmax()
    mapping = {"H": "helix", "E": "strand/sheet", "C": "loop/coil"}
    label = mapping.get(str(top)[0], str(top))
    return f"predominantly {label}"


def build_chunks(gene: str, uniprot: str, df: pd.DataFrame) -> list[ChunkRow]:
    df = df.sort_values("residueId").reset_index(drop=True)
    seq_len = int(df["residueId"].max())
    chunks: list[ChunkRow] = []

    # per-protein summary chunk
    domains = sorted({_domain_of_row(r) for _, r in df.iterrows() if _domain_of_row(r)})
    summary_text = (
        f"[{gene} {uniprot} | protein summary | {seq_len} residues]\n"
        f"{gene} (UniProt {uniprot}) is {seq_len} residues long. "
        + (f"Annotated UniProt domains/regions: {'; '.join(domains)}. " if domains else "No named UniProt domains in this record. ")
        + "This summary indexes the protein for retrieval; residue-level detail is in the domain and cluster chunks."
    )
    chunks.append(ChunkRow(
        id=f"{gene}:{uniprot}:summary",
        text=summary_text, gene=gene, uniprot_id=uniprot, start=1, end=seq_len,
        domain="(summary)", chunk_kind="summary", length=seq_len,
        mean_plddt=_mean_plddt(df), has_active_site=False, has_binding_site=False,
        has_dna_binding=False, has_disulfide=False, has_ptm=False,
    ))

    for (start, end, label, kind) in _segments(df):
        span = df[(df["residueId"] >= start) & (df["residueId"] <= end)]
        highlights = _highlights(span)
        region_ctx = _region_context(span)
        plddt = _mean_plddt(span)
        ss = _secondary_structure_summary(span)
        header = f"[{gene} {uniprot} | residues {start}-{end}"
        header += f" | Domain: {label}]" if label else " | inter-domain region]"
        body_lines = [header,
                      f"Span: residues {start}–{end} of {gene} (UniProt {uniprot})."]
        if label:
            body_lines.append(f"UniProt domain: {label}.")
        if region_ctx:
            body_lines.append("Region context: " + "; ".join(region_ctx) + ".")
        if ss:
            body_lines.append(f"Secondary structure: {ss}.")
        if not math.isnan(plddt):
            folded = "well-folded/ordered" if plddt >= 70 else "low-confidence/likely-disordered"
            body_lines.append(f"Mean AlphaFold pLDDT {plddt} ({folded}).")
        if highlights:
            body_lines.append("Notable annotated residues in this span:")
            body_lines.extend(highlights)
        else:
            body_lines.append("No specifically annotated functional residues in this span.")
        text = "\n".join(body_lines)

        chunks.append(ChunkRow(
            id=f"{gene}:{uniprot}:{start}-{end}",
            text=text, gene=gene, uniprot_id=uniprot, start=int(start), end=int(end),
            domain=label or "(inter-domain)", chunk_kind=kind, length=int(end - start + 1),
            mean_plddt=plddt if not math.isnan(plddt) else -1.0,
            has_active_site=_col_present(span, ["Active site (UniProt)"]),
            has_binding_site=_col_present(span, ["Binding site (UniProt)", "Zinc finger (UniProt)"]),
            has_dna_binding=_col_present(span, ["DNA binding (UniProt)"]),
            has_disulfide=_col_present(span, ["Disulfide bond (UniProt)"]),
            has_ptm=_col_present(span, list(PTM_COLUMNS.keys()) + ["Modified residue (UniProt)"]),
        ))
    return chunks


# --------------------------------------------------------------------------- #
# Chroma write
# --------------------------------------------------------------------------- #
def get_collection(reset: bool = False):
    import chromadb

    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    if reset:
        try:
            client.delete_collection(settings.collection)
        except Exception:
            pass
    return client.get_or_create_collection(
        settings.collection, metadata={"hnsw:space": "cosine"}
    )


def _chunk_metadata(c: ChunkRow) -> dict[str, Any]:
    d = asdict(c)
    d.pop("text")
    d.pop("id")
    return d


def ingest(
    genes: list[str] | None = None,
    *,
    reset: bool = True,
    use_cache: bool = True,
    progress=lambda *_: None,
) -> dict[str, Any]:
    """Ingest the given genes (default: baseline set) into Chroma.

    Returns a manifest dict with counts and the resolved backend names.
    """
    gene_map = settings.gene_uniprot(genes)
    embedder = get_embedder()
    collection = get_collection(reset=reset)

    all_chunks: list[ChunkRow] = []
    per_gene: dict[str, int] = {}
    for gene, uniprot in gene_map.items():
        progress(f"fetch {gene} ({uniprot})")
        df = fetch_protein_features(gene, uniprot, use_cache=use_cache)
        chunks = build_chunks(gene, uniprot, df)
        per_gene[gene] = len(chunks)
        all_chunks.extend(chunks)
        progress(f"  -> {len(chunks)} chunks")

    progress(f"embedding {len(all_chunks)} chunks with {embedder.name}")
    vectors = embedder.encode([c.text for c in all_chunks], is_query=False)
    collection.upsert(
        ids=[c.id for c in all_chunks],
        documents=[c.text for c in all_chunks],
        embeddings=[v.tolist() for v in vectors],
        metadatas=[_chunk_metadata(c) for c in all_chunks],
    )

    manifest = {
        "genes": list(gene_map.keys()),
        "n_chunks": len(all_chunks),
        "per_gene": per_gene,
        "embedder": embedder.name,
        "collection": settings.collection,
        "chroma_dir": str(settings.chroma_dir),
    }
    (settings.chroma_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest

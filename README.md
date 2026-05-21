# g2p-agent

**A retrieval-augmented Claude agent over the Broad Institute's [Genomics 2 Proteins (G2P) portal](https://g2p.broadinstitute.org).**
Ask a natural-language question about a protein variant — *"what does the missense
variant R175H in TP53 do to protein stability?"* — and get a **cited,
confidence-scored** answer grounded in G2P feature data, or an honest
*"I don't know based on G2P data."*

> ⚠️ **Not for clinical use.** This is a research/exploration aid. It does not
> provide medical advice or variant pathogenicity classifications for patient care.

---

## Why this exists

The G2P portal maps genetic-screening outputs onto protein sequences and
structures: per-residue domains, active/binding sites, disulfide bonds, PTM
sites, AlphaFold pLDDT, solvent accessibility, druggable pockets, and more
([Kwon et al., *Nat Methods* 2024](https://doi.org/10.1038/s41592-024-02409-0)).
The data is extraordinarily rich, but the access path is *structured*: you query
by gene + UniProt accession and get back a wide per-residue table. To answer
"is R175H a folding mutant or a DNA-contact mutant?" you must know which columns
to read and how to interpret ASA, pLDDT, and proximity to a zinc site.

`g2p-agent` puts a **grounded language interface** in front of that table so a
clinical geneticist or structural biologist can ask in plain English and get a
synthesized answer **with citations back to the underlying G2P records** — and a
calibrated confidence so they know how much to trust it. The hard constraint:
**the agent may only assert what the retrieved G2P chunks support.**

---

## Architecture

```
                          ┌──────────────────────────────────────────────────────┐
                          │                     g2p-agent ask                     │
                          └──────────────────────────────────────────────────────┘
                                                  │  natural-language question
                                                  ▼
   ┌─────────────┐   gene/UniProt   ┌───────────────────────┐
   │  G2P portal │ ───────────────▶ │   ingest.py           │   chunk by domain +
   │ (g2papi)    │  per-residue TSV │  fetch · cache · chunk │   variant cluster,
   └─────────────┘                  └───────────┬───────────┘   derive grounded text
                                                 │ embed (BGE / hash fallback)
                                                 ▼
                                     ┌───────────────────────┐
                                     │   Chroma (persistent) │  rich scalar metadata:
                                     │   vector store        │  gene, span, domain,
                                     └───────────┬───────────┘  site/PTM/disulfide flags
                                                 │
              query ──▶  ┌──────────────────────────────────────────┐
                         │  retrieve.py                             │
                         │  dense (BGE)  +  sparse (BM25)           │
                         │            └──── RRF fusion ────┘        │
                         │            + residue/gene rerank         │
                         └──────────────────────┬───────────────────┘
                                                 │ top-k chunks (with ids)
                                                 ▼
   ┌──────────────────────────────────────────────────────────────────────────────┐
   │  agent.py  — Claude Sonnet tool-use loop (offline: deterministic MockLLM)      │
   │                                                                                │
   │   tools.py:  search_variants · get_variant_context ·                          │
   │              get_protein_structure_annotations                                │
   │                                                                                │
   │   rules:  (1) cite every claim by chunk id                                    │
   │           (2) calibrated confidence (high/medium/low) + reasoning             │
   │           (3) refuse ("I don't know based on G2P data") if unsupported        │
   └──────────────────────────────────────────┬───────────────────────────────────┘
                                                 ▼
                          AgentResponse {answer, claims[+citations], confidence, ...}
                                                 │
                                                 ▼
                         ┌───────────────────────────────────────────┐
                         │  eval.py — LLM-as-judge (Claude Opus) +    │
                         │  deterministic rubric judge                │
                         │  success · grounding · hallucination ·     │
                         │  calibration                               │
                         └───────────────────────────────────────────┘
```

### Backends degrade gracefully

Every LLM/embedding touchpoint is a swappable backend, auto-selected at runtime:

| component | default (production) | offline fallback | selector |
|---|---|---|---|
| embeddings | `BAAI/bge-small-en-v1.5` (sentence-transformers) | `HashingEmbedder` (pure-numpy, deterministic) | `G2P_EMBEDDER=auto\|bge\|hash` |
| agent LLM | Claude Sonnet (`anthropic`) | `MockLLM` (rule-based tool-user) | `G2P_LLM=auto\|anthropic\|mock` |
| judge LLM | Claude Opus | `MockJudge` (deterministic rubric) | resolved from `ANTHROPIC_API_KEY` |

`auto` uses the real backend when `ANTHROPIC_API_KEY` / sentence-transformers are
available, otherwise the offline one — so the **entire pipeline runs and produces
numbers with zero credentials**, and upgrades to live Claude with no code changes.
Run `g2p-agent info` to see what resolved.

---

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone <this repo> && cd g2p-agent
uv venv --python 3.12
uv pip install -e ".[dev]"          # core + dev
uv pip install -e ".[embeddings]"   # optional: real BGE embeddings (pulls torch)
```

For live Claude answers and the Opus judge:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## Quickstart

```bash
# 1. Pull + index the 10 baseline disease genes from the G2P portal
g2p-agent ingest
#    (or a custom set:)  g2p-agent ingest --genes "TP53,BRCA1,CFTR"

# 2. Ask
g2p-agent ask "What does the missense variant R175H in TP53 do to protein stability?"

# 3. Machine-readable output
g2p-agent ask "How does C124S affect the catalytic activity of PTEN?" --json

# 4. Show resolved backends / config
g2p-agent info

# 5. Run the benchmark
g2p-agent eval                       # or: python eval/run_eval.py --name baseline
```

Example (offline `mock` backend, abbreviated):

```
answer: TP53 R175H — residue 175 lies within the UniProt DNA-binding region and is
        buried (low solvent accessibility, ASA 6 Å²); a substitution here
        destabilizes the DNA-binding domain fold.
confidence: high — 2 retrieved chunk(s) directly cover the queried residue.
claims:  R175 buried, within DNA-binding region   →  TP53:P04637:161-200
```

---

## Evaluation

The harness (`src/g2p_agent/eval.py`, driver `eval/run_eval.py`) scores the agent
on `eval/benchmark.jsonl` — **32 hand-curated questions** spanning variant types
(missense, nonsense, indel, splicing) and mechanism categories (stability,
binding-site, PTM-site, splicing), plus out-of-scope refusal probes.

**Metrics**
- **Task success** — did the answer correctly characterize the variant (gold-keyword coverage), or correctly refuse when unanswerable?
- **Grounding rate** — fraction of factual claims with a citation to a retrieved chunk (also enforced structurally: ungrounded citations are dropped).
- **Hallucination rate** — fraction of claims unsupported by any retrieved chunk.
- **Calibration** — accuracy within each stated-confidence bucket; count of overconfident answers.

**Judging** — `AnthropicJudge` (Claude Opus, LLM-as-judge) with a deterministic
`MockJudge` rubric used for the shipped baseline and as a manual-gold sanity layer.

### Current baseline scores

Shipped numbers in [`eval/results/baseline.md`](eval/results/baseline.md), produced
by actually running ingest → retrieve → agent → judge on **real G2P data** with
the **offline backends** (`embedder=hash`, `llm=mock`, `judge=mock`; no API key in
the build environment):

| metric | value |
|---|---|
| items | 32 |
| task success rate | **0.875** |
| grounding rate | **1.000** |
| hallucination rate | **0.125** |
| confidence-appropriate rate | **0.875** |
| retrieval hit rate | **0.906** |

| variant type | success | | mechanism | success |
|---|---|---|---|---|
| missense (22) | 1.000 | | stability (6) | 1.000 |
| nonsense (3) | 1.000 | | binding-site (8) | 1.000 |
| indel (3) | 1.000 | | PTM-site (9) | 1.000 |
| splicing (4) | **0.000** | | splicing (4) | **0.000** |

**Reading the splicing result honestly:** G2P protein-feature data contains *no*
splicing/intronic annotations, so the correct behavior on those 4 items is
refusal. The deterministic `mock` backend retrieves protein chunks and answers
anyway — the harness catches this as hallucination=1.0 and overconfidence on
exactly those items. This is the primary case where the **live Claude backend is
expected to do better** (recognizing the retrieved chunks don't address splicing
and refusing). The eval is built to *expose* that gap, not hide it; re-run with
`ANTHROPIC_API_KEY` set to quantify it.

---

## Limitations

- **Not clinical.** No pathogenicity calls, no medical advice. A grounded literature/structure summary is not a clinical interpretation.
- **Coverage = whatever is ingested.** Answers are bounded by the genes you index and by what G2P annotates. No splicing/regulatory/non-coding information (it isn't in the protein-feature tables). ClinVar significance is not part of the G2P feature table and is therefore absent from chunk metadata.
- **The offline mock is a stand-in, not a model.** Baseline numbers from the `mock` backend exercise the full pipeline deterministically but do *not* reflect Claude's mechanistic reasoning. Treat them as a regression/grounding harness, not a measure of answer quality. Use the live backend for that.
- **LLM-as-judge bias.** The Opus judge can be lenient/sycophantic and shares blind spots with the agent. The `MockJudge` rubric and gold keywords provide a deterministic cross-check, but human review remains the gold standard.
- **Hashing embedder ≠ BGE.** Without the `embeddings` extra, dense retrieval uses a lexical hashing projection; install `[embeddings]` for semantic BGE vectors.

---

## How to extend

See **[AGENTS.md](AGENTS.md)** for code conventions and step-by-step guides to:
add a new agent tool, add a new evaluation metric, and ingest a new data source
or gene. Designed for Claude Code sub-agents to pick up and extend.

---

## Citation

If you use G2P data, please cite:

> Kwon, S., Safer, J., Nguyen, D.T. et al. **Genomics 2 Proteins portal: a resource
> and discovery tool for linking genetic screening outputs to protein sequences and
> structures.** *Nat Methods* **21**, 1947–1957 (2024).
> https://doi.org/10.1038/s41592-024-02409-0

```bibtex
@article{kwon_genomics_2024,
  author  = {Kwon, Seulki and Safer, Jordan and Nguyen, Duyen T. and Hoksza, David
             and May, Patrick and Arbesfeld, Jeremy A. and Rubin, Alan F. and
             Campbell, Arthur J. and Burgin, Alex and Iqbal, Sumaiya},
  title   = {Genomics 2 Proteins portal: a resource and discovery tool for linking
             genetic screening outputs to protein sequences and structures},
  journal = {Nature Methods}, volume = {21}, pages = {1947--1957}, year = {2024},
  doi     = {10.1038/s41592-024-02409-0}
}
```

Data access via [`g2papi`](https://github.com/broadinstitute/g2papi). This project
is independent of and not endorsed by the Broad Institute.

## License

MIT — see [LICENSE](LICENSE).

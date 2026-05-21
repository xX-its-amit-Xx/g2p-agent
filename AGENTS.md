# AGENTS.md — extending g2p-agent

Instructions for Claude Code (and human) contributors working on this repo. Read
this before adding features; it captures the conventions and the three most
common extension paths.

## Repo map

```
src/g2p_agent/
  config.py      Settings + gene→UniProt map + backend resolution (single source of truth)
  schemas.py     Pydantic models — the contracts between every module
  embeddings.py  Embedder backends: BGEEmbedder (default) / HashingEmbedder (offline)
  llm.py         LLMClient backends: AnthropicLLM / MockLLM (rule-based tool-user)
  ingest.py      G2P fetch+cache → chunking → Chroma upsert
  retrieve.py    Hybrid dense+BM25 retrieval, RRF fusion, rerank
  tools.py       Agent tools + Anthropic tool schemas + dispatch
  agent.py       Tool-use loop; parses + validates the final AgentResponse
  eval.py        Judges (Anthropic/Mock) + metrics harness
  prompts/       system.md (the rules) + few_shot.md
tests/           pytest (offline; live tests gated behind the `live` marker)
eval/            benchmark.jsonl, run_eval.py, results/
```

## Code conventions

- **Python 3.11+**, `from __future__ import annotations` at the top of every module.
- **Pydantic v2** for all structured I/O. If data crosses a module boundary, it should be a model in `schemas.py`.
- **No `os.environ` outside `config.py`.** Add a field to `Settings` and read `settings.x`.
- **Backends stay swappable and offline-capable.** Anything calling Claude or a heavy ML model must have a deterministic fallback selected via `settings.resolve_*()`, so `pytest` runs with no network/API key. Lazy-import heavy deps (`anthropic`, `sentence_transformers`, `chromadb`) *inside* functions, never at module top level.
- **Grounding is sacred.** Never let the agent assert something not traceable to a retrieved chunk id. The `agent.py` validator drops citations to unretrieved chunks — keep that invariant if you refactor.
- **Lint:** `ruff check src tests`. Line length 100, ignore E501.
- **Tests:** offline by default. Mark anything needing `ANTHROPIC_API_KEY` or live G2P network with `@pytest.mark.live`.

Run the gate before committing:
```bash
ruff check src tests && pytest -q
```

## How to add a new agent tool

1. **Implement** the function in `tools.py`. Return a JSON-serializable `dict`
   whose top-level `chunks` list holds retrieved units with stable `id`s (these
   are what the agent cites). Add extra grounded facts alongside `chunks`.
2. **Declare the schema** in `TOOL_SCHEMAS` (Anthropic tool format: `name`,
   `description`, `input_schema`). Write the description for the model — say when
   to use it and that returned ids must be cited.
3. **Register** it in the `_DISPATCH` map.
4. **Teach the mock** (optional but recommended): if the tool should be called in
   a particular phase, add the branch in `MockLLM.message` so the offline path and
   tests exercise it. The real Claude backend needs no change.
5. **Test** it in `tests/` (offline) and, if useful, add a benchmark item that
   forces its use.

Tools must be read-only over ingested data. Do not let a tool mutate the Chroma
collection or fetch arbitrary URLs.

## How to add a new evaluation metric

1. Compute the per-item signal in `eval.py`. If the judge produces it, extend
   `JudgeVerdict` (in `schemas.py`) and both `MockJudge.judge` *and*
   `AnthropicJudge` (+ the `_JUDGE_SYSTEM` prompt) so the deterministic and live
   judges stay in sync.
2. Aggregate it in `_aggregate()` and add it to the returned `metrics` dict.
3. Surface it in `eval/run_eval.py::to_markdown` (a table row) and in the CLI
   `eval` command's table if it's headline-worthy.
4. Add or adjust benchmark items in `eval/benchmark.jsonl` that stress the new
   metric, and re-run `python eval/run_eval.py --name baseline` to refresh
   `eval/results/baseline.md`.

Keep `MockJudge` deterministic — no randomness, no clock — so baseline numbers are
reproducible across machines.

## How to add a new ingested data source / gene

**A new gene (same G2P source):** add its symbol→UniProt accession to
`DEFAULT_GENE_UNIPROT` in `config.py` (use the *canonical* UniProt accession),
optionally add it to `BASELINE_GENES`, then `g2p-agent ingest --genes NEWGENE`.

**A new G2P endpoint or external source:**
1. Add a fetch+cache function in `ingest.py` modeled on `fetch_protein_features`
   (cache raw output under `data/raw/` so re-ingestion is offline/reproducible).
2. Map the raw records into `ChunkRow`s in a `build_chunks`-style function. Each
   chunk needs: a stable `id` (`GENE:UNIPROT:span` convention), grounded `text`
   synthesized from real fields, `start`/`end`, and **scalar-only** metadata
   (Chroma rejects nested values). Add new metadata flags to `ChunkRow` and
   `_chunk_metadata` together.
3. If residue-level tools should read the new source, extend the raw-table helpers
   in `tools.py` (`_load_by_uniprot`, `_residue_facts`).
4. Re-`ingest` and re-run the eval.

**Chunking philosophy:** segment on true structural domains; window the rest into
fixed `cluster` spans (`MAX_WINDOW`); summarize span-level region/topology once and
list per-residue facts (sites, PTMs, buried cores) so a queried residue's evidence
is always present in its chunk.

## Gotchas

- `get_retriever()` is `lru_cache`d per process; after re-ingesting in the same
  process, build a fresh `Retriever()` (the eval/CLI run in fresh processes).
- Chroma metadata must be `str|int|float|bool`. No `None`, lists, or dicts.
- The mock parses variant notation (`R175H`, `R213X`, `F508del`) in `llm._parse_variant`;
  extend the regexes there if you add notations the offline tests rely on.
- `manifest.json` in the Chroma dir records the last ingest (genes, chunk counts,
  embedder) — handy for debugging "why is retrieval empty?".

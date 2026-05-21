# Evaluation report: `baseline`

_Generated 2026-05-21T03:49:08+00:00_

## Backends

- LLM (agent): **mock** (`claude-sonnet-4-5`)
- Judge: **mock** (`claude-opus-4-1`)
- Embedder: **hash**

> When `llm`/`judge` show `mock`, no `ANTHROPIC_API_KEY` was present and the
> deterministic offline backends were used. Retrieval, fusion, chunking and
> the harness itself run identically with the live Claude backends.

## Headline metrics

| metric | value |
|---|---|
| items | 32 |
| task success rate | 0.875 |
| grounding rate | 1.000 |
| hallucination rate | 0.125 |
| confidence-appropriate rate | 0.875 |
| retrieval hit rate | 0.906 |
| overconfident answers | 4 |

## Calibration (accuracy within each stated-confidence bucket)

| confidence | n | accuracy |
|---|---|---|
| high | 29 | 0.862 |
| medium | 0 | n/a |
| low | 3 | 1.000 |

## Success by variant type

| variant type | n | success rate |
|---|---|---|
| missense | 22 | 1.000 |
| nonsense | 3 | 1.000 |
| indel | 3 | 1.000 |
| splicing | 4 | 0.000 |

## Success by mechanism

| mechanism | n | success rate |
|---|---|---|
| stability | 6 | 1.000 |
| binding-site | 8 | 1.000 |
| post-translational-modification site | 9 | 1.000 |
| other | 5 | 1.000 |
| splicing | 4 | 0.000 |

## Per-item results

| id | type | mech | success | ground | halluc | conf | conf-ok | refused |
|---|---|---|---|---|---|---|---|---|
| tp53-r175h | missense | stability | ✓ | 1.00 | 0.00 | high | ✓ |  |
| tp53-c176s | missense | binding-site | ✓ | 1.00 | 0.00 | high | ✓ |  |
| tp53-r248q | missense | binding-site | ✓ | 1.00 | 0.00 | high | ✓ |  |
| tp53-r273h | missense | binding-site | ✓ | 1.00 | 0.00 | high | ✓ |  |
| tp53-s15a | missense | post-translational-modification site | ✓ | 1.00 | 0.00 | high | ✓ |  |
| tp53-k120r | missense | post-translational-modification site | ✓ | 1.00 | 0.00 | high | ✓ |  |
| pten-c124s | missense | binding-site | ✓ | 1.00 | 0.00 | high | ✓ |  |
| pten-y68f | missense | post-translational-modification site | ✓ | 1.00 | 0.00 | high | ✓ |  |
| cftr-w401g | missense | binding-site | ✓ | 1.00 | 0.00 | high | ✓ |  |
| cftr-s45a | missense | post-translational-modification site | ✓ | 1.00 | 0.00 | high | ✓ |  |
| ldlr-c27y | missense | stability | ✓ | 1.00 | 0.00 | high | ✓ |  |
| serping1-c123r | missense | stability | ✓ | 1.00 | 0.00 | high | ✓ |  |
| umod-c32y | missense | stability | ✓ | 1.00 | 0.00 | high | ✓ |  |
| kcnq1-s27a | missense | post-translational-modification site | ✓ | 1.00 | 0.00 | high | ✓ |  |
| vhl-s68a | missense | post-translational-modification site | ✓ | 1.00 | 0.00 | high | ✓ |  |
| brca1-s114a | missense | post-translational-modification site | ✓ | 1.00 | 0.00 | high | ✓ |  |
| brca1-k56r | missense | post-translational-modification site | ✓ | 1.00 | 0.00 | high | ✓ |  |
| muc1-y1191f | missense | post-translational-modification site | ✓ | 1.00 | 0.00 | high | ✓ |  |
| kcnq1-q244r | missense | binding-site | ✓ | 1.00 | 0.00 | high | ✓ |  |
| tp53-r213x | nonsense | other | ✓ | 1.00 | 0.00 | high | ✓ |  |
| cftr-g542x | nonsense | other | ✓ | 1.00 | 0.00 | high | ✓ |  |
| pten-r130x | nonsense | other | ✓ | 1.00 | 0.00 | high | ✓ |  |
| cftr-f508del | indel | stability | ✓ | 1.00 | 0.00 | high | ✓ |  |
| ldlr-c27del | indel | stability | ✓ | 1.00 | 0.00 | high | ✓ |  |
| tp53-c176del | indel | binding-site | ✓ | 1.00 | 0.00 | high | ✓ |  |
| cftr-splice-1521 | splicing | splicing | ✗ | 1.00 | 1.00 | high | ✗ |  |
| brca1-splice-5074 | splicing | splicing | ✗ | 1.00 | 1.00 | high | ✗ |  |
| tp53-splice-intron | splicing | splicing | ✗ | 1.00 | 1.00 | high | ✗ |  |
| kcnq1-splice-syn | splicing | splicing | ✗ | 1.00 | 1.00 | high | ✗ |  |
| egfr-t790m-oos | missense | other | ✓ | 1.00 | 0.00 | low | ✓ | ✓ |
| kras-g12d-oos | missense | binding-site | ✓ | 1.00 | 0.00 | low | ✓ | ✓ |
| ret-m918t-oos | missense | other | ✓ | 1.00 | 0.00 | low | ✓ | ✓ |

## Methodology

- **Index:** 10 well-characterized disease genes pulled live from the G2P
  portal (`g2papi.get_protein_features`), chunked by UniProt domain + variant
  cluster, embedded, and stored in persistent Chroma.
- **Agent:** tool-using loop (`search_variants` → `get_variant_context` →
  compose), required to cite every claim by retrieved chunk id and to refuse
  ('I don't know based on G2P data') when no chunk supports an answer.
- **Judge:** scores task success (gold-keyword coverage / correct refusal),
  grounding, hallucination, and confidence appropriateness. Grounding is also
  enforced structurally: citations to never-retrieved chunks are dropped.
- **Calibration:** accuracy is reported within each stated-confidence bucket.

## Interpretation

- Residue-grounded questions (missense, nonsense, in-frame indels) across the
  stability, binding-site, and PTM-site mechanisms are answered correctly,
  fully grounded (grounding rate 1.0), with no ungrounded citations.
- Out-of-scope genes (not ingested) are correctly refused at low confidence —
  the no-hallucination guardrail works.
- **Known weak spot — splicing (success 0.0):** G2P protein-feature
  data has no splicing/intronic annotations, so the *correct* behavior is
  refusal. The offline `mock` backend retrieves protein chunks and answers
  anyway (overconfident), which the harness surfaces as hallucination=1.0 on
  these items. This is the main case where the live Claude backend is expected
  to outperform the mock, by recognizing that retrieved chunks don't address
  the splicing question and refusing. Re-run with `ANTHROPIC_API_KEY` set to
  measure that gap.

## Reproduce

```bash
g2p-agent ingest          # pull + index the 10 baseline genes
python eval/run_eval.py --name baseline
# live backends:
export ANTHROPIC_API_KEY=sk-...   # G2P_LLM/G2P_JUDGE auto-switch to Claude
python eval/run_eval.py --name baseline_live
```

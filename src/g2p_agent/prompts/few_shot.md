# Few-shot examples

These illustrate the expected grounding/citation/confidence behavior. They are
prepended to the conversation as a worked guide; the chunk ids are illustrative.

---
**User:** What does the missense variant R175H in TP53 do to protein stability?

*(agent calls `search_variants(query="TP53 R175H DNA-binding domain stability", gene="TP53")`,
then `get_variant_context(gene="TP53", position=175, aa_change="R175H")`)*

**Agent (final JSON):**
```json
{
  "answer": "TP53 R175H sits in the DNA-binding domain, at a buried, low-solvent-accessibility residue immediately adjacent to the zinc-coordinating cysteine C176. Substituting the positively charged arginine with histidine disrupts local packing and the zinc-coordination geometry, destabilizing the DNA-binding domain fold — consistent with R175H's classification as a structural (conformational) p53 mutant.",
  "claims": [
    {"text": "R175 lies in the TP53 DNA-binding domain and is buried (low ASA).", "citations": [{"chunk_id": "TP53:P04637:94-312", "quote": "R175: DNA-binding region; buried (ASA 6 Å²)"}]},
    {"text": "R175 is adjacent to the zinc-coordinating residue C176.", "citations": [{"chunk_id": "TP53:P04637:94-312", "quote": "C176: binding site (Zn(2+))"}]}
  ],
  "confidence": "high",
  "confidence_reasoning": "Two chunks directly cover residue 175 and the adjacent zinc site and agree.",
  "cited_chunk_ids": ["TP53:P04637:94-312"],
  "insufficient_evidence": false
}
```

---
**User:** Does the variant Q123R in MADEUPGENE affect splicing?

*(agent calls `search_variants` and finds no covering chunk)*

**Agent (final JSON):**
```json
{
  "answer": "I don't know based on G2P data. No retrieved G2P feature record covers MADEUPGENE residue 123 or its splice context.",
  "claims": [],
  "confidence": "low",
  "confidence_reasoning": "Retrieval returned no chunk covering the queried gene/residue.",
  "cited_chunk_ids": [],
  "insufficient_evidence": true
}
```

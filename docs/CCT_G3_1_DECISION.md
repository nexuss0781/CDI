# CCT-G3.1 Geometry-Observability Decision

> **Decision:** `EARNED_GEOMETRY_EVIDENCE`; **global quality status:** `REDESIGN_BEFORE_SCALE`; **scale authorization:** `False`.

CCT-G3.1 was a pre-registered, controlled mechanism test under the frozen CCT-G2.1 corpus, tokenizer, seed, token-budget, optimizer, context, precision, and held-out evaluation contract. Its purpose was narrow: determine whether the repaired Laplacian geometry path contributes to causal language prediction once the mean-only readout cancellation has been removed. It was **not** a scale experiment or a production language-quality claim.

## Submitted Artifact Identity

| Field | Recorded value |
|---|---|
| Result source | Submitted Colab `results/colab_cct_g3_1_geometry/REPORT.md` and `latest.json` |
| Code revision | `646c2726c3589a36fa6ab1292d43c0a4c99cdb8b` |
| Result fingerprint | `3a4c7346ddb3e5b97dfcd7f6f0ce25931f6ba6dda00fae537b327d4ee779dec0` |
| Dataset | `Nexuss0781/synaxarium`, 321 governed deduplicated documents |
| Manifest | `947d152f129f2fd91433fa9b64574c674f9aea8e472ef3cb4fd7dafa5a2bd0d9` |
| Tokenizer artifact | EthioBBPE fingerprint `d78996f0aca122d74054b927902aa9bf80c2b5cf00747a7cf4327ff0f7d1a88c` |
| Seeds | `[11, 29, 47]` |
| Training | 1,000 steps/model/seed; 30,000 causal positions/model/seed |
| Evaluation | All held-out batches, document/content-hash split isolation, deterministic per-epoch shuffle |
| Host-memory guard | 11.0 GiB maximum; 1.8596 GiB peak |

## Registered Geometry Gate

The full CDI and geometry-free CDI variants shared the mean-plus-fixed-zero-sum-contrast readout and exactly the same trainable parameter count. The geometry-free variant disabled only the sparse Laplacian correction. The matched GRU and Transformer controls used the same governed data and token budget.

| Model | Parameters | Mean validation loss | Mean test loss | Mean tokens/second |
|---|---:|---:|---:|---:|
| Full CDI | 80,510 | 6.847487 | 6.877744 | 209.9 |
| Geometry-free CDI | 80,510 | 6.861976 | 6.894169 | 256.8 |
| GRU | 80,120 | 6.800372 | 6.828304 | 1,838.1 |
| Transformer | 80,172 | 6.856126 | 6.880668 | 2,843.7 |

| Pre-registered gate | Result | Evidence |
|---|---|---|
| Complete finite record matrix | Pass | All 12 model/seed records are finite and complete. |
| Full CDI learns | Pass | Training loss decreased in all three CDI seeds. |
| Parameter matching | Pass | 0.49% maximum spread, below the 1.00% tolerance. |
| Full CDI beats geometry-free CDI in each seed | Pass | Full CDI validation loss is lower for seeds 11, 29, and 47. |
| Repeated geometry value | Pass | Mean validation-loss improvement is 0.014489 for full CDI over exact geometry-free CDI. |
| Memory safety | Pass | Peak recorded process/container memory was 1.8596 GiB, below 11.0 GiB. |

The mechanism decision is therefore **`EARNED_GEOMETRY_EVIDENCE`**. It is valid only for the current 48-state nano configuration, the Synaxarium manifest, the 16-token context, and the bounded CPU comparison protocol.

## Quality and Scale Boundary

The mechanism result does not erase the global CCT-G2.1 quality gate. Full CDI was 0.126% better than the Transformer mean validation loss, but it remained 0.693% above the GRU mean validation loss and was above GRU in every seed. The artifact's strict base decision is therefore `REDESIGN_BEFORE_SCALE`.

| Global quality condition | Result | Consequence |
|---|---|---|
| CDI within 5% of Transformer | Pass | The current model remains within the declared Transformer tolerance. |
| CDI matches or beats GRU in every seed | Fail | It lost to GRU in all three seeds. |
| Geometry has repeated value | Pass | Retain the geometry mechanism for the next controlled diagnostic. |
| CCT-G2.2, context, capacity, corpus, or performance work | Blocked | No scale, context, capacity, corpus, or speed step is authorized. |

> CCT-G3.1 shows that the repaired geometry is not inert. It does **not** show that the full CDI quality problem is solved, nor that CDI is faster than either baseline.

## Next Permitted Action

The only permitted next experiment is **CCT-G3.2 — controlled readout-contribution ablation**. It must pre-register and isolate the fixed vertex-contrast readout contribution from the geometry correction while retaining the same 321-document manifest, EthioBBPE artifact, 1,000-step budget, three seeds, context length, optimizer, precision, all-held-out evaluation, and parameter-aware controls. It must not increase data, steps, context, capacity, or the host-memory ceiling.

Its purpose is to distinguish these two questions before returning to any G2 quality rung:

1. Does the fixed contrast readout itself improve or harm language quality relative to the former mean-only route?
2. Given that readout, does the sparse geometry correction retain the repeated contribution already measured in G3.1?

Only after CCT-G3.2 produces a documented decision may CCT reconsider a corrected G2.1 quality rerun. CCT-G2.2 remains blocked.

## References

[1]: [CCT-G3.1 pre-registration](CCT_G3_1_PREREGISTRATION.md)  
[2]: [CCT-G2.1 decision](CCT_G2_1_DECISION.md)  
[3]: [Authoritative CCT checklist](../Todo.md)  
[4]: [CCT evidence index](CCT_EVIDENCE_INDEX.md)

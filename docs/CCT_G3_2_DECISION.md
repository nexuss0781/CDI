# CCT-G3.2 Readout-Contribution Decision

> **Mechanism decision:** `EARNED_READOUT_EVIDENCE`; **geometry status:** re-confirmed; **global quality status:** `REDESIGN_BEFORE_SCALE`; **scale authorization:** `False`.

CCT-G3.2 was a pre-registered five-model decomposition under the frozen CCT-G3 contract. It tested whether fixed zero-sum vertex contrast features add held-out language value beyond the historical mean-only feature values, while retaining the same sparse geometry computation and the same 48-to-4 readout layer. It also re-confirmed the CCT-G3.1 geometry comparison. It was not a scale, long-context, speed, or fluent-generation experiment.

## Submitted Artifact Identity

| Field | Recorded value |
|---|---|
| Result source | Submitted Colab `results/colab_cct_g3_2_readout/REPORT.md` and `latest.json` |
| Code revision | `fb50b572228ae64ea8de7c2481dd9894061ace02` |
| Result fingerprint | `b964ca95943e2800381bb339d47b2ea90fd4a55e19e479a5f7023861847ff91e` |
| Dataset and manifest | 321 governed Synaxarium documents; manifest `947d152f129f2fd91433fa9b64574c674f9aea8e472ef3cb4fd7dafa5a2bd0d9` |
| Tokenizer artifact | EthioBBPE fingerprint `d78996f0aca122d74054b927902aa9bf80c2b5cf00747a7cf4327ff0f7d1a88c` |
| Seeds and budget | `[11, 29, 47]`; 1,000 steps and 30,000 causal positions per model/seed |
| Evaluation | All held-out validation/test batches; deterministic per-epoch shuffle; document/content-hash isolation |
| Host-memory guard | 11.0 GiB maximum; 1.8634 GiB peak |

## Five-Model Result

| Model | Parameters | Mean validation loss | Mean test loss | Mean tokens/second |
|---|---:|---:|---:|---:|
| Full CDI | 80,510 | 6.847487 | 6.877744 | 200.2 |
| Mean-readout control | 80,510 | 6.904783 | 6.939423 | 212.4 |
| Geometry-free CDI | 80,510 | 6.861976 | 6.894169 | 248.1 |
| GRU | 80,120 | 6.800372 | 6.828304 | 1,786.5 |
| Transformer | 80,172 | 6.856126 | 6.880668 | 3,037.7 |

## Registered Gates

| Pre-registered gate | Result | Evidence |
|---|---|---|
| Complete finite five-model matrix | Pass | All 15 model/seed records are finite and complete. |
| Full CDI learning | Pass | Full CDI training loss decreased in all three seeds. |
| Parameter fairness | Pass | 0.49% maximum spread, below the 1.00% tolerance. |
| Full CDI lower validation loss than mean-readout control in each seed | Pass | Seed improvements: 0.077012, 0.045238, and 0.049637. |
| Repeated readout value | Pass | Mean full-CDI validation-loss improvement over the capacity-matched mean control: 0.057296. |
| Full CDI lower validation loss than geometry-free CDI in each seed | Pass | Seed improvements: 0.026886, 0.014050, and 0.002529. |
| Geometry re-confirmation | Pass | Mean geometry improvement: 0.014489. |
| Memory safety | Pass | Peak recorded process/container memory: 1.8634 GiB, below 11.0 GiB. |

The fixed contrast readout and sparse geometry correction therefore both have repeated bounded held-out value in the current nano configuration. The readout effect is larger than the incremental geometry effect under this protocol, but this comparison is descriptive; it is not an efficiency claim.

## Quality and Scale Boundary

The G3.2 full CDI record is also a complete corrected 1,000-step quality comparison for the selected architecture. It removes the need to duplicate the same G2.1 budget under an identical configuration. The result remains below the Transformer mean validation loss by 0.126%, but it is 0.693% above the GRU mean and above GRU in all three seeds.

| Global quality condition | Result | Consequence |
|---|---|---|
| CDI within 5% of Transformer | Pass | Full CDI remains within the declared Transformer tolerance. |
| CDI matches or beats GRU in every seed | Fail | Full CDI loses to GRU in seeds 11, 29, and 47. |
| Readout contribution repeated | Pass | Retain the fixed contrast readout for subsequent controlled diagnostics. |
| Geometry contribution repeated | Pass | Retain sparse geometry for subsequent controlled diagnostics. |
| CCT-G2.2, G4/G5, larger corpus, context/capacity changes, or performance work | Blocked | The quality-scale gate remains closed. |

> CCT-G3.2 validates two CDI-specific contributors, but it does not solve the model-quality relation to GRU. The appropriate action is another **single controlled architecture diagnostic**, not more data or a larger run.

## Next Permitted Action

The next permitted action is **CCT-G3.3 — controlled harmonic-memory-band contribution ablation**. It must pre-register an exact harmonic-disabled control that retains the same parameter inventory, tokenizer, corpus manifest, steps, seed list, context, optimizer, precision, all-held-out evaluation, and 11 GiB memory ceiling. It must determine whether the 16–64 time-constant harmonic band adds or subtracts held-out value in the current 16-token reset protocol.

The existing `disable_harmonic` cell hook must not be used as evidence until a dedicated parameter-aware control, inactive-gradient contract, local gates, five-model matrix, and decision rule are committed. No scale or context work is authorized while CCT-G3.3 is pending.

## References

[1]: [CCT-G3.2 pre-registration](CCT_G3_2_PREREGISTRATION.md)  
[2]: [CCT-G3.1 decision](CCT_G3_1_DECISION.md)  
[3]: [Authoritative CCT checklist](../Todo.md)  
[4]: [CCT evidence index](CCT_EVIDENCE_INDEX.md)

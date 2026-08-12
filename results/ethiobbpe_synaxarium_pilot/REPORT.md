# EthioBBPE Synaxarium Matched CDI Pilot

**Verdict:** `EARNED_NEXT_PILOT`. This is a bounded, real-data architecture pilot, not a production language-model claim.

The pilot used document-isolated Amharic readings from [`Nexuss0781/synaxarium`](https://huggingface.co/datasets/Nexuss0781/synaxarium), tokenized by the exact EthioBBPE artifact recorded in the result JSON. It trained each model for the same number of causal token positions under the same three seeds.

| Model | Parameters | Mean validation loss | Mean validation perplexity | Mean test loss | Mean tokens/sec |
|---|---:|---:|---:|---:|---:|
| dcss_cdi | 80,366 | 9.1219 | 9153.9 | 9.0704 | 299.2 |
| gru_baseline | 80,120 | 8.8682 | 7109.6 | 8.7856 | 2667.0 |
| transformer | 80,172 | 8.4611 | 4753.4 | 8.3341 | 4641.2 |

## Decision gates

| Gate | Result | Evidence |
|---|---|---|
| Learning | PASS | DCSS training loss decreased in every seed: `1.0`. |
| Matched baseline | PASS | DCSS relative validation-loss gap: `7.81%`; predeclared tolerance: `10%`. |
| Split isolation | PASS | The governed manifest's document and content-hash leakage checks passed before training. |

> This verdict applies only to the current 48-state CPU nano DCSS configuration and the bounded Synaxarium token budget.

## Reproducibility

| Field | Value |
|---|---|
| Dataset | `Nexuss0781/synaxarium` |
| Dataset license asserted by source card | `MIT` |
| Tokenizer | `EthioBBPE` artifact fingerprint `d78996f0aca122d74054b927902aa9bf80c2b5cf00747a7cf4327ff0f7d1a88c` |
| Seeds | `[11, 29, 47]` |
| Training steps per model/seed | `30` |
| Causal token positions per model/seed | `900` |
| Data manifest | `af695a05e610610f9aedd5ae3039f66db734b127d4ab6d6c1aa70110dc9c57c0` |
| Code revision | `2a27165c007aa6df8620215b8680eddf8fd7f990` |

## Interpretation

If the verdict is `REDESIGN_BEFORE_SCALING`, do not solve the result by adding a large corpus. Change the current DCSS state/readout design, then rerun this exact protocol. If the verdict is `EARNED_NEXT_PILOT`, extend the **same** protocol to a larger document and token budget before any production-scale pretraining.

## References

[1]: https://huggingface.co/datasets/Nexuss0781/synaxarium "Synaxarium dataset card"
[2]: https://huggingface.co/Nexuss0781/Ethio-BBPE "EthioBBPE model artifact"

# Stage D Customized Design — Reproducible Causal LM Integration

## Scope

Stage D connects the verified Stage C recurrence to a token-level causal language-model path. The implementation is intentionally **CPU-first, float32, and small-scale**. It validates reproducibility, causality, masking, checkpointing, and matched-system plumbing; it does not claim language-model quality or replace the later Stage E study.

## EthioBBPE tokenizer

Stage D uses the published `EthioBBPE` Byte-Level BPE tokenizer from `Nexuss0781/Ethio-BBPE`. CDI snapshots the exact `tokenizer.json` into a versioned local artifact when training starts, then restores that snapshot from the checkpoint at inference. No checkpoint depends on a mutable Hugging Face revision or an implicit local cache.

| ID | Token | CDI purpose |
|---:|---|---|
| 0 | `<pad>` | Padding; excluded from loss. |
| 1 | `<unk>` | Backend unknown-token identifier. |
| 2 | `<s>` | CDI document-start boundary. |
| 3 | `</s>` | CDI document-end boundary. |
| 4 | `<mask>` | Reserved CDI document-boundary identifier. |

The tokenizer artifact contains its model identifier, tokenizer JSON snapshot, special IDs, maximum chunk length, vocabulary fingerprint, and SHA-256 fingerprint. Checkpoint load rejects a mismatched tokenizer fingerprint by default. Every embedding and loss target is range-checked against that artifact; token IDs are never silently clamped or converted between tokenizers.

## Data protocol

The Stage D default is a compact, versioned local synthetic corpus stored in the repository. It is explicitly classified as **synthetic debugging data**, rather than a primary language-quality corpus. Every manifest records its local path, document hashes, aggregate content hash, deterministic split indices, document/token counts, boundary policy, and preprocessing revision.

Packing preserves document boundaries: every encoded document has `<bos>` and `<eos>`, and no packed training chunk crosses into another document. The implementation reports truncation counts; the configured corpus uses zero truncations. This supports deterministic causal training and baseline plumbing without representing the run as a WikiText, SciQ, or real-corpus result.

## Token-level model API

`DCSSLanguageModel` exposes the required interface:

```python
logits, new_state = model.forward_chunk(input_ids, state=None, attention_mask=None, return_state=True)
```

Token IDs are embedded, sent through the Stage C SSM, and projected through a tied vocabulary readout. For padded tokens, the model preserves the previous structured state and marks the output inactive. Its loss targets are `input_ids[:, 1:]` against logits at `[:, :-1]`, with padding ignored by a boolean loss mask.

A compact legacy-v2 adapter and a small causal Transformer are included solely for the matched Stage D harness. All three use the same tokenizer, batches, masked cross-entropy, optimizer family, train-token budget, evaluation examples, and documented seeds. The comparison is classified as **synthetic plumbing validation**, not an architecture-quality claim.

## Training and reproducibility protocol

The `nano` training configuration uses a four-dimensional embedding, an eight-token chunk, batch size four, float32, AdamW, clipping norm one, and zero gradient accumulation by default. A deterministic 100-step overfit gate uses a single fixed repeated-pattern document and must reduce masked causal cross-entropy by at least 90 percent with finite gradients.

Checkpoints include model and optimizer state, tokenizer artifact/fingerprint, topology fingerprint, data manifest, global step, cursor, Python/NumPy/Torch random states, configuration, and environment data. A 50-step resume test compares uninterrupted execution to a split run that checkpoints at step 25. The acceptance tolerance is `1e-6` for loss, parameter, and next-logit comparisons in float32 under deterministic CPU execution.

## Stage D acceptance gates

| Gate | Threshold |
|---|---:|
| Tokenizer round trip / artifact | Fixtures, special IDs, Unicode, long input, and unknown fallback succeed; fingerprint is stable. |
| Data integrity | Deterministic splits, zero cross-split document-hash overlap, complete local-source manifest. |
| Causal alignment | Maximum earlier-logit difference after future-token perturbation `≤ 1e-6`. |
| Mask correctness | Masked target change has loss and gradient difference `≤ 1e-7`; unmasked target changes loss. |
| Tiny overfit | At least 90% loss reduction in 100 steps; finite intended gradients. |
| Resume determinism | Loss, parameter, and logits max errors `≤ 1e-6`. |
| Precision safety | CPU float32 output/loss/gradient are finite. |
| Generation | Greedy and seeded sampling are reproducible valid IDs. |
| Baseline plumbing | v2 adapter, DCSS v3, and Transformer complete the same small synthetic protocol with complete manifests. |

## Transition discipline

A Stage D report must retain `stage_e_implementation_allowed: false`. Stage E cannot begin until the user explicitly approves the Stage D gate report.

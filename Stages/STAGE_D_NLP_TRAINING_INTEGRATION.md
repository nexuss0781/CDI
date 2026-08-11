# Stage D — NLP Training Integration and Reproducible Language Modeling

**Dependency:** Stage C must pass all recurrence, causality, stability, state-serialization, and step/chunk transition tests.

**Status:** Specification only. No implementation is authorized by this document.

## Stage objective

Stage D connects the verified DCSS-CDI recurrence to a production-quality causal language-model training path. It introduces explicit tokenizer configuration, deterministic data preparation, packed causal batches, vocabulary projection, optimizer/checkpoint support, mixed precision where available, validation reporting, and the first fair language-model comparisons.

The stage is successful when the new engine can train reproducibly on synthetic data and a documented corpus, produce valid causal logits, resume from a checkpoint, and report quality and systems metrics in a way that can be compared with legacy CDI v2 and a small Transformer baseline.

## Scope and non-goals

Stage D includes tokenizer/configuration cleanup, causal next-token training, deterministic dataset splits, streaming/packing, masking, optimizer setup, checkpoint/resume, gradient accumulation, mixed precision, validation, sample generation, experiment manifests, and matched small-scale baselines.

Stage D does not add episodic retrieval, tools, planning, self-verification, multi-agent behavior, or claims of general intelligence. It does not yet perform the full ablation study of Stage E; it prepares the shared training and evaluation harness that Stage E will use.

## Required model API

The v3 language model must expose:

```python
logits, new_state = model.forward_chunk(
    input_ids,
    state=None,
    attention_mask=None,
    return_state=True,
)
```

The equivalent embedding-level API may be retained for operator tests, but the token-level path must make vocabulary projection, padding, special tokens, and causal alignment explicit.

For input tokens `x[0:L-1]`, the model must produce logits whose target at position `t` is `x[t+1]`. It must never read target tokens or future input tokens when computing a position’s logits.

## Tokenizer and vocabulary contract

The tokenizer configuration must be a versioned artifact containing vocabulary files or a content hash, normalization rules, special-token IDs, padding behavior, maximum training chunk length, byte/unicode policy, and embedding dimension. Loading a checkpoint with a different tokenizer fingerprint must fail unless an explicit conversion is requested.

The harness must test empty text, Unicode text, repeated whitespace, unknown or byte fallback tokens, long text, padding, and end-of-sequence behavior. No silent truncation is allowed in training data preparation; truncation counts must be reported.

## Dataset pipeline

The dataset layer must support deterministic train/validation/test splits and a streaming mode. Every dataset artifact must record source URL or local source, version or revision, preprocessing code revision, split construction, document count, token count, and content hash where legally and technically possible.

The packing pipeline must concatenate only within the training split and must not cross document boundaries unless the configuration explicitly allows it. If document boundaries are preserved, the model must receive a boundary token or reset signal. The loss mask must exclude padding and any deliberately ignored positions.

The current WikiText-2 and SciQ paths may be retained for continuity, but the Stage D report must clearly separate:

| Evaluation class | Use |
|---|---|
| Synthetic | Debugging causality, overfitting, masking, and checkpointing. |
| Legacy continuity | WikiText-2/SciQ comparison with CDI v2. |
| Primary LM quality | A larger, versioned, held-out corpus appropriate for model comparison. |
| Handwritten diagnostics | Optional qualitative smoke tests only. |

## Training implementation requirements

### Optimizer and parameter groups

All trainable parameters must appear exactly once in optimizer parameter groups. The report must list parameter counts by group: token embeddings, output projection, gates, generators, memory bands, sparse geometry, cochain maps, and normalization/readout parameters.

The optimizer, learning rate, betas, weight decay, schedule, warm-up, gradient accumulation, clipping norm, and seed must be stored in the run manifest. If different parameter groups use different learning rates, that choice must be explicit.

### Loss and masking

The primary loss is masked causal cross-entropy. It must be computed with a numerically stable library or a verified equivalent. The harness must test that changing a masked target does not change the loss or gradients, while changing an unmasked target does.

Auxiliary losses for cochain residual, energy, or spectral diagnostics must be independently logged and ablatable. The engine must support a pure cross-entropy mode so that architectural comparisons do not silently include different regularization budgets.

### Gradient and precision management

The training loop must support gradient accumulation and clipping. Gradients must be checked for non-finite values before the optimizer update. If mixed precision is used, loss scaling, overflow detection, scale changes, and fallback behavior must be logged.

`float64` remains available for mathematical reference tests. Training should use `float32` and `bfloat16` where hardware supports them. The report must never compare `float64` v2 throughput against `bfloat16` v3 throughput without an explicitly labeled precision analysis.

### Checkpoint and resume

A checkpoint must include model, tokenizer, optimizer, scheduler, scaler, global step, epoch/stream cursor, random states, data manifest, topology fingerprint, hardware metadata, and code revision. Resume must continue from the same data position or explicitly declare a restart policy.

A resume test must compare uninterrupted training against interrupted-and-resumed training for a short deterministic run. The two runs must produce matching losses, parameters, and next logits within the declared tolerance.

## Baseline comparison protocol

Stage D must train three systems through the same harness:

1. legacy CDI v2;
2. DCSS-CDI v3; and
3. a small causal Transformer baseline with matched tokenizer, parameter budget range, optimizer family, data, training tokens, and evaluation code.

The comparison must report both quality and systems behavior. Parameter count may differ only within a predeclared tolerance; if exact matching is impossible, the report must include a parameter-normalized analysis.

## Evaluation harness

Expose commands equivalent to:

```text
python -m benchmarks.stage_d tokenizer --config configs/tokenizer.json
python -m benchmarks.stage_d data_audit --dataset wikitext2
python -m benchmarks.stage_d train_smoke --model dcss_cdi --steps 100
python -m benchmarks.stage_d train --model dcss_cdi --config configs/stage_d_small.json
python -m benchmarks.stage_d resume_test --steps 50 --interrupt-at 25
python -m benchmarks.stage_d compare --models v2,dcss_cdi,transformer
python -m benchmarks.stage_d report --input results/stage_d/<run_id>
```

### Mandatory test groups

| Test | Procedure | Required result |
|---|---|---|
| Tokenizer round trip | Encode/decode fixtures and verify special-token behavior | Versioned fingerprints and expected outputs. |
| Data audit | Verify splits, hashes, document boundaries, token counts, and no leakage | Complete audit with zero unexplained overlap. |
| Causal alignment | Perturb future tokens and verify earlier logits remain unchanged | Zero illegal future influence. |
| Masking | Perturb masked/unmasked targets | Masked targets have no effect. |
| Tiny overfit | Train on a tiny deterministic corpus | Predeclared loss reduction and token accuracy. |
| Gradient inventory | Check every intended parameter group | Finite, nonzero gradients where expected. |
| AMP/precision | Compare supported precision modes on a fixed batch | Finite outputs and documented error. |
| Checkpoint resume | Compare uninterrupted and interrupted runs | Loss/parameter/logit agreement. |
| Validation loop | Evaluate fixed validation examples without training mutation | Stable metrics and no optimizer updates. |
| Generation | Run deterministic greedy and seeded sampling | Reproducible outputs and valid token IDs. |
| Throughput/memory | Measure training and inference across lengths | Raw latency, memory, tokens/s, and state size. |
| Baseline comparison | Run v2, v3, and Transformer under matched protocol | Complete comparison table with manifests. |

## Pass/fail gates

| Gate | Pass condition | Failure consequence |
|---|---|---|
| Causal alignment | No future-token influence beyond declared numerical tolerance | Stop. No LM result is valid. |
| Mask correctness | Masked targets have zero effect; unmasked targets affect loss as expected | Stop and repair data/loss pipeline. |
| Tiny overfit | At least 90% loss reduction on the fixed synthetic corpus within the declared step budget, with finite gradients | Stop before corpus training. |
| Gradient coverage | Every intended trainable group receives finite gradients on a valid batch | Stop and diagnose disconnected paths. |
| Resume determinism | Interrupted/resumed run matches uninterrupted run within dtype tolerance | Stop. Results are not reproducible. |
| Data integrity | Zero unauthorized split overlap and complete dataset manifest | Stop. Discard contaminated runs. |
| Precision safety | No non-finite values in the standard precision configuration | Stop or downgrade precision with explicit report. |
| Evaluation completeness | All quality, throughput, memory, parameter, seed, and environment fields present | Fail the run; no partial claim. |
| v3 basic viability | v3 trains without worse-than-baseline numerical instability and reaches a nontrivial validation loss | If it fails, return to Stage C; do not proceed to ablations. |

Stage D does not require v3 to beat the Transformer yet. It requires a valid, reproducible, and fair comparison.

## Transition test to Stage E

Stage E may begin only after:

```text
1. The tokenizer and dataset fingerprints are frozen.
2. The synthetic causal/masking/checkpoint suite passes.
3. v2, v3, and Transformer baseline runs complete under one harness.
4. At least three seeds are available for the small comparison or the reason for fewer seeds is documented.
5. Raw logs, checkpoints, configs, and metrics are archived.
6. The Stage D report distinguishes quality, efficiency, and stability results.
```

The transition manifest must declare the exact model widths, parameter counts, training-token counts, data ordering policy, precision, optimizer, and stopping rule. Any later ablation that changes these variables must identify the change.

## Exit artifacts

Stage D exits with a versioned tokenizer artifact, audited dataset manifests, unified training runner, checkpoint/resume tests, synthetic overfit report, precision report, baseline comparison report, and frozen configurations for Stage E.

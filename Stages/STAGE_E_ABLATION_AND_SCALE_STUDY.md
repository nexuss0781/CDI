# Stage E — Matched Ablation, Long-Context, and Scale Study

**Dependency:** Stage D must pass the unified NLP training, data-integrity, causality, checkpoint, and baseline-comparison gates.

**Status:** Specification only. No implementation is authorized by this document.

## Stage objective

Stage E determines whether DCSS-CDI’s proposed innovations provide measurable benefit beyond implementation novelty. It uses controlled ablations and matched baselines to separate the effects of selective gating, sparse geometry, stable discretization, multi-timescale harmonic memory, parameter count, optimizer choice, and data exposure.

The stage has two outputs. The first is an engineering report on sequence scaling, memory, throughput, and numerical stability. The second is a language-model report on validation quality, long-context retention, and task behavior. The reports must not collapse all outcomes into a single “intelligence” score.

## Scope and non-goals

Stage E includes small and medium model comparisons, controlled ablations, sequence-length scaling, streaming tests, long-context synthetic tasks, standardized held-out evaluations, multi-seed statistics, and failure analysis.

Stage E does not add new capabilities such as retrieval, external tools, planning, or multimodal input. It evaluates the core sequence engine. Any memory component used in the main model must be part of the frozen Stage D configuration or be isolated as a named ablation.

## Experimental matrix

The minimum architecture matrix is:

| ID | Model | Purpose |
|---|---|---|
| `T` | Small causal Transformer | Matched conventional baseline. |
| `V2` | Legacy CDI v2 | Historical/reference baseline. |
| `U` | Ungated sparse CDI recurrence | Tests whether selective gating matters. |
| `G` | Selective recurrence without geometric field | Tests whether CDI geometry adds value. |
| `H` | Selective recurrence without harmonic/slow band | Tests persistent multi-timescale memory. |
| `E` | Selective recurrence with explicit Euler | Tests the stable discretization. |
| `C` | Selective recurrence with unconstrained cochain maps | Tests structural cochain construction. |
| `F` | Full DCSS-CDI | Proposed engine. |

The matrix may be extended with Hyena- or RWKV-family references where implementations and licenses permit. External architectures must use the same tokenizer, dataset, token budget, evaluation harness, and reporting fields as the internal models. Published results may be cited as context but cannot substitute for a local matched run.

## Matching rules

Every comparison must predeclare the following:

| Variable | Matching requirement |
|---|---|
| Tokenizer | Same vocabulary and preprocessing unless tokenizer ablation is the subject. |
| Training tokens | Same token count and data order policy. |
| Parameter budget | Within a declared relative tolerance or analyzed with parameter-normalized curves. |
| Optimizer | Same optimizer family and schedule unless optimizer is the ablation. |
| Precision | Same precision mode for speed/memory comparisons. |
| Hardware | Same hardware class and software stack for paired measurements. |
| Seeds | At least three seeds for headline small-scale results where compute permits. |
| Evaluation | Same frozen validation and test files. |
| Stopping | Same token budget or predeclared early-stopping rule. |

No model may receive a longer context, extra memory slots, more training tokens, or additional auxiliary losses without the report labeling that advantage.

## Systems evaluation

### Sequence-length scaling

Measure prefill/training forward, backward, optimizer-step, and streaming decode time over lengths such as `256, 512, 1,024, 2,048, 4,096, 8,192`, extending further when hardware allows. For each length record peak memory, persistent state bytes, temporary activation bytes, tokens per second, and total wall time.

Fit empirical log-log regressions for time and memory. Report the fitted exponent, confidence interval or bootstrap interval, measured range, and residuals. Do not infer asymptotic complexity from a single length pair.

### Streaming behavior

Process a long stream token by token while retaining only the model state. Measure state size before and after the stream, per-token latency after warm-up, and memory growth. Restart from serialized state at checkpoints and verify continuation equivalence.

### Kernel and allocation audit

Use runtime tracing or allocation hooks to prove that no dense sequence-by-sequence `L × L` tensor or dense full-state operator is created in the production DCSS-CDI path. The audit must include source-level checks for forbidden operations and runtime checks for actual allocations.

## Language and long-context evaluation

The core language-model evaluation must include a held-out corpus and report loss/perplexity or bits-per-byte. A second corpus or domain split should be used to test transfer. Report mean and standard deviation across seeds, not only the best run.

Long-context evaluation must separate:

| Capability | Example test | What it measures |
|---|---|---|
| Local syntax | Short-window continuation | Immediate predictive quality. |
| Associative recall | Delayed key-value retrieval | Retention of explicitly presented facts. |
| Distractor rejection | Retrieval with increasing distractors | Selective memory rather than indiscriminate accumulation. |
| Entity tracking | Repeated entities with intervening text | Stable identity/state maintenance. |
| Length extrapolation | Train at one length, evaluate at longer lengths | Generalization beyond training context. |
| Document continuation | Long held-out documents | Distributed discourse modeling. |
| Exact rare-fact recall | Synthetic or licensed fact lists | Whether compact memory loses precision. |

Use a fixed generation seed and decoding policy for qualitative samples. Samples are diagnostic only; quantitative metrics determine pass/fail.

## Statistical analysis

For each primary metric, report the seed-level mean, standard deviation, median, and bootstrap confidence interval where appropriate. A model is not declared better because one seed wins. The report must distinguish statistical uncertainty from hardware noise and from run-to-run instability.

For paired throughput and memory comparisons, use the same input batches and report paired ratios. For validation quality, compare area under the loss-versus-tokens curve as well as final loss, because a model that reaches the same quality with fewer tokens may be operationally superior.

## Evaluation harness

Expose commands equivalent to:

```text
python -m benchmarks.stage_e matrix --config configs/stage_e_matrix.json
python -m benchmarks.stage_e train_all --seeds 1,2,3 --token-budget 10000000
python -m benchmarks.stage_e scaling --lengths 256,512,1024,2048,4096,8192
python -m benchmarks.stage_e long_context --suite all --lengths 1024,4096,16384,65536
python -m benchmarks.stage_e trace_allocations --models dcss_cdi,transformer
python -m benchmarks.stage_e analyze --input results/stage_e/<study_id>
```

### Mandatory test groups

| Test group | Required evidence |
|---|---|
| Configuration audit | Hashes of every model/data/optimizer configuration. |
| Parameter audit | Counts by module and total, including tied parameters. |
| Training stability | Loss curves, finite status, gradient norms, optimizer overflows, restart events. |
| Scaling | Time/memory tables, exponent fits, raw traces, and plots. |
| Streaming | State-size and latency curves, restart equivalence. |
| Long context | Per-task curves by length and distractor count. |
| Quality | Validation/test loss with seed statistics. |
| Ablations | Difference tables against full DCSS-CDI with matched budgets. |
| Allocation guard | Source and runtime evidence for absence of prohibited dense paths. |
| Reproducibility | Rerun of at least one complete configuration from a fresh environment. |

## Pass/fail gates

Stage E has separate **engineering**, **quality**, and **scientific** gates. Passing one category does not imply passing the others.

### Engineering gates

| Gate | Target |
|---|---:|
| Forward memory exponent | ≤ 1.20 over the measured scaling range, or a documented explanation and redesign plan |
| Forward time exponent | ≤ 1.25 over the measured scaling range |
| Dense quadratic sequence tensor | None in runtime trace |
| Streaming state growth | Approximately constant with history length |
| Speed versus legacy CDI | ≥4× at the largest paired length that both can execute |
| Peak memory versus matched Transformer | Target ≤60% at 4k context |
| Numerical failures | Zero non-finite standard runs across headline seeds |

### Quality gates

The primary quality gate is comparative rather than an absolute perplexity promise. Full DCSS-CDI must be no worse than the strongest internal ablation by more than the predeclared uncertainty interval on the primary validation metric, and it must meet a predeclared minimum quality floor established by Stage D.

The long-context gate requires that DCSS-CDI retain a nontrivial advantage over the no-harmonic-memory ablation on at least one long-context task without suffering an unacceptable regression on local validation loss. Exact thresholds must be frozen before training and recorded in `stage_e_preregistration.json`.

### Scientific gates

| Gate | Pass condition |
|---|---|
| Attribution | Each claimed improvement is supported by its corresponding ablation. |
| Matching | No hidden parameter, token, precision, or context advantage. |
| Statistics | Headline results include multiple seeds and uncertainty. |
| Reproducibility | Independent rerun reproduces the direction and approximate magnitude of results. |
| Honesty | Failed hypotheses and negative results are included in the report. |

## Go/no-go outcomes

| Outcome | Criteria | Action |
|---|---|---|
| Go | Engineering gates pass and DCSS-CDI is competitive on quality with evidence for at least one claimed advantage | Proceed to Stage F capability modules. |
| Conditional go | Engineering gains pass but language quality is materially behind; or quality is competitive but efficiency gains are weak | Freeze findings, redesign the identified component, and repeat only affected studies. |
| No-go | Dense path remains, recurrence is unstable, results are not reproducible, or gains disappear under matching | Stop scaling. Return to Stage C or D. |
| Inconclusive | Hardware, data, or variance prevents a reliable conclusion | Increase measurement quality before changing architecture. |

A result is not a go merely because DCSS-CDI is faster. It must also be trainable, causal, reproducible, and at least scientifically competitive on the declared quality tasks.

## Transition test to Stage F

Stage F may begin only when the Stage E report has been preregistered before the final runs, all raw results are archived, the matched matrix is complete, scaling and allocation audits pass, and the go/no-go decision is explicitly signed in the study manifest.

The transition review must answer:

```text
1. Does DCSS-CDI avoid the dense quadratic paths in practice?
2. Is it stable and reproducible across seeds?
3. Does it retain a quality advantage, parity, or an explicitly accepted trade-off?
4. Which components caused each observed gain or failure?
5. What exact capability is justified for Stage F?
```

If any answer is unknown, Stage F must be limited to diagnostic work rather than capability expansion.

## Exit artifacts

Stage E exits with a preregistration, complete model matrix, matched training manifests, raw logs, parameter audits, scaling and allocation traces, long-context results, multi-seed statistics, ablation analysis, negative-result record, and a signed go/no-go decision.

## References

[1]: https://arxiv.org/abs/2312.00752 "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"

[2]: https://proceedings.mlr.press/v202/poli23a.html "Hyena Hierarchy: Towards Larger Convolutional Language Models"

[3]: https://aclanthology.org/2023.findings-emnlp.936/ "RWKV: Reinventing RNNs for the Transformer Era"

[4]: https://proceedings.iclr.cc/paper_files/paper/2025/hash/84a7fc24ed52e8eff514c33e8ac76ea3-Abstract-Conference.html "Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling"

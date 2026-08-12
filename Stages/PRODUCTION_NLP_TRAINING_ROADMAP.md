# DCSS-CDI Production NLP Training and Fine-Tuning Roadmap

**Status:** Proposed roadmap for approval.
**Scope:** Progress from the current DCSS-CDI research prototype to a production-ready NLP system through evidence-gated training, fine-tuning, evaluation, and controlled release.
**Non-claim:** This document does not authorize autonomous tools, external side effects, a production release, or claims that DCSS-CDI outperforms Transformer architectures. Each requires measured evidence and separate approval.

## 1. Starting position and governing principle

Stage F passed its bounded diagnostic gates, but its own report is explicit that the system has only been exercised as a local, synthetic, dry-run diagnostic. It does **not** yet establish natural-language quality on real corpora, scalable training behavior, retrieval quality on realistic data, or safety for external actions.[1] Consequently, the immediate objective is not to scale parameters blindly. The objective is to build a reproducible empirical case that the DCSS-CDI core is trainable, competitive for a defined NLP task, and operationally governable.

> **Production gate principle:** no model, data source, adaptation, deployment route, or capability becomes more privileged merely because a prototype passes. It advances only after reproducible measurements show it meets its predeclared acceptance criteria.

The roadmap uses four inseparable workstreams: **architecture and training**, **data and fine-tuning**, **evaluation and safety**, and **operations and release governance**. The NIST Generative AI profile is a suitable organising reference because it is explicitly designed to help organisations incorporate trustworthiness considerations throughout AI-system design, development, use, and evaluation.[2]

| Current evidence | What it supports | What it does not support |
|---|---|---|
| Stable 48-state nano DCSS recurrence and state codec | Small-scale CPU engineering experiments | Scale-out training feasibility or quality claims |
| Synthetic causal-LM and ablation harnesses | Controlled research comparisons | Real-language or real-user usefulness |
| Local memory, retrieval, tools, planner, verifier, audit trail | Isolated dry-run capability diagnostics | Agents, autonomous execution, or tool permissions |
| Stage F composition gate | Capability modules are optional to the tested core path | Production integration correctness under load |

## 2. Target product definition: the decision required first

Before any real-corpus training, approve a concise **Product Target Brief**. A single model cannot be evaluated rigorously without a clear intended use. The brief must define the primary language(s), use case, prohibited uses, input/output modality, latency and cost envelope, privacy classification, user population, and acceptable risk posture.

The initial target should be deliberately narrow. A defensible first product is, for example, a domain-bounded text classifier, structured extraction model, summarisation assistant with human review, or retrieval-grounded drafting assistant with no autonomous tools. Do not begin with a general-purpose agent or a broad claim of “beyond-Transformer” language intelligence.

| Brief field | Required decision | Why it gates training |
|---|---|---|
| Primary task | One measurable task family | Determines labels, loss, baselines, and evaluation |
| Languages and scripts | Initial coverage and exclusions | Determines tokenizer and corpus requirements |
| Deployment mode | Offline batch, API, or reviewed assistant | Determines latency, observability, and safety controls |
| Risk class | Low, medium, or high impact | Determines human review and release thresholds |
| Data policy | Permitted licenses, PII policy, retention | Determines whether training data may be used at all |
| Success criterion | Quality, latency, cost, and reliability metrics | Prevents post-hoc metric selection |

## 3. Workstream A — architecture, scaling, and pretraining

### A0. Preserve the research baseline

Freeze the published Stage F commit as a reproducible reference. Every new training run must record the code commit, DCSS configuration, tokenizer version, corpus manifest hash, random seed, hardware/software environment, optimizer state, and checkpoint lineage. Use an experiment tracker or equivalent immutable run manifest; MLflow, for example, records parameters, code versions, metrics, and output artifacts for each run.[6]

The existing `nano` configuration remains the correctness oracle. It must never be replaced by a larger configuration without retaining the small reference tests for numerical equivalence, state serialization, cohomological-health bounds, and geometry-ablation behavior.

### A1. Make scalable configurations explicit

Implement a **scale ladder**, not a single large leap. Each rung must retain a matched dense-oracle or reduced reference test when mathematically possible.

| Rung | Purpose | Required comparison | Advance only if |
|---|---|---|---|
| Nano | Regression and numerical oracle | Existing DCSS vs frozen reference | All legacy and Stage F gates pass |
| Small | First real-text learning proof | DCSS vs matched tiny Transformer and recurrent baselines | Stable loss, no NaN/Inf, reproducible checkpoints |
| Medium | Architecture and efficiency validation | Equal parameter, token, FLOP, and wall-clock comparisons | No material quality regression and credible efficiency result |
| Target | Product candidate | Best validated baseline at realistic sequence lengths | Product-specific quality, latency, reliability, and safety gates pass |

Configuration scaling must be driven by measured memory, throughput, and loss curves. The new architecture should be evaluated at matched **parameter count**, **tokens seen**, **optimizer**, **data mixture**, **context length**, and **compute budget**. Report both the raw outcome and the geometry-ablation delta. A DCSS advantage is credible only if it remains after those controls.

### A2. Build a resumable production-grade training loop

The training system should support deterministic configuration manifests, fault-tolerant checkpoints, gradient clipping, mixed precision where validated, learning-rate warmup/decay, activation/memory profiling, evaluation checkpoints, and automatic stop-on-divergence. Checkpoints must include model state, optimizer/scheduler state, tokenizer version, RNG states, data cursor, source commit, corpus manifest, and training manifest.

For multi-GPU experiments, introduce distribution only after the single-device run is numerically stable. PyTorch’s Fully Sharded Data Parallel wrapper shards module parameters across data-parallel workers, but it changes parameter handling and has documented constraints; it therefore belongs behind an equivalence and recovery test suite, not as an assumed first optimisation.[5]

### A3. Define pretraining admission criteria

Do not spend a large training budget until the following items have passed on a small real-data pilot:

| Criterion | Evidence required |
|---|---|
| Optimisation stability | Multiple seeds, monotonic training-health checks, no unrecoverable divergence |
| Resume fidelity | Interrupted-and-resumed run agrees with uninterrupted control within a declared tolerance |
| Throughput correctness | Measured tokens/s, peak memory, and cost attribution with the exact run manifest |
| Baseline comparison | Matched DCSS, Transformer, and simple recurrent/conv baseline results |
| Long-context behavior | Retention and retrieval probes at increasing sequence lengths, including failure cases |
| Geometry contribution | `geometry_ablation` outcome reported, not hidden |

## 4. Workstream B — data, tokenizer, and fine-tuning

### B0. Establish a governed data intake process

Only use data with documented provenance, permission or license, source restrictions, and retention policy. Every corpus release must have a versioned manifest and a dataset card covering motivation, composition, collection or acquisition route, preprocessing, known limitations, permitted use, and removal process. Datasheets for Datasets specifically propose documenting a dataset’s motivation, composition, collection process, and recommended uses to improve transparency and accountability.[3]

Build the ingestion pipeline around immutable source snapshots, per-document content hashes, provenance fields, filtering decisions, deduplication records, language/script labels, and PII/security policy outcomes. Maintain a deletion ledger so a source removal can be traced into future corpus and checkpoint decisions.

| Pipeline stage | Minimum control | Blocking condition |
|---|---|---|
| Intake | License/provenance record and source manifest | Unknown source, prohibited terms, or missing rights |
| Normalisation | Deterministic transform with versioned code | Irreversible transform without original lineage |
| Quality filtering | Documented rules and sampled audit | High duplicate, corrupt, unsafe, or off-domain rate |
| Privacy review | PII policy and documented detection/remediation | Unresolved sensitive-data exposure |
| Split construction | Leakage and duplicate checks across splits | Test/validation contamination |
| Release | Corpus card and manifest hash | No reproducible corpus identity |

### B1. Tokenization strategy

Retain the pure-Python Unicode character tokenizer as the **compatibility baseline**. Production quality should not assume it is optimal. Compare it to carefully governed subword or byte-level alternatives selected for the approved languages and scripts. The comparison must measure fertility, out-of-vocabulary behavior, sequence length inflation, tokenization latency, reconstruction errors, and downstream quality at equal byte/token budgets.

The tokenizer is a versioned model dependency. A tokenizer change creates a new training lineage; it is not a harmless preprocessing tweak. For multilingual or under-resourced scripts, require coverage diagnostics and human linguistic review before training a product candidate.

### B2. Pretraining corpus pilot

Begin with a legally cleared, bounded pilot corpus rather than web-scale ingestion. Train small and medium rungs across at least three seeds. Keep one held-out in-domain set, one out-of-domain robustness set, and one contamination-control set. Build a corpus mixture policy based on explicit weights and source caps, not a hidden global shuffle.

A pilot advances only when it proves data integrity, optimiser stability, and fair matched-baseline evaluation. It is acceptable for the result to be negative; a negative result is evidence that should redirect the research path before additional compute is spent.

### B3. Fine-tuning ladder

Fine-tuning starts only after a frozen pretrained checkpoint, task specification, rights-cleared supervised data, and a task-specific evaluation pack exist. Use the least invasive adaptation that meets the target: prompt/template baseline, linear probe, adapter/low-rank adaptation, selective-layer update, and then full fine-tune only if needed. Every adaptation must preserve an immutable base-checkpoint reference and record its data, hyperparameters, intended use, and rollback route.

| Fine-tuning stage | Objective | Required gate |
|---|---|---|
| Baseline | Establish non-adapted performance | Reproducible zero/few-shot or simple head result |
| Supervised adaptation | Improve defined task quality | Held-out lift without leakage or harmful regression |
| Preference/safety tuning | Improve reviewed behavior | Human-reviewed and adversarial test improvement, no capability escalation |
| Domain refinement | Improve an approved specialty | Domain expert review and subgroup/edge-case evidence |
| Release candidate | Freeze deployment artifact | Model card, reproducibility packet, approval record |

## 5. Workstream C — evaluation, safety, and red-team evidence

### C0. Build the evaluation pack before training the release candidate

The evaluation suite should combine task-quality metrics, calibration or abstention metrics where relevant, long-context probes, robustness tests, latency/throughput measurements, and explicit safety tests. Evaluation must include private holdouts unavailable to training authors and a contamination protocol for public benchmarks.

Standardised tools can help execute repeatable measurements, but they do not define product fitness on their own. Hugging Face’s evaluation documentation similarly distinguishes community leaderboard results, author-provided model-card reporting, and reusable evaluation tooling.[7]

### C1. Use a scorecard, not one headline metric

| Dimension | Example evidence | Release criterion |
|---|---|---|
| Core task quality | Exact match, F1, ROUGE, human rubric, or domain score | Meets preregistered target and matched baseline threshold |
| Reliability | Calibration, abstention accuracy, retry behavior | Fails safely when evidence is insufficient |
| Robustness | Noisy input, distribution shift, multilingual/script probes | No critical degradation outside declared limits |
| Long context | Needle, retention, distractor, and conflict probes | Demonstrated behavior at target context length |
| Efficiency | Tokens/s, p50/p95 latency, peak memory, energy/cost proxy | Meets product envelope at target load |
| Privacy and security | Memorisation probes, PII tests, prompt injection, data exfiltration tests | No unresolved critical finding |
| Governance | Dataset card, model card, lineage, approvals | Complete and independently reviewable |

### C2. Keep capability boundaries explicit

Stage F tools, retrieval, planning, and verification remain **Experimental** until an independent security review approves a narrow, monitored release configuration. Retrieved text and tool output continue to be untrusted data. No shell, network, account, payment, posting, deletion, transfer, or file mutation capability should be added during model-quality training.

If a later product requires retrieval augmentation, evaluate grounding, citation integrity, conflict handling, data access controls, tenant isolation, and prompt-injection resilience independently of language-model quality. If it requires tools, create a separate capability-safety programme with least privilege, allowlisted schemas, dry runs, rate limits, approval flows, tamper-evident audit logs, and emergency disablement.

## 6. Workstream D — operations, governance, and deployment readiness

### D0. Choose infrastructure that fits the workload

The present CPU-only sandbox is suitable for unit tests, deterministic nano experiments, and harness development. It is not an appropriate execution environment for substantial pretraining or production-scale fine-tuning. Because the persistent managed environment described for this project class has no GPU, a production training programme needs either approved user-controlled GPU hardware or a governed third-party GPU training environment. The selected environment must meet data residency, access control, secret management, artifact retention, and incident-response requirements.

Use a layered approach: local CPU for correctness; single GPU for small/medium training validation; distributed GPU execution only after a reproducible single-device baseline exists. Separate training, evaluation, artifact storage, and inference credentials. Store secrets in the platform’s secret manager rather than configuration files or source control.

### D1. Reproducibility and artifact governance

Every accepted model must be reconstructible from a model lineage record containing the code commit, environment lock, DCSS config, tokenizer version, corpus manifest, preprocessing version, run configuration, seed, checkpoint hashes, evaluator version, test-set manifest, and approval decision. Experiment-tracking systems are useful because they tie a run’s parameters, metrics, code versions, and artifacts together, but they must be protected by access control and retention policy.[6]

Publish an internal model card for every candidate. The model-card framework recommends documenting intended use, evaluation procedure, and performance characteristics so users can judge whether a model is suitable for a context.[4] For DCSS-CDI, add architecture-specific sections for cohomological-health diagnostics, state size, recurrence configuration, geometry ablation, long-context behavior, and known failure modes.

### D2. Release control and rollback

Begin with an internal, monitored pilot. Use staged traffic, fixed request budgets, red-team monitoring, privacy-aware logging, performance alarms, and a tested rollback to the preceding approved checkpoint. A release is a reversible operational decision, not just a checkpoint upload.

| Release tier | Permitted scope | Mandatory controls |
|---|---|---|
| Research | Offline evaluation only | Isolated data, no external actions, reproducible run records |
| Internal pilot | Approved staff, narrow task | Access control, monitoring, incident channel, rollback |
| Limited external beta | Small approved cohort | User disclosure, rate limits, human escalation, change freeze |
| General availability | Only after beta evidence | Service objectives, audits, documentation, support and deprecation plans |

## 7. Decision gates and stop conditions

The roadmap is intentionally evidence-gated. Any failure below triggers diagnosis, rollback, or narrowing of scope; it does not trigger a weaker narrative around the result.

| Gate | Required decision | Stop condition |
|---|---|---|
| G0 — Product charter | Approve narrow intended use and prohibited uses | No task, risk, or data-policy owner |
| G1 — Data admission | Approve pilot corpus and dataset card | Unclear rights, provenance, PII handling, or leakage |
| G2 — Training readiness | Authorise small real-data runs | Non-reproducible training, unstable recurrence, missing baselines |
| G3 — Scale evidence | Authorise medium/target scale | No matched quality/efficiency case or unresolved failure modes |
| G4 — Fine-tune readiness | Approve task adaptation | No held-out task data, clear metric, or base-checkpoint lineage |
| G5 — Release-candidate evaluation | Approve internal pilot | Failing reliability, privacy, security, or governance test |
| G6 — Deployment | Approve staged external access | No rollback, monitoring, incident plan, or accountable owner |

## 8. Recommended execution order

The programme should proceed in the following order, with a written gate review after each unit rather than a calendar-driven escalation.

1. **P0 — Product and data charter.** Approve the Product Target Brief, data policy, evaluation card, and risk owner.
2. **P1 — Training-system hardening.** Add run manifests, deterministic resume tests, data lineage, checkpoint verifier, performance profiler, and model-card/data-card templates.
3. **P2 — Real-data pilot.** Ingest a small rights-cleared corpus, run tokenizer and DCSS scale-ladder comparisons, and document negative as well as positive results.
4. **P3 — Independent evaluation.** Create private holdouts, baseline parity tests, long-context diagnostics, memorisation/privacy probes, and adversarial input tests.
5. **P4 — Narrow task fine-tuning.** Build a supervised, rights-cleared task dataset; compare minimal adaptation methods; freeze the best reproducible candidate.
6. **P5 — Internal pilot.** Serve only the approved task with monitoring, access control, logging policy, manual escalation, and rollback.
7. **P6 — Controlled external beta.** Expand only if pilot evidence clears the quality, safety, reliability, and operating gates.

## 9. First implementation package after approval

After you select the first product target, I recommend implementing the following repository package before initiating large training:

| Deliverable | Purpose |
|---|---|
| `cdi/v3/production/config.py` | Typed, immutable train/eval/release configuration schema |
| `cdi/v3/production/lineage.py` | Dataset, tokenizer, checkpoint, and code-hash lineage records |
| `cdi/v3/production/checkpoints.py` | Atomic save/restore, integrity hashes, deterministic-resume test |
| `cdi/v3/production/data.py` | Ingestion manifest, filtering hooks, deduplication, split audit |
| `cdi/v3/production/evaluation.py` | Task metrics, baseline parity, long-context, calibration, safety suites |
| `benchmarks/production_scale.py` | Equal-budget scaling and geometry-ablation harness |
| `tests/production/` | Reproducibility, data-leakage, resume, and regression gates |
| `docs/` cards | Dataset card, evaluation card, model card, and release checklist templates |

This package should remain offline and non-production initially. It must not add tools, network actions, or agentic capabilities. Its first acceptance result is a reproducible small real-corpus experiment and a matched-baseline report—not a deployed model.

## 10. Approval requested

To begin P0, please choose the answers below. Once approved, the next implementation stage can be planned precisely and safely.

1. **First task:** Which narrow use case should DCSS-CDI target first—classification, extraction, summarisation, retrieval-grounded drafting, or another specified task?
2. **Data boundary:** Which languages, sources, licenses, and sensitivity rules are allowed? Will you provide data, or should the project begin with a named public corpus whose terms you approve?
3. **Success metrics:** Which quality, latency, cost, and safety metrics matter for the target use case?
4. **Compute environment:** What GPU environment and data-residency constraints are available or acceptable?
5. **Release boundary:** Is the first milestone offline research, an internal reviewed pilot, or a controlled API beta?

## References

[1]: https://github.com/nexuss0781/CDI/blob/master/Stages/STAGE_F_GATE_REPORT.md "DCSS-CDI Stage F bounded diagnostic gate report"

[2]: https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence "NIST AI RMF: Generative Artificial Intelligence Profile"

[3]: https://arxiv.org/abs/1803.09010 "Datasheets for Datasets"

[4]: https://arxiv.org/abs/1810.03993 "Model Cards for Model Reporting"

[5]: https://docs.pytorch.org/docs/stable/fsdp.html "PyTorch FullyShardedDataParallel documentation"

[6]: https://mlflow.org/docs/latest/ml/tracking/ "MLflow Tracking documentation"

[7]: https://huggingface.co/docs/evaluate/en/index "Hugging Face Evaluate documentation"

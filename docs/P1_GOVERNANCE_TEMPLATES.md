# P1 Governance Templates for DCSS-CDI

**Status:** Offline preparation template. Completing these templates does not authorise real-corpus training, fine-tuning, deployment, or external actions.

## Dataset Card Template

| Field | Required record |
|---|---|
| Dataset name and manifest fingerprint | Immutable name, version, and content-addressed manifest |
| Intended training use | The narrowly approved task and explicit exclusions |
| Source and rights | Source URI, license/permission evidence, and restrictions |
| Composition | Language/script coverage, source mixture, document/token counts |
| Collection and preprocessing | Versioned acquisition, filtering, normalisation, and deduplication rules |
| Privacy and sensitivity | PII policy, review outcome, removal request process, retention |
| Splits | Train/validation/test identities, leakage audit, and contamination controls |
| Known limitations | Biases, gaps, domain shift risks, and prohibited uses |

## Evaluation Card Template

| Field | Required record |
|---|---|
| Intended use and risk class | Product Target Brief reference and disallowed use cases |
| Evaluation data | Manifest fingerprint, split provenance, and held-out controls |
| Metrics | Primary task, reliability, robustness, efficiency, and safety metrics |
| Baselines | Exact matched budget/configuration and evaluator version |
| Stress tests | Long-context, distractor, conflict, noisy-input, and safety probes |
| Acceptance thresholds | Preregistered targets, tolerances, and stop conditions |
| Limitations | Coverage gaps and measurements not represented by the scorecard |

## Model Card Template

| Field | Required record |
|---|---|
| Artifact identity | Model checkpoint hash, lineage fingerprint, code revision |
| Architecture | DCSS configuration, state size, bands, geometry-ablation setting |
| Tokenizer | Tokenizer artifact and fingerprint |
| Training | Corpus manifests, data policy, optimizer, compute environment, seed |
| Evaluation | Evaluation-card fingerprint, metrics, confidence/seed variation |
| Efficiency | Parameter count, tokens/s, peak memory, latency at stated hardware/load |
| Safety and capability boundary | Tool policy, retrieval policy, privacy observations, known failure modes |
| Release decision | Approved scope, owner, monitoring, rollback artifact, expiry/review date |

## Release Checklist Template

A release candidate cannot advance until all fields are completed and independently reviewed.

| Gate | Evidence | Decision |
|---|---|---|
| Data admission | Dataset card, rights/PII review, split audit | Pending |
| Reproducibility | Code/config/tokenizer/corpus/checkpoint lineage | Pending |
| Quality | Preregistered held-out and baseline comparison | Pending |
| Reliability | Calibration/abstention and regression suite | Pending |
| Safety | Red-team, privacy, prompt-injection, and access-control review | Pending |
| Operations | Monitoring, incident owner, service objectives, rollback drill | Pending |
| Scope | Permitted users/tasks and prohibited uses | Pending |

## References

[1]: https://arxiv.org/abs/1803.09010 "Datasheets for Datasets"

[2]: https://arxiv.org/abs/1810.03993 "Model Cards for Model Reporting"

[3]: https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence "NIST AI RMF: Generative Artificial Intelligence Profile"

# CCT Evidence Index

> **Scope:** This index identifies the authoritative reproducibility and empirical artifacts for completed CCT goals. It does not authorize a later goal; every listed transition remains governed by `Todo.md`.

## Completed Evidence

| CCT goal | Decision | Authoritative artifacts | Recorded code revision | Key reproducibility identifiers |
|---|---|---|---|---|
| CCT-G0 — Reproducible Execution Readiness | `READY_FOR_NEXT_GOAL` | `docs/CCT_G0_READINESS.md`; `scripts/run_cct_g0.sh`; generated evidence at `results/cct_g0/a038147/` | `a03814705a73a3cd36658e3d0780a982593070f9` | `master`; EthioBBPE 2.0.0; 246 passing tests; clean post-test tree |
| CCT-G1 — Bounded Learning Proof | `EARNED_NEXT_PILOT` | `Stages/ETHIOBBPE_SYNAXARIUM_PILOT_VERDICT.md`; `results/ethiobbpe_synaxarium_pilot_300steps/latest.json`; `results/ethiobbpe_synaxarium_pilot_300steps/REPORT.md`; `results/ethiobbpe_synaxarium_pilot_300steps/ANALYSIS.json`; `results/ethiobbpe_synaxarium_pilot_300steps/validation_loss.png` | `2a27165c007aa6df8620215b8680eddf8fd7f990` | Manifest `af695a05e610610f9aedd5ae3039f66db734b127d4ab6d6c1aa70110dc9c57c0`; EthioBBPE artifact `d78996f0aca122d74054b927902aa9bf80c2b5cf00747a7cf4327ff0f7d1a88c`; seeds `[11, 29, 47]` |
| CCT-G2.1 — Full-Corpus 1,000-Step Diagnostic | `REDESIGN_BEFORE_SCALE` | `docs/CCT_G2_1_DECISION.md`; submitted Colab `results/colab_stage2a_full_corpus/REPORT.md` and `latest.json` | `d5a2180e6e61494140b8ff221703cef7c317ecd3` | Manifest `2b868a661d628ec0e4507f65ee99e79abfbed12910241f95e7660a99e97e39c8`; EthioBBPE artifact `d78996f0aca122d74054b927902aa9bf80c2b5cf00747a7cf4327ff0f7d1a88c`; seeds `[11, 29, 47]`; all held-out evaluation |
| CCT-G3.1 — Geometry-Observability Ablation | `EARNED_GEOMETRY_EVIDENCE`; global quality remains `REDESIGN_BEFORE_SCALE` | `docs/CCT_G3_1_PREREGISTRATION.md`; `docs/CCT_G3_1_DECISION.md`; submitted Colab `results/colab_cct_g3_1_geometry/REPORT.md` and `latest.json` | `646c2726c3589a36fa6ab1292d43c0a4c99cdb8b` | Result `3a4c7346ddb3e5b97dfcd7f6f0ce25931f6ba6dda00fae537b327d4ee779dec0`; manifest `947d152f129f2fd91433fa9b64574c674f9aea8e472ef3cb4fd7dafa5a2bd0d9`; EthioBBPE artifact `d78996f0aca122d74054b927902aa9bf80c2b5cf00747a7cf4327ff0f7d1a88c`; seeds `[11, 29, 47]`; 11 GiB guard/1.8596 GiB peak |

The CCT-G1 revision is an ancestor of current `master`. Its result supports only the bounded 60-document, 300-step, 16-token-context comparison recorded in those artifacts. CCT-G2.1 improves the evidence by using 321 deduplicated documents, 1,000 steps, deterministic batch shuffle, and complete held-out evaluation, but it does not permit scale expansion because CDI lost to GRU in all three seeds. CCT-G3.1 subsequently established repeated geometry value against an exact geometry-free CDI counterpart, but it retained the same base quality limitation against GRU and therefore also does not authorize scaling.

## Active Gate

| CCT goal | Required input before a decision | Do not proceed to |
|---|---|---|
| CCT-G3.2 — Controlled Readout-Contribution Ablation | A pre-registered parameter-aware contrast-readout control, local causal/gradient gates, exact configuration, `REPORT.md`, and `latest.json` under the frozen CCT-G3 comparison contract | CCT-G2.1 quality rerun, CCT-G2.2, context work, capacity changes, corpus expansion, or optimization work |

## Evidence Handling Rule

Every completed CCT sprint must be added here only after its required artifacts exist and its transition decision is recorded in `Todo.md`. A failed gate must be indexed as a failure with its prescribed next action; it must not be replaced by a larger uncontrolled run.

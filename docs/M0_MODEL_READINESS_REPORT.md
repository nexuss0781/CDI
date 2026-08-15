# CDI Module M0 — Model and Data Readiness

**Status:** `PASS`  
**Model:** `dcss_cdi`  
**Validation mode:** CPU, float32, deterministic local run  
**Report fingerprint:** `61721ff20ac2c1beb473c0453a2d7022bed5ce25cfc10aa3d6dee7d536217e1f`

## Module objective

M0 verifies that CDI can load its model and tokenizer, perform inference, execute a bounded training update, save a checkpoint, reload it with integrity verification, and remain within the memory budget. M0 does not claim language learning quality; it establishes that the model-training pipeline is valid for the next module.

## Gate results

| Gate | Result | Evidence |
|---|---|---|
| Model load and tokenizer binding | **PASS** | 80,510 parameters loaded; vocabulary size 16,000; tokenizer fingerprint reproduced exactly. |
| Forward inference and causality | **PASS** | Logits shape `[4, 8, 16000]`; all outputs and recurrent state tensors finite; causal max error `0.0`. |
| One-step training and gradient health | **PASS** | Loss `9.680342674255371`; 44 parameters changed within 45 declared trainable parameters; no frozen parameter changed. |
| Checkpoint reload and integrity | **PASS** | Reload max error `0.0`; global step `1`; cursor `1`; tampered checkpoint rejected. |
| Memory budget | **PASS** | Peak RSS `0.381851 GiB`; hard limit `11.0 GiB`; operating target `8.5 GiB`. |

## Artifact

The accepted local artifact was written to `results/m0/m0_candidate.pt` with SHA-256:

```text
8ce11c3dd7a0d22a6a43b24c43ce55ba01da4489757b37ae9fd36aa3d6dee7d536217e1f
```

The tokenizer fingerprint used by the artifact is:

```text
9d02c723406029a95f8abfd479a1fc819fc4a0e9f698b655b2f8d7e87bbb554d
```

## Automated validation

The dedicated M0 harness passed all five gates. The permanent M0 regression test passed together with the existing Stage D readiness tests: **11 tests passed**. The complete CDI test suite passed **305 tests**.

The implementation is contained in:

- `benchmarks/m0_model_readiness.py`
- `tests/test_m0_model_readiness.py`

## Promotion decision

**M0 is complete and passed.** The M0 checkpoint is eligible to become the parent for Module M1, English Token Prediction. The next model-training objective is causal English next-token prediction on a bounded, document-disjoint corpus with held-out perplexity and fixed-prompt continuation evaluation.

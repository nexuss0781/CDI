# CDI

> **Research status:** CCT Level 1 is an evidence-gated investigation of a compact causal language engine. This repository does not currently make a production-quality, long-context, throughput, or fluency claim.

## Active Implementation

The active language-engine path is **`cdi.v3`**. It uses the EthioBBPE tokenizer adapter, a compact selective cohomodynamic recurrent state-space model, a sparse graph-Laplacian correction, a tied output projection, and matched GRU/Transformer baselines. The authoritative execution protocol and gates are recorded in [`Todo.md`](Todo.md).

| Document | Purpose |
|---|---|
| [`Architecture.md`](Architecture.md) | Source-grounded description of the active CCT implementation, its mathematical structure, strengths, and current boundaries. |
| [`ISSUES_TODO.md`](ISSUES_TODO.md) | Prioritized engineering backlog, root causes, acceptance tests, and bounded remediation status. |
| [`Todo.md`](Todo.md) | Authoritative gated CCT checklist. |
| [`colab.md`](colab.md) | CPU-safe Colab workflow for an approved CCT experiment. |
| [`docs/CCT_G2_1_DECISION.md`](docs/CCT_G2_1_DECISION.md) | Recorded G2.1 decision: `REDESIGN_BEFORE_SCALE`. |
| [`docs/CCT_G3_1_DECISION.md`](docs/CCT_G3_1_DECISION.md) | Recorded G3.1 result: `EARNED_GEOMETRY_EVIDENCE`; global quality remains `REDESIGN_BEFORE_SCALE`. |
| [`docs/CCT_G3_2_DECISION.md`](docs/CCT_G3_2_DECISION.md) | Recorded G3.2 result: `EARNED_READOUT_EVIDENCE`; geometry re-confirmed; global quality remains `REDESIGN_BEFORE_SCALE`. |
| [`docs/CCT_G3_3_PREREGISTRATION.md`](docs/CCT_G3_3_PREREGISTRATION.md) | Approved G3.3 harmonic-memory-band control, local gates, five-model matrix, and non-scaling decision rules. |
| [`docs/CCT_G3_3_DECISION.md`](docs/CCT_G3_3_DECISION.md) | Recorded G3.3 result: `EARNED_HARMONIC_EVIDENCE`; geometry re-confirmed; global quality remains `REDESIGN_BEFORE_SCALE`. |
| [`docs/CCT_G3_4_PREREGISTRATION.md`](docs/CCT_G3_4_PREREGISTRATION.md) | Approved G3.4 bounded selective token-residual quality-recovery mechanism, exact control, and material-quality gate. |
| [`benchmarks/cct_g3_3_harmonic_ablation.py`](benchmarks/cct_g3_3_harmonic_ablation.py) | CPU-safe executable G3.3 harness with 11 GiB guard and fixed-held-out state/gradient diagnostics. |
| [`benchmarks/cct_g3_4_token_residual.py`](benchmarks/cct_g3_4_token_residual.py) | CPU-safe executable G3.4 five-model quality-recovery harness with a 2% material-GRU target. |

CCT-G3.1 established repeated sparse-geometry value, CCT-G3.2 established repeated contrast-readout value, and CCT-G3.3 established repeated harmonic-memory-band value under capacity-matched three-seed comparisons. Full CDI nevertheless remained above GRU in all three seeds. The only active work is **CCT-G3.4**, a pre-registered bounded token-residual quality-recovery evaluation; the CCT-G2.2 scale ladder remains blocked.

## Safe Setup and Validation

Use Python 3.12 or a compatible environment, then install the declared dependency contract.

```bash
git clone --branch master --single-branch https://github.com/nexuss0781/CDI.git
cd CDI
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import ethiobbpe; print('EthioBBPE OK:', ethiobbpe.__file__)"
pytest -q
```

The complete current regression command is `pytest -q`. The repository-safe convenience interface does not train or fetch external data:

```bash
./run.sh status
./run.sh readiness
```

`./run.sh readiness` runs the clean-master CCT-G0 environment and regression verifier. Generated benchmark artifacts are written under their configured `results/` directories rather than tracked documentation paths.

## CCT Execution Discipline

Every CCT experiment must retain the approved tokenizer artifact, governed split, context, optimizer, precision, token budget, seed list, and matched baseline protocol unless the Todo gate explicitly authorizes a single change. A result must be reviewed against the complete executable CCT transition rule, including finite values, learning, Transformer tolerance, and the declared GRU relation.

> **Do not use `bash run.sh production`, `train.py`, or the top-level `cdi` legacy API as CCT evidence.** External production ingestion/training is fail-closed because it lacks an approved immutable source registry and split-safe data contract. The v2 code remains as a legacy mathematical reference, not the active CCT training route.

## Repository Structure

```text
cdi/v3/                         Active CCT tokenizer, DCSS recurrence, language model, training, and verified inference
benchmarks/ethiobbpe_synaxarium_pilot.py
                                Matched real-data CCT pilot harness
benchmarks/stage_*.py           Historical/diagnostic benchmark gates; write evidence to results/
tests/                          Unit, integration, CCT, production-boundary, and regression tests
scripts/run_cct_g0.sh          Clean-master reproducibility verifier
```

## License

This project is released under the repository [`LICENSE`](LICENSE).

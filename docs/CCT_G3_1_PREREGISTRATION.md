# CCT-G3.1 Pre-registration: Geometry Observability Through State-to-Readout Access

> **Status:** `IMPLEMENTATION_AND_UNIT_GATES_ONLY`. This document authorizes one narrowly defined code change and its local correctness tests. It does **not** authorize a scale increase, a corpus change, a context change, or a positive architecture claim.

## Problem Statement

CCT-G2.1 established that the active DCSS model learns stably but loses to GRU in every seed. The review found that the weighted graph-Laplacian correction changes vertex-resolved state contrast while the current readout averages every band across vertices. Since the Laplacian has zero vertex mean, geometry had zero causal-loss gradient and no material effect on token logits.

## Single Changed Mechanism

The only changed mechanism is **state-to-readout access**. For each memory band \(z_b\in\mathbb{R}^{V\times w}\), replace the prior mean-only feature

\[
\operatorname{mean}_V(z_b)\in\mathbb{R}^{w}
\]

with its mean plus fixed zero-sum vertex contrasts. Let \(Q\in\mathbb{R}^{V\times(V-1)}\) be a deterministic orthonormal basis of the subspace perpendicular to \(\mathbf{1}\). The readout feature is

\[
\phi(z_b)=\left[\frac{1}{V}\mathbf{1}^{\top}z_b,\ Q^{\top}z_b\right].
\]

For the nano configuration \(V=w=4\), each band contributes 16 values rather than four. Across three bands the linear readout receives 48 values. The recurrence, topology, Laplacian, gates, integrator, tokenizer, tied vocabulary projection, data split, optimizer, precision, seeds, context, and causal loss remain unchanged.

The **full** and **geometry-disabled** variants use this exact same readout. The geometry-disabled variant remains an exact no-Laplacian counterpart, not a different capacity model.

## Parameter Control

The readout increases from 52 to 196 parameters, a difference of 144. With the 16,000-token tied projection the total CDI model count is expected to remain within one percent of the matched GRU and Transformer totals. The implementation must record all model counts and enforce the declared tolerance before an empirical run.

## Local Unit Gates

Before CCT-G3.1 Colab execution, the implementation must demonstrate all of the following.

| Gate | Required result |
|---|---|
| Causality | A perturbation at a future token cannot change earlier logits. |
| Numerical validity | Forward values, loss, state, and gradients are finite. |
| Geometry observability | Identically initialized full and geometry-disabled models differ in logits and causal loss by more than a declared floating-point threshold. |
| Gradient reachability | Full-model causal loss gives finite, nonzero gradient norm to geometry edge weights. |
| Exact ablation | The geometry-disabled model produces zero geometry correction while retaining the same contrast readout and parameter count. |
| Existing recurrence guarantees | Step/chunk equivalence, state serialization, padding behavior, and sparse-allocation guards remain valid. |

## Frozen Empirical Contract

If and only if the local unit gates pass, the next user-run CCT-G3.1 experiment will use the CCT-G2.1 contract: 321 deduplicated governed documents, the existing manifest split, EthioBBPE artifact, seeds `[11, 29, 47]`, 1,000 steps, chunk length 16, batch size 2, 30,000 causal positions per model/seed, AdamW learning rate 0.01, CPU float32, deterministic per-epoch shuffle, and all held-out evaluation batches.

The comparison set is: full geometry-observable CDI, exact geometry-disabled CDI, GRU, and Transformer. No CCT-G2.2 step increase is permitted from the implementation result alone.

## Empirical Decision Rule

The report must show seed-level finite values, training-loss direction, validation/test loss, token accuracy, parameter count, throughput, full-versus-geometry-disabled effect, CDI-to-GRU relation, and CDI-to-Transformer relation. It must distinguish a mechanism signal from an overall architecture pass. A null or negative geometry effect remains a CCT-G3 failure and does not unlock scaling.

## Amendment A — Bounded Geometry-Weight Parameterization

The first submitted CCT-G3.1 execution on master `bcc7e0b` stopped before producing any model metric because AdamW increased a `softplus(edge_log_weights)` value above the declared maximum edge weight. The earlier implementation rejected that valid optimizer state after the update, which preserved the numerical bound but made the controlled experiment non-executable.

This amendment changes **only the safety parameterization of the existing geometry edge weight**. It does not change the recurrence, topology, readout, tokenizer, corpus, split, seed list, context, optimizer, parameter count, or comparison set. The effective initial weights are analytically preserved: the initial softplus weights are converted to logits whose bounded sigmoid has the same values. During training the effective weights are

\[
w_e = w_{\max}\,\sigma(\theta_e), \qquad 0 < w_e < w_{\max}.
\]

The same `max_geometry_edge_weight` and explicit-step stability envelope therefore remain enforced continuously rather than by terminating after an optimizer update. The full and geometry-disabled models retain the exact same parameter count and mapping. Before rerun, tests must establish finite forward/backward values, strict edge-weight bounds even for extreme raw logits, and preserved geometry-loss reachability.

This amendment permits one clean rerun of the same frozen CCT-G3.1 command. It does not authorize an architecture-quality claim, additional training budget, or any scale progression.

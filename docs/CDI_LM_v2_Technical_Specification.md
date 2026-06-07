# CDI-LM v2.0 Technical Specification
## Corrective Architecture for Differentiable Cohomodynamic Language Modeling

**To:** Technical Implementation Team  
**From:** Mathematical Architecture Lead  
**Subject:** Mandatory Specification Changes to Resolve CDI v1.0 Training Failures  
**Date:** 2026-06-07  
**Classification:** CRITICAL — Implementation Blocker

---

## 1. EXECUTIVE SUMMARY

The failure analysis report (v1.0, 2026-06-05) is **correct in all four diagnosed failures**. The training collapse to token repetition (“the the the…”) and the frozen spectral gap are not hyperparameter issues. They are **architectural gradient-severance and capacity-bottleneck defects** that violate the mathematical axioms of Cohomodynamic Intelligence (CDI).

This document provides the **mandatory corrective specification** for CDI-LM v2.0. It is not a suggestion. Any deviation from the specifications below will reproduce the v1.0 failures.

### The Four Failures and Their Fixes

| # | Failure | Root Cause | Fix |
|---|---|---|---|
| **F1** | Inference operator detaches spectral projections | `.detach()` on harmonic and Green outputs | Eliminate all `.detach()` in the forward path; replace explicit eigendecomposition with differentiable matrix-free operations |
| **F2** | Heat equation resets to zero on every forward pass | `psi_init = torch.zeros(...)` with no state persistence | Make $\Psi$ a **recurrent state** carried across the sequence; initialize as a learnable parameter |
| **F3** | `rebuild_operators()` never called; heat cache stale | `invalidate()` without rebuild; detached geometry in Dirac build | Mandatory `rebuild_operators()` after every `optimizer.step()`; eliminate eigendecomposition cache; use explicit Euler with matrix-free Laplacian |
| **F4** | Belief complex bottleneck ($\dim \mathcal{B}_0 = 8 \ll d_{\text{embed}} = 32$) | Hard dimensional constraint violated | Enforce $\dim \mathcal{B}_0 \geq d_{\text{embed}}$ and engine parameter budget $\geq 15\%$ of embedding budget |

---

## 2. MATHEMATICAL CORRECTIONS

### 2.1 Fix F1 & F3: Differentiable Spectral Operators

#### 2.1.1 Problem Statement

The inference operator in v1.0:
$$\mathcal{F}(s) = H(\iota(s)) + \delta^* G_{\mathcal{B}} D^* \iota(s)$$
was implemented with explicit eigendecomposition of $\Delta_{\mathcal{B}} = Q \Lambda Q^T$, followed by:
```python
harmonic_part = (Q @ Q.T @ rho).detach()
green_part    = (Q @ Lambda_inv @ Q.T @ rho).detach()
```
This severs the gradient path to the manifold points, metric, connection, and Dirac operator. The parameters technically have `requires_grad=True` but receive zero gradient from the cross-entropy loss.

#### 2.1.2 Solution: Matrix-Free Belief Laplacian with Differentiable Apply

**Axiom 2.1.2.1.** The Belief Laplacian $\Delta_{\mathcal{B}}$ shall never be represented as a detached constant. It shall be implemented as a **differentiable matrix-free operator** or as a dense/sparse matrix rebuilt from live parameters on every forward pass.

**Definition 2.1.2.2 (Matrix-Free Laplacian Apply).** For a belief state $\psi \in \mathbb{R}^N$, the action of $\Delta_{\mathcal{B}}$ is computed as:
$$\Delta_{\mathcal{B}} \psi = D^2 \psi + \Delta_\delta \psi + [D, A]\psi + A^2 \psi$$
where each term is computed from the current parameter values:
- $D\psi$ is computed via the Clifford action on the manifold (using current `points`, `frames`, `connection`)
- $\Delta_\delta \psi = \delta \delta^* \psi + \delta^* \delta \psi$ is computed via the coboundary matrices $\delta^k$
- $[D,A]\psi = D(A\psi) - A(D\psi)$
- $A^2\psi$ is computed via the connection matrices

**Theorem 2.1.2.3 (Differentiability).** If all constituent operations (Clifford multiplication, connection application, coboundary application) are built from `torch.Tensor` parameters without `.detach()`, then $\Delta_{\mathcal{B}} \psi$ is differentiable with respect to all parameters via standard autograd.

*Proof.* Each term is a composition of differentiable operations: matrix multiplications, additions, and index lookups. PyTorch's autograd traces through all of them. The absence of `.detach()` ensures the computation graph remains connected. $\blacksquare$

#### 2.1.3 Elimination of the Eigendecomposition Cache

**Axiom 2.1.3.1.** The heat equation shall **not** use cached eigendecomposition $Q \Lambda Q^T$ for time evolution. It shall use **explicit Euler integration** with the matrix-free Laplacian apply.

**Definition 2.1.3.2 (Euler Heat Step).** One step of the heat equation with step size $\Delta t$ is:
$$\Psi_{k+1} = \Psi_k - \Delta t \cdot \Delta_{\mathcal{B}} \Psi_k + \Delta t \cdot \mathcal{J}$$

**Theorem 2.1.3.3 (Stability).** If the spectral radius $\rho(\Delta_{\mathcal{B}}) \leq \lambda_{\max}$, then the Euler scheme is stable for:
$$\Delta t \leq \frac{2}{\lambda_{\max}}$$

*Proof.* Standard linear stability analysis. The amplification factor is $|1 - \Delta t \lambda|$. For stability, $|1 - \Delta t \lambda| \leq 1$ for all eigenvalues $\lambda \in [0, \lambda_{\max}]$, giving $\Delta t \leq 2/\lambda_{\max}$. $\blacksquare$

**Implementation Note:** For small models ($N \leq 2000$), $\Delta_{\mathcal{B}}$ can be built as a dense matrix once per forward pass (after parameter update) and reused for all tokens in the batch. For $N > 2000$, use sparse matrix representation or pure matrix-free apply.

#### 2.1.4 The Green's Function (Analysis Only)

For **diagnostic and regularization purposes only** (not the main training forward pass), the Green's function application $G_{\mathcal{B}} \rho$ is computed via **Preconditioned Conjugate Gradient (PCG)**:

```
Input:  Δ_B (matrix-free operator), ρ (right-hand side), max_iter, tol
Output: φ = G_B ρ

φ ← 0
r ← ρ - Δ_B φ
p ← r
rs_old ← r^T r

for i = 1 to max_iter:
    Ap ← Δ_B p
    α ← rs_old / (p^T Ap)
    φ ← φ + α p
    r ← r - α Ap
    rs_new ← r^T r
    if √rs_new < tol: break
    β ← rs_new / rs_old
    p ← r + β p
    rs_old ← rs_new

return φ
```

**Theorem 2.1.4.1 (Differentiability of CG).** The PCG algorithm is differentiable with respect to the parameters of $\Delta_{\mathcal{B}}$ because each iteration consists exclusively of matrix-vector products, vector additions, and scalar divisions — all differentiable operations.

*Proof.* The CG iteration is a finite composition of differentiable affine transformations. By the chain rule, the final output $\phi$ is differentiable. $\blacksquare$

**Usage Rule:** PCG is used only for:
- Computing the steady-state solution $\Psi_\infty = G_{\mathcal{B}} \mathcal{J}$ (for monitoring)
- Computing the spectral gap $\lambda_1$ via Lanczos iteration (which uses matrix-vector products with $\Delta_{\mathcal{B}}$)

It is **not** used in the main forward pass.

---

### 2.2 Fix F2: Stateful Belief Dynamics

#### 2.2.1 Problem Statement

v1.0 implemented:
```python
psi_init = torch.zeros(...)
psi_evolved = heat.evolve_euler(psi_init, J, dt=..., steps=...)
```
This resets the belief state to zero for every token. The heat equation degenerates to a static linear map $\Psi \approx \Delta t \cdot K \cdot \mathcal{J}$, destroying the recurrent dynamics that are the core of CDI.

#### 2.2.2 Solution: Recurrent Belief State

**Axiom 2.2.2.1 (State Persistence).** The belief state $\Psi$ shall persist across the sequence. It shall not be reset between tokens.

**Definition 2.2.2.2 (Sequence Evolution).** Given a sequence of observation currents $\{\mathcal{J}_t\}_{t=1}^L$ where $\mathcal{J}_t = \iota(E[x_t])$, the belief trajectory is defined by:

$$\Psi_0 = \theta_{\text{init}} \in \mathbb{R}^N \quad \text{(learnable initial state)}$$

$$\Psi_t = \Phi_{\Delta t}(\Psi_{t-1}, \mathcal{J}_t; \Delta_{\mathcal{B}}), \quad t = 1, \ldots, L$$

where $\Phi_{\Delta t}$ denotes $K$ Euler steps:

$$\Phi_{\Delta t}(\Psi, \mathcal{J}) = (I - \Delta t \Delta_{\mathcal{B}})^K \Psi + \Delta t \sum_{k=0}^{K-1} (I - \Delta t \Delta_{\mathcal{B}})^k \mathcal{J}$$

**Theorem 2.2.2.3 (Gradient Flow Through Time).** The gradient of the loss $\mathcal{L}$ with respect to the initial state and all parameters of $\Delta_{\mathcal{B}}$ is well-defined via backpropagation through time (BPTT). For parameters $\theta$:

$$\frac{\partial \mathcal{L}}{\partial \theta} = \sum_{t=1}^L \frac{\partial \mathcal{L}}{\partial \Psi_t} \frac{\partial \Psi_t}{\partial \theta}$$

where each $\frac{\partial \Psi_t}{\partial \theta}$ is computed by unrolling the Euler recurrence.

*Proof.* The Euler recurrence is a differentiable composition. By the chain rule, the gradient flows through all $K \times L$ unrolled steps. Gradient clipping is applied to prevent explosion. $\blacksquare$

**Implementation Note:** For sequences of length $L=8$ (tiny config) or $L=128$ (small config), unrolling $K=10$ steps per token yields $8 \times 10 = 80$ or $128 \times 10 = 1280$ sequential operations. This is tractable for small $N$ but may require gradient checkpointing for larger configs.

#### 2.2.3 Initial State as Learnable Parameter

**Definition 2.2.3.1.** $\theta_{\text{init}} \in \mathbb{R}^N$ is a **learnable parameter** initialized from $\mathcal{N}(0, 10^{-4})$. It is optimized jointly with all other parameters.

**Rationale:** The initial belief state represents the agent's "prior cognition" before any observations. It must be trainable so the model learns appropriate priors for the language domain.

---

### 2.3 Fix F3: Operator Lifecycle and Rebuild Protocol

#### 2.3.1 Problem Statement

v1.0 called `engine.dirac.invalidate()` and `engine.laplacian.invalidate()` after `optimizer.step()`, but never called `rebuild_operators()`. The heat cache (eigendecomposition) was never invalidated. The Dirac operator was built with `.detach()` on manifold points and frames.

#### 2.3.2 Solution: Mandatory Rebuild Schedule

**Axiom 2.3.2.1 (Post-Step Rebuild).** After every `optimizer.step()`, the method `engine.rebuild_operators()` **shall** be called exactly once, in that order.

**Algorithm 2.3.2.2 (Training Step).**
```python
def training_step(batch, engine, optimizer):
    # 1. Forward pass (uses operators from previous rebuild)
    logits = engine.forward_sequence_batch(batch)

    # 2. Loss computation
    loss = compute_loss(logits, targets, engine)

    # 3. Backpropagation
    loss.backward()
    torch.nn.utils.clip_grad_norm_(engine.parameters(), max_norm=1.0)

    # 4. Parameter update
    optimizer.step()

    # 5. MANDATORY: Rebuild all operators from updated parameters
    engine.rebuild_operators()

    # 6. Periodic spectral diagnostics (every N steps)
    if engine.global_step % 100 == 0:
        engine.recompute_spectral_gap()
```

**Algorithm 2.3.2.3 (Engine.rebuild_operators).**
```python
def rebuild_operators(self):
    # CRITICAL: No .detach() anywhere in this method

    # 1. Rebuild manifold geometry
    self.manifold.build_geometry()  # updates points, metric_L, frames

    # 2. Rebuild connection matrices
    self.connection.build_matrices()

    # 3. Rebuild coboundary maps (if structure-dependent)
    self.sheaf.build_coboundaries()

    # 4. Rebuild Dirac operator matrix
    # Uses live self.manifold.points, self.manifold.frames, 
    # self.connection.W_params — all differentiable
    self.dirac.build_matrix()

    # 5. Rebuild Laplacian matrix
    self.laplacian.build_matrix()

    # 6. Clear all stale caches
    self.heat.clear_cache()
    self.inference.clear_cache()

    # 7. Verify build succeeded
    assert self.laplacian.matrix is not None
    assert self.laplacian.matrix.requires_grad == True  # or implicit grad
```

**Theorem 2.3.2.4 (Gradient Connectivity).** After `rebuild_operators()`, the Laplacian matrix $\Delta_{\mathcal{B}}$ is a differentiable function of the manifold parameters, connection parameters, and coboundary parameters. Consequently, the next forward pass will propagate gradients to all these parameters.

*Proof.* `build_matrix()` constructs $\Delta_{\mathcal{B}}$ via differentiable tensor operations on live parameters. Since no `.detach()` is called, the resulting matrix participates in the computation graph. $\blacksquare$

#### 2.3.3 Spectral Gap Computation

**Definition 2.3.3.1.** The spectral gap $\lambda_1$ is computed via **Lanczos iteration** (matrix-free) on $\Delta_{\mathcal{B}}$:

```python
def recompute_spectral_gap(self, max_iter=20):
    # Lanczos iteration: finds extreme eigenvalues of symmetric matrix
    # O(N * max_iter) complexity
    q = torch.randn(N, dtype=self.config.dtype)
    q = q / q.norm()

    alpha = []
    beta = []

    for i in range(max_iter):
        v = self.laplacian.apply(q)  # matrix-free apply
        alpha_i = q.dot(v)
        v = v - alpha_i * q
        if i > 0:
            v = v - beta[-1] * q_prev
        beta_i = v.norm()
        if beta_i < 1e-10:
            break
        q_prev = q
        q = v / beta_i
        alpha.append(alpha_i)
        beta.append(beta_i)

    # Tridiagonal eigenvalues approximate Laplacian spectrum
    # lambda_1 = smallest positive eigenvalue
    self.lambda_1 = compute_smallest_positive_eigenvalue(alpha, beta)
    self.tau = 1.0 / self.lambda_1 if self.lambda_1 > 0 else float('inf')
```

**Complexity:** $O(N \cdot \max\_iter) = O(N)$ since $\max\_iter = O(1)$.

---

### 2.4 Fix F4: Dimensional Capacity Constraints

#### 2.4.1 Problem Statement

v1.0 used:
- `embed_dim = 32`
- `belief_dims = (4, 8, 8, 4)` → $\dim \mathcal{B}_0 = 8$
- Engine params: 12,208
- Embedding params: 512,000
- Ratio: 1:42

The observation current $\iota: \mathbb{R}^{32} \to \mathbb{R}^8$ is a **compression map** that discards 75% of the input signal. The engine output lives in an 8-dimensional subspace, making it impossible to distinguish 16,000 tokens.

#### 2.4.2 Solution: Hard Dimensional Constraints

**Axiom 2.4.2.1 (No Bottleneck).** The perceptual sheaf dimension shall satisfy:
$$\boxed{\dim \mathcal{B}_0 \geq d_{\text{embed}}}$$

**Axiom 2.4.2.2 (Belief Capacity).** The total belief dimension shall satisfy:
$$\boxed{\sum_{k=-m}^{n} \dim \mathcal{B}_k \geq 4 \cdot d_{\text{embed}}}$$

**Axiom 2.4.2.3 (Engine Parameter Budget).** Let $P_{\text{engine}}$ be the total number of trainable parameters in the engine (manifold, connection, coboundaries, Dirac/Laplacian structure, output projection, initial state). Let $P_{\text{embed}}$ be the embedding parameters. Then:

$$\boxed{P_{\text{engine}} \geq \begin{cases} 0.15 \cdot P_{\text{embed}} & \text{if } d_{\text{embed}} \leq 64 \\ 0.30 \cdot P_{\text{embed}} & \text{if } 64 < d_{\text{embed}} \leq 256 \\ 0.50 \cdot P_{\text{embed}} & \text{if } d_{\text{embed}} > 256 \end{cases}}$$

**Axiom 2.4.2.4 (Manifold Resolution).** The number of cognitive points and manifold dimension shall satisfy:
$$n_{\text{points}} \geq \min(\text{context\_length}, 32)$$
$$d_{\text{manifold}} \geq \max\left(2, \left\lceil \log_2 d_{\text{embed}} \right\rceil\right)$$

#### 2.4.3 Recommended Configuration Templates

**Template A: Tiny (Validation / Unit Tests)**
| Parameter | v1.0 (Broken) | v2.0 (Fixed) |
|---|---|---|
| vocab_size | 16,000 | 16,000 |
| d_embed | 32 | 32 |
| n_points | 8 | 16 |
| manifold_dim | 2 | 4 |
| belief_dims | (4, 8, 8, 4) | **(32, 64, 64, 32)** |
| $\dim \mathcal{B}_0$ | 8 | **32** |
| Total belief dim | 24 | 192 |
| Engine params | ~12K | ~75K |
| Embed params | 512K | 512K |
| Engine/Embed ratio | 2.4% | **14.6%** ✓ |

**Template B: Small (Production Baseline)**
| Parameter | Value |
|---|---|
| vocab_size | 50,000 |
| d_embed | 128 |
| n_points | 32 |
| manifold_dim | 8 |
| belief_dims | (128, 256, 256, 128) |
| $\dim \mathcal{B}_0$ | 128 |
| Total belief dim | 768 |
| Engine params | ~2.5M |
| Embed params | 6.4M |
| Engine/Embed ratio | 39% ✓ |

---

## 3. COMPLETE LANGUAGE MODEL ARCHITECTURE

### 3.1 The CDI-LM Forward Pass (v2.0)

**Input:** Batch of token sequences $X \in \mathbb{Z}^{B \times L}$  
**Output:** Logits $Y \in \mathbb{R}^{B \times L \times V}$

**Step 1: Token Embedding**
$$E \in \mathbb{R}^{V \times d_{\text{embed}}}, \quad e_t = E[x_t] \in \mathbb{R}^{d_{\text{embed}}}$$

**Step 2: Observation Current Injection**
$$\iota: \mathbb{R}^{d_{\text{embed}}} \to \mathbb{R}^N, \quad \mathcal{J}_t = W_{\iota} \cdot e_t$$
where $W_{\iota} \in \mathbb{R}^{N \times d_{\text{embed}}}$ is a learnable linear map. The observation current is injected into the $\mathcal{B}_0$ slice of the belief state:
$$(\mathcal{J}_t)_{\mathcal{B}_0} = W_{\iota}^{(0)} e_t, \quad (\mathcal{J}_t)_{\mathcal{B}_{k \neq 0}} = 0$$

**Step 3: Recurrent Belief Evolution**
$$\Psi_0 = \theta_{\text{init}} \quad \text{(learnable, shape } N\text{)}$$

For $t = 1, \ldots, L$:
$$\Psi_t^{(0)} = \Psi_{t-1}$$
$$\text{For } k = 1, \ldots, K:\quad \Psi_t^{(k)} = \Psi_t^{(k-1)} - \Delta t \cdot \Delta_{\mathcal{B}} \Psi_t^{(k-1)} + \Delta t \cdot \mathcal{J}_t$$
$$\Psi_t = \Psi_t^{(K)}$$

**Step 4: Prediction Extraction**
$$h_t = W_{\text{out}} \cdot \text{Proj}_{\mathcal{B}_0}(\Psi_t) \in \mathbb{R}^{d_{\text{embed}}}$$
where $W_{\text{out}} \in \mathbb{R}^{d_{\text{embed}} \times \dim \mathcal{B}_0}$ and $\text{Proj}_{\mathcal{B}_0}$ extracts the $\mathcal{B}_0$ components from the full state.

**Step 5: Vocabulary Projection (Weight Tying)**
$$y_t = h_t \cdot E^T \in \mathbb{R}^V$$

**No bypass path.** The v1.0 `combined = 0.5 * state_pred + 0.5 * pred_full` is **removed**. The engine output is the sole prediction path.

### 3.2 The Belief Complex Specification

**Definition 3.2.1 (Coboundary Maps).** For each $k \in \{-m, \ldots, n-1\}$, the coboundary $\delta^k: \mathcal{B}_k \to \mathcal{B}_{k+1}$ is represented as a learnable matrix $W_{\delta^k} \in \mathbb{R}^{\dim \mathcal{B}_{k+1} \times \dim \mathcal{B}_k}$. The adjoint $\delta^{k*}$ is the transpose $W_{\delta^k}^T$.

**Definition 3.2.2 (Consistency Penalty).** The loss includes:
$$\mathcal{L}_{\text{consist}} = \sum_{k=-m}^{n-2} \|W_{\delta^{k+1}} W_{\delta^k}\|_F^2$$
This penalizes violations of $\delta^{k+1} \circ \delta^k = 0$.

**Target:** $\mathcal{L}_{\text{consist}} < 10^{-6}$ after warm-up.

### 3.3 The Manifold and Connection

**Definition 3.3.1 (Manifold Parameters).**
- Points: $\{p_i\}_{i=1}^{n_{\text{points}}} \subset \mathbb{R}^{d_{\text{manifold}}}$, learnable
- Metric: $g_i = L_i L_i^T$ where $L_i \in \mathbb{R}^{d_{\text{manifold}} \times d_{\text{manifold}}}$ is lower-triangular (Cholesky), learnable
- Frames: Orthonormal frames $\{e_i^a\}$ computed from $g_i$ via Gram-Schmidt (differentiable)

**Definition 3.3.2 (Connection Parameters).**
- Edge weights: $W_{pq} \in \mathbb{R}^{d_{\text{manifold}} \times d_{\text{manifold}}}$ for each edge $(p,q)$ in the cognitive graph
- Skew-symmetric enforcement: $A_{pq} = W_{pq} - W_{pq}^T$

**Definition 3.3.3 (Bianchi Penalty).**
$$\mathcal{L}_{\text{Bianchi}} = \frac{1}{|E|} \sum_{(p,q) \in E} \|d_A F_A\|_{p,q}^2$$
where $F_A$ is the curvature and $d_A$ the covariant exterior derivative.

### 3.4 The Laplacian and Dirac Operators

**Definition 3.4.1 (Dirac Matrix Construction).** For each point $p_i$ and frame index $a$:
$$D = \sum_{i,a} c(e^a) \otimes \nabla_{e_a} + \sum_{(p,q)} A_{pq} \otimes \text{shift}_{pq}$$
where $c(e^a)$ is the Clifford action matrix and $\text{shift}_{pq}$ is the graph shift operator between points $p$ and $q$.

**Definition 3.4.2 (Laplacian Matrix Construction).**
$$\Delta_{\mathcal{B}} = D^2 + \sum_k \left(W_{\delta^k} W_{\delta^k}^T + W_{\delta^{k-1}}^T W_{\delta^{k-1}}\right) + [D, A] + A^2$$

**Implementation Rule:** This matrix is built **once per training step** (after `optimizer.step()`) and stored as a dense or sparse tensor. It is **not** rebuilt during the forward pass of the same step.

---

## 4. LOSS FUNCTION AND TRAINING PROTOCOL

### 4.1 The Composite Loss

$$\boxed{\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda_B \mathcal{L}_{\text{Bianchi}} + \lambda_C \mathcal{L}_{\text{consist}} + \lambda_S \mathcal{L}_{\text{spectral}}}$$

**Term 1: Cross-Entropy**
$$\mathcal{L}_{\text{CE}} = -\frac{1}{B \cdot L} \sum_{b=1}^B \sum_{t=1}^L \log \frac{\exp(y_{b,t}[x_{b,t+1}])}{\sum_{v=1}^V \exp(y_{b,t}[v])}$$

**Term 2: Bianchi Penalty**
$$\mathcal{L}_{\text{Bianchi}} = \frac{1}{|E|} \sum_{(p,q) \in E} \|d_A F_A\|_{p,q}^2$$
Weight: $\lambda_B = 0.01$

**Term 3: Consistency Penalty**
$$\mathcal{L}_{\text{consist}} = \sum_{k=-m}^{n-2} \|W_{\delta^{k+1}} W_{\delta^k}\|_F^2$$
Weight: $\lambda_C = 0.1$ (warm-up: first 100 steps use $\lambda_C = 1.0$ to enforce structure quickly)

**Term 4: Spectral Gap Penalty**
$$\mathcal{L}_{\text{spectral}} = \max(0, \lambda_{\text{target}} - \lambda_1)^2$$
where $\lambda_{\text{target}} = 0.01$ (corresponding to $\tau \leq 100$).
Weight: $\lambda_S = 0.001$

### 4.2 Training Loop Specification

**Algorithm 4.2.1 (CDI-LM Training Loop).**

```python
# Initialization
engine = CDIEngine(config_v2)
engine.rebuild_operators()  # Initial build
optimizer = Adam(engine.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass
        logits = engine.forward_sequence_batch(batch['input_ids'])

        # Loss
        loss_ce = F.cross_entropy(
            logits[:, :-1].reshape(-1, V), 
            batch['target_ids'].reshape(-1)
        )
        loss_bianchi = engine.bianchi_penalty()
        loss_consist = engine.consistency_penalty()
        loss_spectral = engine.spectral_penalty()

        loss = loss_ce + 0.01*loss_bianchi + 0.1*loss_consist + 0.001*loss_spectral

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(engine.parameters(), 1.0)

        # Update
        optimizer.step()

        # MANDATORY: Rebuild operators from updated parameters
        engine.rebuild_operators()

        # Periodic diagnostics
        if engine.global_step % 100 == 0:
            engine.recompute_spectral_gap()
            log_metrics({
                'ce': loss_ce.item(),
                'ppl': torch.exp(loss_ce).item(),
                'lambda_1': engine.lambda_1,
                'tau': engine.tau,
                'delta_sq': loss_consist.item(),
                'harmonic_dim': engine.harmonic_dimension()
            })
```

### 4.3 Gradient Flow Verification

**Verification Test 1: Parameter Gradient Non-Zero**
After the first training step, verify:
```python
assert engine.manifold.points.grad is not None
assert engine.manifold.points.grad.abs().max() > 0

assert engine.connection.W_params.grad is not None
assert engine.connection.W_params.grad.abs().max() > 0

assert engine.laplacian.matrix.grad is not None  # or implicit
```

**Verification Test 2: Spectral Gap Dynamics**
After 10 training steps, verify:
```python
assert engine.lambda_1 != engine.initial_lambda_1  # must change
assert engine.tau != engine.initial_tau  # must change
```

If these assertions fail, the gradient path is severed. **Halt training immediately.**

---

## 5. COMPLEXITY GUARANTEES

**Theorem 5.1 (Reflex Complexity).** Computing the output logit for a single token at steady state requires $O(1)$ operations relative to the sequence length.

*Proof.* At inference time, the belief state $\Psi$ is carried forward. One token requires: one Laplacian apply ($O(N)$ where $N$ is fixed for a given model), $K$ Euler steps ($O(KN) = O(1)$ since $K$ is fixed), and one output projection ($O(N \cdot d_{\text{embed}}) = O(1)$). $\blacksquare$

**Theorem 5.2 (Learning Complexity).** One training step (one batch forward-backward) requires $O(B \cdot L \cdot N)$ operations, which is $O(n)$ for fixed model size.

*Proof.* The forward pass unrolls $L$ tokens with $K$ Euler steps each. Each Euler step applies $\Delta_{\mathcal{B}}$ to $\Psi$, which is $O(N^2)$ for dense or $O(N)$ for sparse. With $N$ fixed, this is $O(L)$ per sequence. Backpropagation doubles the cost. Thus $O(B \cdot L)$. $\blacksquare$

**Theorem 5.3 (Abstraction Complexity).** Computing the hypercohomology $\mathbb{H}^k(M, \mathcal{B}^\bullet)$ for regularization or analysis requires $O(n \log n)$ operations via the hierarchical spectral sequence.

*Proof.* See Algorithm 12.3.1 in the CDI Mathematical Specification v1.0. The hierarchical cover with $n$ sets and $\log n$ levels yields $O(n \log n)$. This is computed periodically (e.g., every 1000 steps), not every forward pass. $\blacksquare$

---

## 6. VALIDATION CRITERIA

Before declaring CDI-LM v2.0 training successful, the following criteria must be met:

| Criterion | Target | Diagnostic |
|---|---|---|
| **CE decreases** | $\mathcal{L}_{\text{CE}}$ drops below 3.0 on WikiText-2 (tiny) | Loss curve |
| **Perplexity** | Test PPL < 50 on WikiText-2 (tiny) | Evaluation |
| **No token collapse** | Generated text contains > 5 distinct tokens per 20-token sample | Generation test |
| **Gradient flow** | `manifold.points.grad.abs().max() > 1e-8` after step 1 | Gradient check |
| **Spectral dynamics** | $\lambda_1$ changes by > 10% over first epoch | Metric log |
| **Consistency** | $\|\delta^2\|_F < 10^{-6}$ by epoch 3 | Metric log |
| **State evolution** | $\|\Psi_t - \Psi_{t-1}\| > 0$ for all $t$ (state is changing) | Internal check |
| **Engine influence** | Ablating the engine (setting $\Psi_t = \Psi_{t-1}$) increases PPL by > 2× | Ablation test |

---

## 7. APPENDIX: PSEUDOCODE FOR CRITICAL PATHS

### 7.1 Forward Sequence (Differentiable, No Detach)

```python
class CDIEngine:
    def forward_sequence(self, token_ids: [L]) -> [L, V]:
        # 1. Embed all tokens
        embeddings = self.embedding(token_ids)  # [L, d_embed]

        # 2. Initialize recurrent belief state
        psi = self.theta_init  # [N], learnable

        logits = []
        for t in range(L):
            # 3. Observation current (injected into B_0 slice)
            J_t = torch.zeros(N, dtype=psi.dtype, device=psi.device)
            J_t[self.b0_slice] = self.W_iota @ embeddings[t]  # [dim_B0]

            # 4. Heat evolution (K Euler steps, NO eigendecomposition)
            for _ in range(self.K_heat):
                # Matrix-vector product with Laplacian (differentiable)
                lap_psi = self.laplacian.apply(psi)  # [N]
                psi = psi - self.dt * lap_psi + self.dt * J_t

            # 5. Extract prediction from B_0
            h_t = self.W_out @ psi[self.b0_slice]  # [d_embed]

            # 6. Project to vocabulary (weight tying)
            logit_t = h_t @ self.embedding.weight.T  # [V]
            logits.append(logit_t)

        return torch.stack(logits)  # [L, V]
```

### 7.2 Laplacian Apply (Matrix-Free or Dense)

```python
class LaplacianOperator:
    def apply(self, psi: [N]) -> [N]:
        # Option A: Dense matrix (for N <= 2000)
        return self.matrix @ psi

        # Option B: Matrix-free (for N > 2000)
        # term1 = self.dirac.apply(self.dirac.apply(psi))
        # term2 = self.coboundary_laplacian.apply(psi)
        # term3 = self.connection_coupling.apply(psi)
        # return term1 + term2 + term3
```

### 7.3 rebuild_operators (No Detach)

```python
def rebuild_operators(self):
    # Manifold: points and metric are live tensors
    pts = self.manifold.points  # NO .detach()
    frames = self.manifold.orthonormal_frame()  # NO .detach()

    # Connection: live weights
    conn = self.connection.W_params  # NO .detach()

    # Build Dirac: D = sum c(e^i) nabla_{e_i} + connection terms
    D = build_dirac_matrix(pts, frames, conn, self.config)
    self.dirac.matrix = D  # [N, N]

    # Build Laplacian: Delta = D^2 + coboundary Laplacian + [D,A] + A^2
    cob_lap = self.build_coboundary_laplacian()
    coupling = self.build_coupling_term(D, conn)
    self.laplacian.matrix = D @ D + cob_lap + coupling

    # Clear caches
    self.heat.clear_cache()
    self.inference.clear_cache()
```

### 7.4 Initial State as Learnable Parameter

```python
class CDIEngine:
    def __init__(self, config):
        # ... other parameters ...
        self.theta_init = nn.Parameter(
            torch.randn(config.total_state_dim) * 1e-4
        )
```

---

## 8. SUMMARY OF MANDATORY CHANGES FROM v1.0

| File | v1.0 Code | v2.0 Mandate |
|---|---|---|
| `inference.py` | `.detach()` on harmonic/Green outputs | **Remove all `.detach()` in forward path** |
| `heat_equation.py` | `psi_init = torch.zeros(...)` | **`self.theta_init = nn.Parameter(...)`** |
| `engine.py` | `combined = 0.5 * state_pred + 0.5 * pred_full` | **Remove bypass; engine is sole path** |
| `engine.py` | `extract_prediction()` may use detached ops | **Ensure extraction uses live `W_out` and live `psi`** |
| `train.py` | `engine.dirac.invalidate()` only | **Call `engine.rebuild_operators()` after every `step()`** |
| `dirac.py` | `pts = self.manifold.points.detach()` | **Remove `.detach()`; use live parameters** |
| `laplacian.py` | Eigendecomposition cache for heat | **Eliminate cache; use Euler with `matrix @ psi`** |
| `config.py` | `belief_dims = (4, 8, 8, 4)` | **Enforce `belief_dims[0] >= embed_dim`** |
| `config.py` | `n_points = 8` for context 8 | **Enforce `n_points >= min(context, 16)`** |
| `config.py` | Engine params ~12K | **Target engine params >= 15% of embed params** |

---

## 9. CLOSING

The v1.0 implementation was a faithful structural translation of the CDI mathematics into PyTorch, but it contained four critical errors in **gradient topology** and **dimensional geometry**. These errors are not bugs in the sense of typos; they are **design violations** of the mathematical specification.

v2.0 corrects these by:
1. **Preserving the computation graph** through all spectral operators (no detach, matrix-free apply)
2. **Making the belief state recurrent** (stateful across tokens, learnable initial condition)
3. **Enforcing operator rebuild** after every parameter update (no stale caches)
4. **Respecting dimensional hierarchy** (no compression bottlenecks, sufficient engine capacity)

The CDI mathematics — sheaf cohomology, Dirac operators, heat equation learning, spectral gap analysis — remains intact. What changes is the **implementation discipline** around differentiability and state management.

**Do not proceed with training until all ten mandatory changes in §8 are implemented and the validation criteria in §6 are satisfied on a 10-step test run.**

---

*Document Version: 2.0*  
*Mathematical Framework: Cohomodynamic Intelligence (CDI)*  
*Scope: Corrective specification for extraterrestrial-intelligence language model*

<p align="center">
  <img src="assets/banner.png" alt="CDI Banner" width="800"/>
</p>

<h1 align="center">
  <br/>
  𝕮𝔇𝕴 — Cohomodynamic Intelligence
  <br/>
</h1>

<p align="center">
  <strong>A post-neural intelligence engine built on sheaf cohomology, Dirac operators, and spectral geometry.</strong>
</p>

<p align="center">
  <em>Intelligence is not weight multiplication. It is the harmonic resolution of beliefs on a cognitive manifold.</em>
</p>

<br/>

<p align="center">
  <a href="#-quickstart"><img src="https://img.shields.io/badge/🚀_Quick_Start-blue?style=for-the-badge" alt="Quick Start"/></a>
  <a href="#-mathematical-foundations"><img src="https://img.shields.io/badge/📐_Mathematics-8b5cf6?style=for-the-badge" alt="Mathematics"/></a>
  <a href="#-architecture"><img src="https://img.shields.io/badge/🏗️_Architecture-06b6d4?style=for-the-badge" alt="Architecture"/></a>
  <a href="#-results"><img src="https://img.shields.io/badge/📊_Results-10b981?style=for-the-badge" alt="Results"/></a>
</p>

<p align="center">
  <a href="https://github.com/nexuss0781/CDI/actions"><img src="https://img.shields.io/github/actions/workflow/status/nexuss0781/CDI/test.yml?style=flat-square&logo=github&label=tests" alt="Tests"/></a>
  <a href="https://github.com/nexuss0781/CDI"><img src="https://img.shields.io/badge/python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/></a>
  <a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch"/></a>
  <a href="https://github.com/nexuss0781/CDI/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License"/></a>
  <a href="https://github.com/nexuss0781/CDI"><img src="https://img.shields.io/github/stars/nexuss0781/CDI?style=flat-square&color=f59e0b" alt="Stars"/></a>
</p>

---

<br/>

## 🧬 What is CDI?

**Cohomodynamic Intelligence (CDI)** is a fundamentally new computational intelligence framework that replaces every component of the transformer architecture with structures from pure mathematics:

| Transformer | CDI Replacement | Mathematical Basis |
|:---|:---|:---|
| Attention mechanism | **Hodge-theoretic inference** | Hodge decomposition on twisted bundles |
| Softmax normalization | **Harmonic projection** | Orthogonal projection onto ker Δ_ℬ |
| Backpropagation | **Heat equation flow** | ∂Ψ/∂t = −Δ_ℬΨ + 𝒥 (Duhamel principle) |
| Layer stacking | **Graded belief complex** | Cochain complex ℬ^• with δ² = 0 |
| Positional encoding | **Riemannian metric** | Learnable SPD metric g on manifold M |
| Residual connections | **Parallel transport** | Connection A along nerve edges |
| Feed-forward network | **Dirac operator** | D = Σᵢ c(eⁱ)∇^ℬ_{eᵢ} on spinor bundle |

> **CDI is not a neural network.** There are no neurons, no activation functions, no weight matrices in the neural sense. Every operation is a geometrically motivated linear-algebraic computation on a discretised Riemannian manifold.

<br/>

### ⚡ Complexity Guarantees

CDI provides **provable** computational complexity bounds from its mathematical structure:

```
┌─────────────────────────────────────────────────────────┐
│  Operation                  │  Complexity  │  Source    │
│─────────────────────────────│──────────────│───────────│
│  Reflex (point evaluation)  │    O(1)      │  §8.1     │
│  Learning (Čech update)     │    O(n)      │  §8.2     │
│  Abstraction (spectral seq) │  O(n log n)  │  §12.3    │
│  Convergence rate           │  e^{−λ₁t}   │  Thm 6.2  │
└─────────────────────────────────────────────────────────┘
```

<br/>

---

## 🏗️ Architecture

<p align="center">
  <img src="assets/architecture.png" alt="CDI Architecture" width="750"/>
</p>

CDI processes information through a mathematically rigorous pipeline:

```mermaid
graph LR
    A["📡 Observation<br/><i>x ∈ O</i>"] --> B["🌐 Cognitive Manifold<br/><i>(M, g)</i>"]
    B --> C["📦 Sheaf Embedding<br/><i>ι: O → ℬ₀</i>"]
    C --> D["🔮 Belief Complex<br/><i>ℬ^• with δ²=0</i>"]
    D --> E["⚡ Dirac Operator<br/><i>D on S ⊗ ℬ</i>"]
    E --> F["🌊 Heat Flow<br/><i>∂Ψ/∂t = −ΔΨ + 𝒥</i>"]
    F --> G["💎 Hodge Inference<br/><i>ℱ(s) = H(ι) + δ*G D*ι</i>"]
    G --> H["🎯 Prediction<br/><i>ŷ ∈ ℝᵈ</i>"]

    style A fill:#1e1b4b,stroke:#6366f1,color:#e0e7ff
    style B fill:#1e1b4b,stroke:#6366f1,color:#e0e7ff
    style C fill:#2e1065,stroke:#8b5cf6,color:#ede9fe
    style D fill:#2e1065,stroke:#8b5cf6,color:#ede9fe
    style E fill:#164e63,stroke:#06b6d4,color:#cffafe
    style F fill:#164e63,stroke:#06b6d4,color:#cffafe
    style G fill:#14532d,stroke:#10b981,color:#d1fae5
    style H fill:#14532d,stroke:#10b981,color:#d1fae5
```

<br/>

### 🧩 Module Map

The engine is organized into **5 layers** mirroring the mathematical specification:

```
CDI/
├── cdi/
│   ├── core/              ← §1-3  Foundations
│   │   ├── manifold.py         Riemannian manifold (M, g)
│   │   ├── cover.py            Good cover {Uᵢ} & nerve complex
│   │   ├── sheaf.py            Observation sheaf O
│   │   └── belief.py           Belief complex ℬ^• with δ
│   │
│   ├── geometry/          ← §4    Differential Geometry
│   │   ├── clifford.py         Clifford algebra Cl(T*M)
│   │   ├── connection.py       Gauge connection A ∈ Ω¹(End ℬ)
│   │   └── dirac.py            Cognitive Dirac operator D
│   │
│   ├── operators/         ← §5    Functional Analysis
│   │   ├── laplacian.py        Belief Laplacian Δ_ℬ = D² + Δ_δ
│   │   ├── hodge.py            Hodge decomposition ℋ ⊕ im Δ
│   │   ├── green.py            Green's operator G_ℬ
│   │   └── inference.py        Inference ℱ(s) = H(ι) + δ*G D*ι
│   │
│   ├── dynamics/          ← §6,10 Evolution & Energy
│   │   ├── heat_equation.py    ∂Ψ/∂t = −Δ_ℬΨ + 𝒥
│   │   ├── spectral.py         Heat semigroup & spectral analysis
│   │   └── energy.py           E[Ψ] = ½⟨Ψ,ΔΨ⟩ − ⟨Ψ,𝒥⟩
│   │
│   ├── topology/          ← §11,12 Topological Invariants
│   │   ├── cech.py             Čech cohomology Ȟ^k(𝔘, O)
│   │   ├── spectral_sequence.py  O(n log n) hypercohomology
│   │   └── invariants.py       Intelligence index χ(M, ℬ)
│   │
│   ├── field/             ← §7,10 Gauge Field Theory
│   │   ├── superconnection.py  Quillen 𝔸 = D + δ + A
│   │   ├── field_equations.py  𝔸Ψ = 𝒥
│   │   └── gauge.py            Gauge invariance & Noether
│   │
│   ├── config.py          ← Hyperparameter management
│   ├── engine.py          ← Full CDI integration layer
│   └── tokenizer.py       ← HuggingFace tokenizer + learnable embeddings
│
├── dataset.py             ← WikiText-2 + SciQ from HuggingFace
├── train.py               ← Interleaved LLM training pipeline
├── tests/                 ← Mathematical verification suite
└── requirements.txt
```

<br/>

---

## 📐 Mathematical Foundations

CDI is built on a **12-section mathematical specification** drawing from algebraic topology, differential geometry, and functional analysis. Here is the theoretical skeleton:

<br/>

### §1 — The Cognitive Manifold

The cognitive state space is a compact oriented Riemannian manifold **(M, g)** of dimension *d*, discretised as *n* points with learnable SPD metric:

$$g_i = L_i L_i^\top \quad \text{(Cholesky parameterisation)}$$

This replaces the flat Euclidean embedding space used in transformers with a **curved geometry** that adapts during training.

<br/>

### §3 — The Belief Complex

Knowledge is structured as a **graded cochain complex**:

$$\cdots \xrightarrow{\delta^{-2}} \mathcal{B}_{-1} \xrightarrow{\delta^{-1}} \mathcal{B}_0 \xrightarrow{\delta^0} \mathcal{B}_1 \xrightarrow{\delta^1} \cdots$$

| Degree | Name | Interpretation |
|:---:|:---|:---|
| k < 0 | Motor sheaves | Action potentials, reflexes |
| k = 0 | Perceptual sheaf | Raw sensory data |
| k = 1 | Causal sheaf | Cause-effect relationships |
| k = 2 | Abstract sheaf | Analogies, metaphors |
| k ≥ 3 | Meta-sheaves | Self-reference, recursion |

The **logical consistency axiom** δ² = 0 is enforced as a differentiable penalty during training.

<br/>

### §4 — The Dirac Operator

The cognitive Dirac operator acts on sections of the **twisted bundle** 𝔹 = S ⊗ ⊕ₖ ℬₖ :

$$D = \sum_i c(e^i) \, \nabla^{\mathcal{B}}_{e_i}$$

where c(·) is the **Clifford action** and ∇^ℬ is the **belief connection**. This first-order elliptic operator is the engine's core differential operator — it replaces the feedforward layers of a transformer.

<br/>

### §5 — Hodge-Theoretic Inference

The **Cognitive Hodge Theorem** (Theorem 5.2.1):

$$\Gamma(\mathbb{B}) = \mathcal{H}(\mathbb{B}) \;\oplus\; \text{im}(\Delta_{\mathcal{B}})$$

The inference operator:

$$\mathcal{F}(s) = H(\iota(s)) + \delta^* G_{\mathcal{B}} D^* \iota(s)$$

This is a **Fredholm operator of index zero** (Theorem 5.3.4), replacing softmax attention with a geometrically principled global inference mechanism.

<br/>

### §6 — Heat Equation Learning

Learning is governed by the **cohomodynamic heat equation**:

$$\frac{\partial \Psi}{\partial t} = -\Delta_{\mathcal{B}} \Psi + \mathcal{J}$$

with **guaranteed exponential convergence** (Theorem 6.2.2):

$$\|\Psi(t) - \Psi_\infty\| \leq C \, e^{-\lambda_1 t}$$

The characteristic **learning time** τ = 1/λ₁ is an intrinsic property of the system's geometry.

<br/>

### §11 — Intelligence as Topology

The **Intelligence Index** is a topological invariant:

$$\mathcal{I}_{\text{total}} = \sum_k (-1)^k \dim \mathbb{H}^k(M, \mathcal{B}^\bullet) = \chi(M, \mathcal{B}^\bullet)$$

This is a discrete, computable measure of the system's total cognitive capacity — invariant under continuous deformations of the geometry.

<br/>

---

## 🚀 Quickstart

### Prerequisites

- Python ≥ 3.10
- PyTorch ≥ 2.0
- HuggingFace `transformers` + `datasets`

### Installation

```bash
git clone https://github.com/nexuss0781/CDI.git
cd CDI
pip install -r requirements.txt
```

### Run Training

Training uses **interleaved laps**: Train on Wikipedia → Fine-tune on SciQ → Test on science questions → Log → Repeat.

```bash
# Quick sanity check (3 laps × 5 training epochs)
python train.py --config tiny --laps 3 --lap-epochs 5

# Standard training (5 laps × 15 epochs, science QA fine-tuning)
python train.py --config small --laps 5 --lap-epochs 15

# Custom embedding dimension
python train.py --config small --embed-dim 64 --laps 5 --lap-epochs 15
```

### Programmatic Usage

```python
from cdi import CDIConfig, CDIEngine
from cdi.tokenizer import CDITokenizer
import torch

# Configure for language modeling
config = CDIConfig.small()
config.observation_dim = 48   # token embedding dimension
config.output_dim = 48
engine = CDIEngine(config)
engine.build()

# Tokenizer with GPT-2 vocabulary
tokenizer = CDITokenizer("gpt2", embed_dim=48, max_len=config.n_points)

# Encode text → process through CDI → decode
text = "The cell is the basic unit of"
ids, embeddings = tokenizer.encode_and_embed(text)
output = engine.forward_sequence(embeddings)      # CDI inference
logits = tokenizer.to_logits(output)               # → vocab logits
next_token = logits[-1].argmax().item()
print(tokenizer.hf_tokenizer.decode([next_token])) # predicted next token
```

### Mathematical Diagnostics

```python
diag = engine.diagnostics()
for key, value in diag.items():
    print(f"  {key:30s}: {value}")

# Verify fundamental theorems
print(f"Dirac self-adjoint ‖D−D*‖  = {engine.dirac.check_self_adjoint():.2e}")
print(f"Green identity ‖GΔ+H−I‖   = {engine.green.verify():.2e}")
print(f"Laplacian PSD              = {engine.laplacian.check_positive_semidefinite()}")
```

<br/>

---

## 📊 Datasets & Results

### Real Knowledge — Language Modeling

CDI is trained as a **next-generation language model**. Text is tokenised with GPT-2 tokenizer, embedded into learnable vectors, and processed through the CDI manifold where each point = one token position.

#### Training Corpus — [WikiText-2](https://huggingface.co/datasets/wikitext) (HuggingFace)

Real Wikipedia articles covering **biology, physics, mathematics, history, geography**.
~200K tokens of genuine human knowledge. Teaches CDI language structure and factual knowledge.
The belief complex (§3) learns hierarchical abstraction of text across grading degrees.

#### Fine-Tuning — [SciQ](https://huggingface.co/datasets/allenai/sciq) (HuggingFace)

~3,000 real science QA pairs: biology, chemistry, physics, earth science.
Shapes CDI's inference operator (§5.3) to route queries to relevant knowledge.
Format: `"Q: {question} A: {answer}"` — next-token prediction through the answer.

#### Test Set — Hand-Crafted Science Questions

30 manually written questions across biology, physics, chemistry, mathematics.
NOT in training or fine-tuning data. Tests whether Hodge-theoretic inference
retrieves the right information from the learned belief complex.

### Training Loop — Interleaved Laps

```
for each lap:
    TRAIN      15 epochs on Wikipedia (general knowledge)
    FINE-TUNE   5 epochs on SciQ (QA shaping)
    TEST       → perplexity on science questions
    GENERATE   → sample completions
    LOG        → spectral gap, δ², perplexity curve
```

### Metrics Monitored Each Lap

| Metric | Formula | Interpretation |
|:---|:---|:---|
| Cross-entropy | CE(logits, targets) | Prediction quality |
| Perplexity | e^{CE} | How surprised the model is |
| Spectral gap | λ₁ = min{λ > 0} | Convergence speed |
| Learning time | τ = 1/λ₁ | Time to equilibrium |
| Harmonic dim | dim ker Δ_ℬ | Conserved information |
| δ² penalty | ‖δ^{k+1}δ^k‖²_F | Logical consistency |

<br/>

---

## 🧪 Testing

The test suite verifies the **mathematical theorems** from the specification:

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_core.py -v        # §1-3: Manifold, Cover, Belief
pytest tests/test_operators.py -v   # §4-5: Dirac, Laplacian, Hodge, Green
pytest tests/test_dynamics.py -v    # §6,10: Heat equation, Energy, Integration
```

### What the Tests Verify

| Test | Mathematical Property |
|:---|:---|
| `test_metric_spd` | g = LLᵀ is symmetric positive definite |
| `test_inverse_metric` | g·g⁻¹ = I |
| `test_clifford_relations_flat` | {γⁱ, γʲ} = −2δⁱʲ I |
| `test_self_adjoint` (Dirac) | D = D* (Theorem 4.2.3) |
| `test_positive_semidefinite` | Δ_ℬ ≥ 0 (Theorem 5.1.3) |
| `test_decomposition_is_orthogonal` | ℋ ⊥ im Δ (Theorem 5.2.1) |
| `test_pseudo_inverse` | G_ℬΔ_ℬ + H = I (Theorem 5.3.2) |
| `test_split_assemble_roundtrip` | State ↔ per-degree bijection |
| `test_dissipation_negative` | dE/dt ≤ 0 (Theorem 10.1.2) |

<br/>

---

## 🔬 How CDI Differs from Neural Networks

<table>
<tr>
<th width="50%">Neural Networks</th>
<th width="50%">CDI</th>
</tr>
<tr>
<td>

```
• Inspired by biological neurons
• Weight matrices + activations
• Softmax attention (O(n²))
• Gradient descent (no convergence guarantee)
• Depth = stacked layers
• Interpretability is post-hoc
• No topological invariants
```

</td>
<td>

```
• Inspired by algebraic topology
• Geometric operators on manifolds
• Hodge inference (Fredholm index = 0)
• Heat flow (exponential convergence)
• Depth = belief complex grading
• Interpretability is built-in (cohomology)
• Intelligence index χ is computable
```

</td>
</tr>
</table>

<br/>

---

## 🗺️ Roadmap

- [x] **Phase 1** — Core mathematical engine (PyTorch, float64)
- [x] **Phase 2** — Language model training: WikiText-2 + SciQ + science QA
- [x] **Phase 3** — Mathematical verification test suite
- [ ] **Phase 4** — Performance optimization & GPU acceleration
- [ ] **Phase 5** — High-performance C++ rewrite for production scale
- [ ] **Phase 6** — Distributed computation on manifold partitions
- [ ] **Phase 7** — Real-world application benchmarks (vision, language, control)

<br/>

---

## 📖 Specification Reference

Every module in the codebase maps to a section of the **CDI Mathematical Specification v1.0**:

| Section | Title | Module |
|:---:|:---|:---|
| §1 | Cognitive Site (M, T) | `core/manifold.py` |
| §2 | Good Cover & Observation Sheaf | `core/cover.py`, `core/sheaf.py` |
| §3 | Belief Complex ℬ^• | `core/belief.py` |
| §4.1 | Clifford Algebra Cl(T*M) | `geometry/clifford.py` |
| §4.2 | Cognitive Dirac Operator | `geometry/dirac.py` |
| §4.3 | Belief Connection | `geometry/connection.py` |
| §5.1 | Belief Laplacian Δ_ℬ | `operators/laplacian.py` |
| §5.2 | Cognitive Hodge Theorem | `operators/hodge.py` |
| §5.3 | Green's Operator & Inference | `operators/green.py`, `operators/inference.py` |
| §6 | Cohomodynamic Heat Equation | `dynamics/heat_equation.py` |
| §7 | Quillen Superconnection | `field/superconnection.py` |
| §10 | Energy & Gauge Invariance | `dynamics/energy.py`, `field/gauge.py` |
| §11 | System Invariants | `topology/invariants.py` |
| §12 | Spectral Sequence Algorithm | `topology/spectral_sequence.py` |

<br/>

---

## 🤝 Contributing

CDI is an open research project. Contributions are welcome in:

- **Mathematical verification** — proving or disproving convergence results
- **Algorithmic optimization** — sparse matrix methods, GPU kernels
- **New datasets** — testing CDI on diverse domains
- **C++ port** — high-performance rewrite for production

Please open an issue to discuss before submitting PRs for major changes.

<br/>

---

## 📜 Citation

If you use CDI in your research, please cite:

```bibtex
@software{cdi2025,
  title     = {CDI: Cohomodynamic Intelligence},
  author    = {Nexus},
  year      = {2025},
  url       = {https://github.com/nexuss0781/CDI},
  note      = {A post-neural intelligence engine based on sheaf cohomology
               and spectral geometry}
}
```

<br/>

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

<br/>

---

<p align="center">
  <br/>
  <em>"The universe is not made of atoms. It is made of cohomology classes."</em>
  <br/><br/>
  <strong>CDI</strong> — Where intelligence meets topology.
  <br/><br/>
  <a href="https://github.com/nexuss0781/CDI">⭐ Star this project</a> · 
  <a href="https://github.com/nexuss0781/CDI/issues">🐛 Report Bug</a> · 
  <a href="https://github.com/nexuss0781/CDI/issues">💡 Request Feature</a>
  <br/><br/>
</p>

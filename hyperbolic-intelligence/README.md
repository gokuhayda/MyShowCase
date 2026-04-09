# Contrastive Geometric Transfer (CGT)

**Hyperbolic Sentence Embedding Compression & Language Model Distillation**

> This repository introduces the **Contrastive Geometric Transfer (CGT)** framework,
> which compresses high-dimensional Euclidean embeddings into lower-dimensional
> hyperbolic representations while preserving semantic structure.
>
> A secondary line of work (**HyDRA**) extends this framework to language model
> distillation and reveals a fundamental failure mode: **Degenerate Equilibrium (DegEq)**.

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![Status: Research Code](https://img.shields.io/badge/Status-Research%20Code-blueviolet.svg)](#6-experimental--research-status)
[![HypLoRA](https://img.shields.io/badge/HypLoRA-100%25%20Hyperbolic-8A2BE2.svg)](#hyplora-100-hyperbolic-lora-adapter)
[![DOI v2](https://img.shields.io/badge/DOI%20v2-10.5281%2Fzenodo.19480998-blue.svg)](https://doi.org/10.5281/zenodo.19480998)
[![DOI v3](https://img.shields.io/badge/DOI%20v3-10.5281%2Fzenodo.19483206-blue.svg)](https://doi.org/10.5281/zenodo.19483206)

---

## Research Philosophy

This repository represents **exploratory research**, not production-ready software. All code is provided as proof-of-concept implementations accompanying preprint publications.

**Why this exists:**

The dominant paradigm in AI scales parameters and compute, but rarely questions the geometric substrate. Scaling has been remarkably effective — but effectiveness does not imply exhaustiveness. This project asks a different question: *what if the bottleneck isn't size, but shape?*

These experiments aim to:
- **Explore alternative architectures** grounded in differential geometry and dynamical systems
- **Democratize frontier research** by making theoretical frameworks implementable and testable
- **Invite collaboration** from researchers who share curiosity about non-Euclidean approaches to intelligence

This is not a claim of superiority over existing methods. It is an invitation to investigate whether different mathematical foundations might unlock capabilities that scaling alone cannot reach.

*Contributions, critiques, and collaborations welcome.*

---

## Key Findings

### Embedding (CGT)

Hyperbolic geometry can compress high-dimensional Euclidean embeddings into lower-dimensional representations while preserving semantic structure and pairwise similarity. The CGT framework provides a student-teacher distillation pipeline with geometric fidelity guarantees on the Lorentz manifold.

### Extension: Language Model Distillation (HyDRA)

When extending this framework to autoregressive language model distillation, a failure mode emerges: **Degenerate Equilibrium (DegEq)** — a stable attractor where angular alignment stabilises while radial dynamics remain active, leading to geometrically valid but semantically degraded representations.

Four ablation experiments — standard KL, Projective KL (D1), Decoupled Radial-Angular (D3), and Origin-Tangent Euclidean Distillation (OTED) — converge to the same fixed point (rdc* ≈ 10, relative deviation <5%) by step ≈ 1,600. OTED eliminates all hyperbolic geometry from the loss backward pass yet reaches rdc* = 10.25 ± 0.30, establishing by exhaustion that DegEq originates in the model's forward dynamics via the Christoffel coupling ṙ = −sinh(r)cosh(r)·θ̇².

**What this repository shows:**
- Hyperbolic distillation converges to a stable fixed point (DegEq), empirically invariant across tested loss variants
- HypLoRA adapter successfully injects hyperbolic geometry into frozen Euclidean LLMs without DegEq
  (single exp/log round-trip per layer avoids the 4-layer Christoffel cascade)
- Geometric validity does not imply linguistic quality
- A Lyapunov-inspired potential (L_q = ½ · rdc²) characterises attractor onset
- Late intervention produces transient suppression with relaxation time τ ≈ 905 steps (R² = 0.964)
- OTED (T_o loss) reaches the same attractor as standard KL — proof that DegEq is forward-path intrinsic

**What this repository does not show:**
- That hyperbolic models outperform Euclidean ones on language modelling
- That DegEq is universal (tested only in this architecture and dataset)
- That any tested intervention eliminates DegEq (all four formulations converge to rdc* ≈ 10)

---

## 1. Project Overview

CGT (Contrastive Geometric Transfer) is a research framework exploring whether hyperbolic geometry can compress high-dimensional Euclidean sentence embeddings into lower-dimensional representations while preserving semantic structure. The core hypothesis: negatively curved spaces may offer a more natural substrate for hierarchical information than flat Euclidean space.

### The Question

Pre-trained sentence embedding models (e.g., MiniLM-384d, MPNet-768d) produce high-dimensional vectors that incur significant storage and computational costs at scale. But the deeper question is geometric: *can we exploit the exponential volume growth of negatively curved spaces to embed hierarchical semantic structures more compactly?* This project is an empirical investigation of that hypothesis.

### The Geometric Intuition

Hyperbolic space (specifically, the Lorentz/hyperboloid model) offers volume growth proportional to $e^r$ rather than the polynomial $r^n$ of Euclidean space. This property suggests that hierarchical structures — trees, taxonomies, semantic hierarchies — might embed with lower distortion in fewer dimensions. CGT tests this intuition by projecting from flat Euclidean embeddings to a curved hyperbolic manifold while attempting to preserve pairwise similarity structure.

### What This Repository Implements

- A student-teacher framework where a small hyperbolic encoder learns to preserve the similarity structure of a large Euclidean teacher model
- Lorentz manifold operations with numerical stability guarantees (exp/log maps, geodesic distances, parallel transport)
- Multi-objective training combining contrastive learning, distance distillation, and spectral alignment
- Evaluation protocols with falsification tests for geometric property validation
- Ablation studies comparing hyperbolic vs. Euclidean baselines under identical conditions
- **HypLoRA**: 100% hyperbolic LoRA adapter — injects a Lorentz manifold branch into any frozen LLM
  (`inject_hyplora`, `LorentzLowRank`, learnable curvature K per layer, `RiemannianAdamW` V5)
- Hyperbolic embedding and retrieval pipeline (experimental):
  - Euclidean teacher → hyperbolic student projection via `CGTStudentHardened`
  - Batch text encoding and corpus indexing (FAISS + Lorentz buffer)
  - Hybrid retrieval: Euclidean ANN (FAISS) candidate selection + Lorentz geodesic reranking
  - End-to-end semantic retrieval pipeline (`HyperbolicPipeline`)

### What This Repository Does NOT Implement

- Production-ready deployment code
- Formal theoretical proofs of optimality
- Complete topological invariant computation (only differentiable proxies are implemented)
- Real-time inference optimization

---

## 2. Theoretical Foundations

### 2.1 Lorentzian Hyperbolic Geometry

The geometric substrate is the Lorentz (hyperboloid) model, embedding $H^n$ as the upper sheet of a two-sheeted hyperboloid in Minkowski space $\mathbb{R}^{n,1}$ with signature $(-,+,+,\ldots,+)$.

**Implemented operations:**
- Minkowski inner product: $\langle x, y \rangle_L = -x_0 y_0 + \sum_i x_i y_i$
- Geodesic distance: $d_H(x,y) = \frac{1}{\sqrt{K}} \text{arccosh}(-K \langle x, y \rangle_L)$
- Exponential map: Projects tangent vectors to the manifold
- Logarithmic map: Projects manifold points to tangent space
- Parallel transport: Transports vectors along geodesics
- Fréchet mean: Iterative algorithm following Lou et al. (2020)

All operations include numerical stability measures (safe acosh, tangent space projection) explicitly labeled as such in the code.

> **Reference:** The Lorentz model implementation and the geometric compression methodology are formally specified in *Lorentzian Hyperbolic Compression: A Family of Geometric Models for Euclidean-to-Hyperbolic Representation Transfer* (DOI: [10.5281/zenodo.18382872](https://doi.org/10.5281/zenodo.18382872)).

> **Numerical Stability Reference:** The deterministic root causes of Lorentz manifold instabilities — catastrophic cancellation in the Minkowski inner product, Precision Boundary Rupture at dtype boundaries, and the `acosh` singularity — are rigorously analyzed in *Topological Bridges and Precision-Aware Geometry: Eradicating Deterministic Numerical Instability in Lorentz Manifold Deep Learning* (DOI: [10.5281/zenodo.19362794](https://doi.org/10.5281/zenodo.19362794)). That paper formally introduces the TB-PAG framework and proves that a floating-point residual δ ∼ 10⁻⁷ is geometrically amplified to O(10⁻³) geodesic errors via `acosh(1+δ) ≈ √(2δ)`. The `lorentz_hardened.py` module implements the hyperboloid reprojection and `acosh`-safe surrogate strategies described therein.

### 2.2 Contrastive Geometric Transfer

The core CGT method trains a student encoder to project Euclidean teacher embeddings onto the Lorentz manifold while preserving similarity structure. The loss function combines:

1. **Hyperbolic InfoNCE**: Contrastive loss using geodesic distances as similarity metric
2. **Power-law distillation**: Empirical distance scaling ($d_{target} = d_{teacher}^\alpha$) to address capacity mismatch
3. **Spectral alignment**: Graph Laplacian eigenvalue matching (first-order approximation)
4. **Topological regularization**: Differentiable proxy for Betti-0 (connectivity)

> **Reference:** The CGT methodology is documented in *Contrastive Geometric Transfer: Efficient Sentence Embeddings via Hyperbolic Projection with 24× Compression* (DOI: [10.5281/zenodo.18379741](https://doi.org/10.5281/zenodo.18379741)).

### 2.3 Hyperbolic Transformer (H-LLM)

A complete implementation of a Hyperbolic-native Transformer for language modeling, based on the Lorentz model with geodesic attention.

**Architecture:**
- **HyperbolicEmbedding**: Token embeddings stored in tangent space, projected via exp_map
- **HyperbolicLayerNorm**: Riemannian normalization (log → normalize spatial → exp)
- **HyperbolicAttention**: Geodesic distance-based attention: $s_{ij} = -d_H(q_i, k_j)^2 / \tau$
- **HyperbolicFFN**: Tangent space processing (log → linear → GELU → linear → exp)
- **Geodesic Residual**: Approximated via tangent space addition

**Training:**
- Multi-objective loss: LM + InfoNCE + manifold fidelity + radius regularization
- Teacher distillation from GPT-2 (soft targets + hidden state alignment)
- Stability monitoring: radius bounds, manifold violations

**Notebooks:**
- `hyperbolic_llm_training.ipynb`: Full training pipeline with WikiText-2
- `prototype_hyperbolic_llm_training.ipynb`: Extended prototype with chat interface
- `h_akorn_llm_training.ipynb`: H-AKORN attention variant with Kuramoto phase dynamics

> **Reference:** The Hyperbolic Transformer specification is described in *Hyperbolic-Native Large Language Model: Complete Mathematical Specification for Direct Implementation* (DOI: [10.5281/zenodo.18383897](https://doi.org/10.5281/zenodo.18383897)). The H-AKORN attention mechanism is specified in *Geometric Dynamics in Neural Architectures: The Integration of Hyperbolic Adaptive Kuramoto Oscillators into Large Language Models* (DOI: [10.5281/zenodo.18394033](https://doi.org/10.5281/zenodo.18394033)).

### 2.4 Ψ-SLM Extensions (Experimental)

The `psi_extensions/` module contains experimental implementations inspired by the Geometric Control Manifolds framework. These are prototypes, not production components.

**H-NCA (Hyperbolic Neural Cellular Automata):**
- Local state updates on the Lorentz manifold
- Four-stage cycle: perception → log_map → neural operator → exp_map
- Supports k-NN or adjacency-based neighbor selection

**H-AKOrN (Hyperbolic Artificial Kuramoto Oscillatory Neurons):**
- Phase synchronization dynamics for the binding problem
- Kuramoto coupling: $\frac{d\theta_i}{dt} = \omega_i + \sum_j K_{ij} \sin(\theta_j - \theta_i)$
- Coupling decays exponentially with hyperbolic distance

**Topological Constraint Field:**
- Persistence landscape approximation for differentiable topology
- Provides scalar feedback for "topological downward causation"
- This is a CLASSICAL APPROXIMATION, not exact TDA

> **References:**
> - *Hyperbolic Neural Cellular Automata* (DOI: [10.5281/zenodo.18334091](https://doi.org/10.5281/zenodo.18334091))
> - *Geometric Control Manifolds for Hyperbolic Self-Organizing Intelligence* (DOI: [10.5281/zenodo.18334132](https://doi.org/10.5281/zenodo.18334132))
> - *From Geometric Control Manifolds to Toy Ψ-SLM* (DOI: [10.5281/zenodo.18334140](https://doi.org/10.5281/zenodo.18334140))

### 2.5 Gromov-Wasserstein Transfer

An optional geometric alignment mechanism using entropic Gromov-Wasserstein optimal transport to compare metric spaces without requiring coordinate correspondence.

> **Reference:** *Unsupervised Topological Alignment Between Neural and Phenomenal Spaces via Gromov-Wasserstein Transport* (DOI: [10.5281/zenodo.18334153](https://doi.org/10.5281/zenodo.18334153)).

---

## 3. Code Architecture

### Directory Structure

```
src/cgt/
├── geometry/
│   ├── lorentz.py              # Lorentz manifold operations (exp/log maps, distances)
│   ├── lorentz_hardened.py     # Numerically hardened version with TB-PAG stability
│   └── frechet.py              # Fréchet mean (Lou et al. 2020)
│
├── models/
│   ├── cgt_hardened.py         # CGTStudentHardened encoder (main model)
│   ├── student.py              # Base student architecture
│   ├── transformer_v2.py       # 4L × 128d × 4H hyperbolic transformer
│   ├── lm_head_v2.py           # Intrinsic Lorentz LM head
│   ├── geodesic_lm_head.py     # GeodesicLMHeadV2, AngularLMHead (V5)
│   ├── hyplora.py              # HypLoRA: LorentzLowRank, HypLoRALayer, inject_hyplora
│   └── hyperbolic_transformer.py  # H-LLM prototype
│
├── losses/
│   ├── core.py                 # HyperbolicInfoNCE, PowerLawDistillation, TopoLoss
│   ├── losses_hardened.py      # Hardened multi-objective loss
│   ├── hyperbolic_lm_losses.py # H-LLM losses
│   ├── hybrid_active_loss.py   # Experimental active learning losses
│   └── topo_bootstrap.py       # Topological bootstrap regularization
│
├── distillation/
│   ├── distillation_v2.py      # DistillationTrainerV2, EarlyStoppingV3, DegEqController
│   ├── geometric_distillation.py  # D1 (ProjectiveKL), D3 (DecoupledRadialAngular), F1/F2/φ
│   └── hyperbolic_projector.py    # HyperbolicProjectorV3 — angular/radial gradient decoupling
│
├── dynamics/
│   ├── riemannian_adamw.py     # RiemannianAdamW + parallel transport of momentum
│   └── kuramoto_v2.py          # Kuramoto oscillator (post-convergence, negative result)
│
├── evaluation/
│   └── metrics.py              # Falsification protocols (F1-F3), STS-B evaluation
│
├── regularization/
│   ├── lipschitz.py            # Lipschitz regularization
│   └── lipschitz_analysis.py   # Lipschitz constant analysis
│
├── psi_extensions/             # Research prototypes — not central to main findings
│   ├── dynamics/h_nca.py       # Hyperbolic Neural Cellular Automata
│   ├── binding/h_akorn.py      # Hyperbolic Kuramoto oscillators
│   ├── topology/topological_field.py  # Differentiable topology proxy
│   └── transfer/gw_transfer.py # Gromov-Wasserstein alignment
│
├── utils/
│   └── helpers.py              # Utility functions
│
├── embedding/                  # Hyperbolic retrieval pipeline (experimental)
│   ├── encoder.py              # Teacher → hyperbolic projection (HyperbolicEncoder)
│   ├── index.py                # FAISS + Lorentz buffer
│   ├── retrieval.py            # FAISS candidate + Lorentz geodesic rerank
│   ├── distance.py             # Lorentz batch distance utilities (inference-optimised)
│   └── pipeline.py             # HyperbolicPipeline end-to-end
│
└── api/
    └── entrypoint.py           # SafeHyperbolicModel unified API

experiments/
├── ablations/
│   ├── euclidean_ablation.py   # CGT vs Euclidean baseline (Part IV)
│   ├── dimensional_ablation.py # Dimension crossover analysis
│   ├── mrl_comparison.py       # Matryoshka Representation Learning comparison
│   └── bq_comparison.py        # Binary quantization comparison
│
├── benchmarks/
│   ├── mteb_evaluation.py      # MTEB dataset evaluation
│   ├── latency_benchmark.py    # Inference timing
│   └── multi_model_benchmark.py # Multi-teacher benchmarks
│
└── unified/
    ├── final_executor_v2.py    # Cartesian experiment executor
    └── evaluation.py           # Unified evaluation pipeline

notebooks/
├── hyperbolic_llm_training.ipynb           # H-LLM training with GPT-2 distillation
├── prototype_hyperbolic_llm_training.ipynb # Extended H-LLM with chat interface
├── CGT_GW.ipynb                            # CGT-GW three-phase training
├── HyDRA.ipynb                             # HyDRA reproduction (Google Colab compatible)
└── final_experiment_launcher_v*.ipynb      # CGT experiment notebooks (v0-v7)
```

### Module Mapping

| Module | Implements | Theoretical Basis |
|--------|-----------|-------------------|
| `geometry/lorentz.py` | Lorentz model operations | Exact closed-form for constant negative curvature |
| `geometry/frechet.py` | Fréchet mean | Lou et al. (2020), exact iterative algorithm |
| `models/cgt_hardened.py` | CGT student encoder | MLP projector + exp_map + homeostatic field |
| `models/hyperbolic_transformer.py` | H-LLM Transformer | Geodesic attention, tangent FFN, Riemannian LayerNorm |
| `losses/core.py` | CGT multi-objective loss | InfoNCE (exact), distillation (heuristic), topo (proxy) |
| `losses/hyperbolic_lm_losses.py` | H-LLM losses | LM + manifold fidelity + radius + teacher distillation |
| `models/hyplora.py` | HypLoRA adapter | LLR on H^n, inject/extract/merge, diagnostics |
| `psi_extensions/dynamics/h_nca.py` | H-NCA | Discretized Riemannian flow (Euler) |
| `psi_extensions/binding/h_akorn.py` | H-AKOrN | Kuramoto dynamics with geodesic coupling |
| `psi_extensions/topology/topological_field.py` | Topological loss | Persistence landscape (differentiable proxy) |

### 3.1 Hyperbolic Embedding and Retrieval Pipeline

The repository includes a full inference pipeline bridging Euclidean embeddings, hyperbolic projection, and semantic retrieval (`src/cgt/embedding/`), intended for use after a `CGTStudentHardened` model has been trained.

This module is **experimental** — not optimised for production latency.

**Pipeline stages:**

1. **Text → teacher embedding** — A multilingual `SentenceTransformer` encodes texts into L2-normalised Euclidean vectors (`float32`, shape `[N, teacher_dim]`).
2. **Euclidean → hyperbolic projection** — The teacher embeddings are passed through `CGTStudentHardened`, mapping them onto the Lorentz manifold H^n (`[N, student_dim + 1]`).
3. **Corpus indexing** — Two parallel structures: a **FAISS `IndexFlatIP`** on Euclidean embeddings for fast ANN, and a **`torch.Tensor` buffer** of hyperbolic embeddings for geodesic reranking.
4. **FAISS candidate search** — The query is encoded by the teacher and submitted to FAISS for `k_candidates` approximate neighbours.
5. **Lorentz geodesic reranking** — The query is projected to H^n; geodesic distances are computed via the Lorentz inner product: `d(q, c) = (1/√K) · arccosh(−K · ⟨q, c⟩_L)`. Candidates are reordered by ascending geodesic distance.

**Key design insight:** Retrieval is decomposed into fast approximate search in Euclidean space (FAISS) and precise semantic ranking in hyperbolic space (Lorentz geodesics), avoiding prohibitive brute-force geodesic computation over the full corpus.

---

## 4. Implemented Geometry and Constraints

This project uses the **Lorentz (hyperboloid) model** exclusively.

**Numerical stability measures:**
- `safe_acosh()`: Taylor expansion near x=1 to avoid gradient explosion
- `safe_sqrt()`: Minimum clamp to prevent infinite gradients at zero
- `proj()`: Enforces the hyperboloid constraint ⟨x,x⟩ = -1/K (clamp-based, no additive ε), preventing manifold drift
- `proj_tangent()`: Ensures vectors satisfy ⟨x, v⟩_L = 0

**What is NOT implemented:**
- Exact Betti numbers (only differentiable proxies via persistence landscapes)
- OT-optimal bounds (power-law distillation is an empirical heuristic)
- Global Lipschitz bounds (local estimates only)
- Poincaré ball model

---

## 5. Relationship to the Papers

### Directly Implemented

| Concept | DOI | Module |
|---------|-----|--------|
| Lorentz manifold operations | [10.5281/zenodo.18382872](https://doi.org/10.5281/zenodo.18382872) | `geometry/lorentz.py` |
| CGT student-teacher framework | [10.5281/zenodo.18379741](https://doi.org/10.5281/zenodo.18379741) | `models/cgt_hardened.py` |
| Hyperbolic InfoNCE loss | [10.5281/zenodo.18379741](https://doi.org/10.5281/zenodo.18379741) | `losses/core.py` |
| Euclidean ablation methodology | [10.5281/zenodo.18379741](https://doi.org/10.5281/zenodo.18379741) | `experiments/ablations/euclidean_ablation.py` |
| TB-PAG numerical stability | [10.5281/zenodo.19362794](https://doi.org/10.5281/zenodo.19362794) | `geometry/lorentz_hardened.py` |
| Hyperbolic Transformer (H-LLM) | [10.5281/zenodo.18383897](https://doi.org/10.5281/zenodo.18383897) | `models/hyperbolic_transformer.py` |
| H-LLM training losses | [10.5281/zenodo.18383897](https://doi.org/10.5281/zenodo.18383897) | `losses/hyperbolic_lm_losses.py` |
| HyDRA / DegEq characterisation | [10.5281/zenodo.19480998](https://doi.org/10.5281/zenodo.19480998) | `distillation/` |
| HypLoRA adapter | Yang et al. NeurIPS 2025 (arxiv 2410.04010) | `models/hyplora.py` |

### Partially Approximated

| Concept | DOI | Approximation |
|---------|-----|---------------|
| Topological regularization | [10.5281/zenodo.18334132](https://doi.org/10.5281/zenodo.18334132) | Persistence landscape proxy (not exact Betti) |
| H-NCA dynamics | [10.5281/zenodo.18334091](https://doi.org/10.5281/zenodo.18334091) | Euler discretization (not continuous flow) |
| H-AKOrN binding | [10.5281/zenodo.18334091](https://doi.org/10.5281/zenodo.18334091) | RK2 integration of Kuramoto equations |
| GW alignment | [10.5281/zenodo.18334153](https://doi.org/10.5281/zenodo.18334153) | Entropic regularization via POT library |

### Theoretical Only (Not Implemented)

| Concept | DOI | Status |
|---------|-----|--------|
| Quantum topological estimation | [10.5281/zenodo.18334098](https://doi.org/10.5281/zenodo.18334098) | Not implemented (classical approximation only) |
| Continuous Riemannian flow | [10.5281/zenodo.18334140](https://doi.org/10.5281/zenodo.18334140) | Discretized as Euler updates |
| Anosov flow stability | [10.5281/zenodo.18334123](https://doi.org/10.5281/zenodo.18334123) | Not implemented |
| Lorentz-Manifold Transformers | [10.5281/zenodo.18334083](https://doi.org/10.5281/zenodo.18334083) | Prototype only in notebooks |

---

## 6. Experimental / Research Status

This is **research code** intended for reproducing experimental results, ablation studies, and academic reference. It is **not** production-ready software, optimized for deployment, or validated beyond the reported experiments.

**Single-seed results:** All experiments use SEED=42. Multi-seed validation is reserved for future work. The Euclidean baseline comparison is estimated rather than directly run at identical parameter count and steps.

**Attractor characterisation scope:** The rdc* ≈ 10, τ ≈ 905 steps results are empirical and restricted to the tested architecture (4L × 128d × 4H) and dataset (WikiText-2). Generalisation to other hyperbolic models or training objectives requires further investigation.

### Limitations

1. **Topological claims**: Differentiable proxies only. No claims about exact homology preservation.
2. **Scalability**: Experiments use sentence-level embeddings (hundreds to thousands of samples). Behavior at larger scales is not validated.
3. **Hyperparameter sensitivity**: Loss weights were tuned empirically; different datasets may require re-tuning.
4. **Numerical precision**: The code uses float64 for geometric operations. Float32 may introduce manifold violations.
5. **Embedding and retrieval pipeline**: Experimental, not optimised for production latency.
6. **FAISS scalability**: Not validated beyond medium-scale datasets; no IVF/HNSW tuning applied.
7. **Hybrid retrieval heuristic**: The Euclidean + hyperbolic decomposition is a heuristic, not theoretically optimal.

### Falsification Protocols

The code includes three falsification tests (not proofs):
- **F1 (Homotopy)**: Tests β₀ stability under input perturbation
- **F2 (Stability)**: Measures encoder perturbation amplification
- **F3 (Forman-Ricci)**: Checks discrete curvature on k-NN graph

These tests can **disprove** claims but cannot **prove** geometric properties hold globally.

---

## 7. Installation and Usage

### Installation

```bash
pip install -e .
```

### Basic Usage — CGT Sentence Embedding Compression

```python
from cgt.models.cgt_hardened import CGTStudentHardened, StudentConfig
from cgt.geometry.lorentz_hardened import LorentzSubstrateHardened, LorentzConfig
import torch

# Initialize geometry
substrate = LorentzSubstrateHardened(LorentzConfig(curvature=-1.0))

# Initialize model
model = CGTStudentHardened(
    config=StudentConfig(input_dim=384, hidden_dim=256, output_dim=32),
    lorentz=substrate,
)

# Forward pass
teacher_embeddings = torch.randn(32, 384, dtype=torch.float64)
hyperbolic_embeddings = model(teacher_embeddings)  # [32, 33] on H^32

# Compute similarity using Lorentz distance
dist = substrate.dist(hyperbolic_embeddings[:16], hyperbolic_embeddings[16:])
similarity = -dist  # Negative distance = similarity
```

### Running Experiments

```python
from experiments.unified.final_executor_v2 import run_cartesian_execution
from pathlib import Path

summary = run_cartesian_execution(
    output_base=Path("./outputs"),
    scope="canonical",  # "minimal", "canonical", or "full_cartesian"
    seed=42,
)
```

### Hyperbolic Embedding and Retrieval (Experimental)

```python
from sentence_transformers import SentenceTransformer
from cgt.models.cgt_hardened import CGTStudentHardened
from cgt.embedding.pipeline import HyperbolicPipeline

teacher = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
student = CGTStudentHardened(teacher_dim=384, student_dim=32, hidden_dim=256)
# student.load_state_dict(torch.load("checkpoint.pt")["model_state_dict"])

pipeline = HyperbolicPipeline(teacher, student)
pipeline.index_corpus(texts)

results = pipeline.query("What is hyperbolic embedding?")
for ev in results:
    print(ev.rank, ev.score, ev.text[:80])
```

---

## 10. HyDRA: Hyperbolic Distillation with Riemannian Adaptation

> **Paper:** HyDRA v3 — Hyperbolic Distillation with Riemannian Adaptation: Attractor Invariance and Proof by Elimination  
> **DOI (v2):** [10.5281/zenodo.19480998](https://doi.org/10.5281/zenodo.19480998)  
> **DOI (v3):** [10.5281/zenodo.19483206](https://doi.org/10.5281/zenodo.19483206)  
> **SHA-256:** `9c844919d58f29cb7bb19b90154131996b5e3b2d6458c57834605068d3e8685e`

HyDRA extends the CGT framework to **language model distillation**, training a compact hyperbolic student (≈12M parameters, H^128) via knowledge distillation from GPT-2-small (117M) on WikiText-2. Every layer — attention, feed-forward, and residual connections — preserves the Riemannian manifold constraint to float64 precision. The primary contribution is the identification, formal characterization, and mitigation of **Degenerate Equilibrium (DegEq)**: a stable fixed point where angular convergence completes but radial scale grows monotonically, converging to a geometrically stable but semantically degraded fixed point.

### New Modules (v2)

| Module | Description |
|--------|-------------|
| `geometry/lorentz_v2.py` | Isolated Lorentz substrate with TB-PAG precision policy, `safe_acosh_v2`, L2-norm clamped `exp_map_zero` |
| `models/transformer_v2.py` | Compact hyperbolic transformer (4L × 128d × 4H) with geodesic residuals |
| `models/lm_head_v2.py` | **Intrinsic Lorentz LM head** — Minkowski inner product scoring, replaces broken `log_map_zero` approach |
| `models/layer_v2.py` | Riemannian Layer Normalization with `r_max = 1.5` tangent clamp |
| `models/hakorn_attention_v2.py` | HaKOrN attention variant (experimental) |
| `distillation/distillation_v2.py` | Full distillation trainer: loss, EarlyStoppingV3, LossBalancer, AdaptiveTuner, DegEqController |
| `distillation/geometric_distillation.py` | D1 (ProjectiveKL), D3 (DecoupledRadialAngular), F1/F2/φ metrics |
| `distillation/hyperbolic_projector.py` | HyperbolicProjectorV3 — angular/radial gradient decoupling |
| `distillation/geometric_distillation.py` | **OTEDLoss** — all loss in T_o H^n (Christoffel-free backward) |
| `dynamics/riemannian_adamw.py` | RiemannianAdamW + parallel transport of momentum |
| `dynamics/kuramoto_v2.py` | Kuramoto oscillator system (`dθ/dt = ω + K/N Σ sin(θⱼ − θᵢ)`) |
| `api/entrypoint.py` | `SafeHyperbolicModel` + `SafeModelConfig` unified API |

### Key Innovations

**1. Degenerate Equilibrium (DegEq) Characterization**

A stable fixed point where angular convergence is complete but radial scale grows monotonically. Formally defined by three simultaneous conditions:
- `L_hidden < 0.25` (angular learning complete)
- `rdc_ema > 15` (radial drift dominant)
- `σ_logit > 2.0` (logit overconfidence)

**2. Radial Drift Coefficient (RDC)**

Real-time proxy for DegEq onset, computed in-training at every step:
```
RDC(t) = σ_logit(t) / (L_hidden(t) + 0.01)
rdc_ema(t) = 0.95 · rdc_ema(t−1) + 0.05 · RDC(t)
```
Reliably predicts DegEq onset 500–1,000 steps in advance.

**3. Intrinsic Lorentz LM Head**

Replaces the manifold-folding `log_map_zero + spatial slice` scoring with Minkowski inner product in `float64`:
```
score(h, wₖ) = ⟨h, wₖ⟩_L = −h₀w₀ + Σᵢ hᵢwᵢ
logit_k = clamp(τ, 0.01, 2.5) · ⟨h, wₖ⟩_L
```

**4. Riemannian Natural Gradient Correction**

`r/sinh(r)` scaling of Euclidean Adam gradients on manifold parameters, derived from the Lorentz metric tensor pullback (Amari 1998):
```python
r = ‖θ_{1:n}‖₂.clamp(ε)
grad *= r / sinh(r.clamp(max=10))
```

**5. Symmetric Hyperbolic Vocabulary (Variant E/F)**

Stores vocabulary as tangent vectors `[V, n]` and lifts to manifold via `exp_map_zero` with `vocab_r_max=3.0`. Prevents unlimited radial growth without constraining semantic expressiveness.

**6. EarlyStoppingV3 — Dual-EMA**

Dual-reference EMA (fast β=0.3, slow β=0.9) with sliding window local maximum and detrended noise estimation. Correctly accumulates patience on true plateaus where single-reference EMA perpetually fires false positives.

**7. Adaptive Training Infrastructure**

Two-layer control system:
- `LossBalancer` (proactive): equalizes effective gradient contributions across all loss terms at every evaluation
- `AdaptiveTuner` (reactive): fires on regime alerts, adjusts temperature and loss weights

### Experimental Results

| Variant | Description | Steps | Best PPL | DegEq onset |
|---------|-------------|-------|----------|-------------|
| A | Baseline — no intervention | 10k | 482.8 | step ≈ 5,400 |
| D | Hyperbolic vocab via `proj()` | 10k | 401.0 | step ≈ 5,400 |
| E | Symmetric vocab (`exp_map_zero`) | 13k | 377.6 | > 13k |
| **F** | **Full Riemannian (HyDRA)** | **33.4k** | **291.3** | **Delayed (> 33.4k steps)** |

**Note on Variant F:** The Riemannian natural gradient correction delays DegEq onset beyond the observed training window (> 33,400 steps). It does not eliminate the attractor — it delays entry into this regime.

**Negative result:** Kuramoto phase coupling applied post-convergence consistently degrades perplexity (+175%, PPL 291 → 850) across all tested coupling strengths, consistent with hyperbolic representations being geometrically locked against post-hoc tangent-space perturbations.

### DegEq Attractor (Cross-Variant)

| Run | rdc* (mean) | rdc* (σ) | Step rdc > 9 |
|-----|-------------|----------|--------------|
| Variant F (standard KL) | 9.82 | 0.17 | ≈ 1,600 |
| D1 (Projective KL) | 10.04 | 0.12 | ≈ 1,600 |
| D3 (Decoupled R-A) | 10.04 | 0.12 | ≈ 1,600 |
| **OTED (T₀ loss)** | **10.25** | **0.30** | **≈ 1,600** |

**Proof by elimination:** OTED moves all loss computation to the flat tangent space
T_o H^n, eliminating Christoffel symbols from the backward pass. It reaches rdc* = 10.25 ± 0.30 —
indistinguishable from the KL baseline. Four interventions, same fixed point: DegEq originates
in the model's forward dynamics, not in the loss surface.

### Usage

```python
from cgt.api.entrypoint import SafeHyperbolicModel, SafeModelConfig
from cgt.distillation.distillation_v2 import DistillationTrainerV2, DistillationConfigV2

# Build student (≈12M parameters, H^128)
student_cfg = SafeModelConfig(
    vocab_size=50257, n_embd=128, n_layer=4, n_head=4, n_positions=128,
    riemannian_correct_vocab=True,
    riemannian_correct_embed=True,
    riemannian_correct_encoder=True,
)
student = SafeHyperbolicModel(student_cfg)

# Configure distillation
dist_cfg = DistillationConfigV2(
    alpha=0.25, temperature=1.2,
    lambda_hidden=0.15, lambda_radius=0.05, lambda_contrast=0.10,
    max_steps=100000, learning_rate=3e-4,
    riemannian_correct_vocab=True,
    riemannian_correct_embed=True,
    riemannian_correct_encoder=True,
)

# Train
trainer = DistillationTrainerV2(student, teacher, dist_cfg, ckpt_dir)
trainer.train(train_loader, val_loader)
```

**Reproduction notebook:** `notebooks/HyDRA.ipynb` — Google Colab compatible.

---


---

## 11. HypLoRA: 100% Hyperbolic LoRA Adapter

> **Architecture:** CGT-native implementation of [HypLoRA (Yang et al., NeurIPS 2025)](https://arxiv.org/abs/2410.04010)  
> **Module:** `src/cgt/models/hyplora.py`

HypLoRA injects a fully hyperbolic adapter branch into any frozen LLM. Unlike naive tangent-space LoRA,
it operates **directly on Lorentz manifold coordinates**, preserving the hierarchical radial structure
that empirically characterises LLM token embeddings (frequent tokens near origin, rare tokens far).

**Why no DegEq:** Each `HypLoRALayer` performs exactly ONE `exp_map` and ONE `log_map` per forward pass.
The frozen backbone accumulates no manifold momentum — only adapter parameters `A`, `B`, `K` are updated.
This avoids the 4-layer Christoffel cascade that causes DegEq in fully-hyperbolic models trained from scratch.

### Architecture

```
z = W·x  +  Π_log( LLR( B·A,  Π_exp(x) ) )  ×  (alpha / rank)
↑                   ↑          ↑
frozen          Lorentz     project to
Euclidean       Low-Rank    H^n via
weight          transform   exp_map_zero
```

Where `LLR` (Lorentz Low-Rank) applies the low-rank transform directly on manifold coordinates:

1. `log_map_zero(x_H)` → tangent vector `v ∈ T_o H^n ≅ R^n`
2. `v → v @ A @ B` (rank-r bottleneck in tangent space)
3. `exp_map_zero(v_adapted)` → adapted point on H^n

Curvature `K` is a learnable parameter per layer (log-parameterised for positivity `K > 0`).

### New Modules (HypLoRA)

| Class / Function | Description |
|---|---|
| `LorentzLowRank` | Core adapter: rank-r transform directly on H^n with learnable K |
| `HypLoRALayer` | Wraps any `nn.Linear` with frozen base + hyperbolic branch |
| `HypLoRAConfig` | Dataclass: rank, alpha, n_embd, curvature, dropout, dtype |
| `inject_hyplora(model, config, target_modules)` | Replaces target layers in any `nn.Module` |
| `extract_hyplora(model)` | Saves only adapter weights (independent of backbone) |
| `load_hyplora(model, state)` | Loads extracted adapter weights |
| `merge_hyplora(model)` | Fuses adapter into backbone for inference (first-order approx) |
| `delta_hyperbolicity(embeddings)` | Measures δ-hyperbolicity via four-point condition |
| `token_freq_norm_stats(embeddings, freqs)` | Token frequency vs embedding norm correlation |
| `print_trainable_params(model)` | Trainable / frozen parameter summary |

### Usage — Fine-tuning any LLM

```python
from cgt.models.hyplora import HypLoRAConfig, inject_hyplora
from cgt.dynamics.riemannian_adamw import RiemannianAdamW

# 1. Configure adapter (set n_embd to the model's hidden dimension)
config = HypLoRAConfig(
    rank            = 8,
    alpha           = 16.0,
    n_embd          = 4096,   # LLaMA-3-8B; use 768 for GPT-2, 2048 for LLaMA-3-1B
    curvature       = 1.0,
    learn_curvature = True,   # K is learnable per layer
)

# 2. Inject hyperbolic adapters (base weights frozen automatically)
inject_hyplora(model, config, target_modules=["q_proj", "v_proj"])
# Common choices:
#   ["q_proj", "v_proj"]                          — HypLoRA default (attention)
#   ["q_proj", "k_proj", "v_proj", "o_proj"]      — full attention
#   ["gate_proj", "up_proj", "down_proj"]          — FFN only

# 3. Riemannian optimizer with V5 radial momentum fix
optimizer = RiemannianAdamW(
    list(model.named_parameters()),
    substrate                 = config.get_substrate(),
    lr                        = 3e-4,
    weight_decay              = 0.01,
    radial_momentum_projection = True,   # V5: breaks Christoffel-momentum coupling
)

# 4. Train (standard loop — no changes needed)
for batch in train_loader:
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### Diagnostics

```python
from cgt.models.hyplora import delta_hyperbolicity, token_freq_norm_stats

# Measure hyperbolic structure of token embeddings
embeddings = model.get_input_embeddings().weight.detach()

dh = delta_hyperbolicity(embeddings, n_samples=500)
print(f"δ-hyperbolicity: {dh:.4f}")
# < 0.1 → strong tree-like structure (supports HypLoRA)
# > 0.5 → weak hyperbolic structure

stats = token_freq_norm_stats(embeddings, token_frequencies)
print(f"Freq-norm correlation: {stats['freq_norm_corr']:.4f}")
# Negative → frequent tokens closer to origin (HypLoRA paper finding)
```

### Saving and Loading Adapters

```python
from cgt.models.hyplora import extract_hyplora, load_hyplora
import torch

# Save only adapter weights (~MB, not GB)
adapter_state = extract_hyplora(model)
torch.save(adapter_state, "hyplora_adapter.pt")

# Load into a fresh model with injected adapters
adapter_state = torch.load("hyplora_adapter.pt")
load_hyplora(model, adapter_state)
```

### Relationship to HyDRA

| | HyDRA | HypLoRA |
|---|---|---|
| **Training** | From scratch | Fine-tuning |
| **Architecture** | All 4 layers on H^n | Frozen Euclidean + 1 hyp branch |
| **DegEq risk** | High (Christoffel cascade) | None (single exp/log) |
| **Geometric coverage** | Full (all internal reps) | Partial (adapter only) |
| **DegEq lesson** | Identifies the mechanism | Architecturally avoids it |

HyDRA discovered *why* DegEq occurs; HypLoRA shows *how to avoid it architecturally*.
Together they establish both the failure mode and the design principle for hyperbolic LLMs.


## 8. References

1. **HyDRA v3: Hyperbolic Distillation with Riemannian Adaptation — Attractor Invariance and Proof by Elimination**  
   de Sena, Éric Gustavo Reis. (2026).
   - v2: DOI [10.5281/zenodo.19480998](https://doi.org/10.5281/zenodo.19480998)
   - **v3 (current):** DOI [10.5281/zenodo.19483206](https://doi.org/10.5281/zenodo.19483206)

2. **Lorentzian Hyperbolic Compression: A Family of Geometric Models for Euclidean-to-Hyperbolic Representation Transfer**  
   Reis, Éric. Zenodo (2026). DOI: [10.5281/zenodo.18382872](https://doi.org/10.5281/zenodo.18382872)

3. **Contrastive Geometric Transfer: Efficient Sentence Embeddings via Hyperbolic Projection with 24× Compression**  
   Reis, Éric. Zenodo (2026). DOI: [10.5281/zenodo.18379741](https://doi.org/10.5281/zenodo.18379741)

4. **Topological Bridges and Precision-Aware Geometry: Eradicating Deterministic Numerical Instability in Lorentz Manifold Deep Learning**  
   Reis, Éric. Zenodo (2026). DOI: [10.5281/zenodo.19362794](https://doi.org/10.5281/zenodo.19362794)

5. **Hyperbolic-Native Large Language Model: Complete Mathematical Specification for Direct Implementation**  
   Reis, Éric. Zenodo (2026). DOI: [10.5281/zenodo.18383897](https://doi.org/10.5281/zenodo.18383897)

6. **Geometric Dynamics in Neural Architectures: The Integration of Hyperbolic Adaptive Kuramoto Oscillators into Large Language Models**  
   Reis, Éric. Zenodo (2026). DOI: [10.5281/zenodo.18394033](https://doi.org/10.5281/zenodo.18394033)

7. **Geometric Control Manifolds for Hyperbolic Self-Organizing Intelligence**  
   Reis, Éric. Zenodo (2026). DOI: [10.5281/zenodo.18334132](https://doi.org/10.5281/zenodo.18334132)

8. **From Geometric Control Manifolds to Toy Ψ-SLM: A Classical, Variational Implementation on Hyperbolic Manifolds**  
   Reis, Éric. Zenodo (2026). DOI: [10.5281/zenodo.18334140](https://doi.org/10.5281/zenodo.18334140)

9. **Hyperbolic Neural Cellular Automata: A Geometric Framework for Emergent Complexity**  
   Reis, Éric. Zenodo (2026). DOI: [10.5281/zenodo.18334091](https://doi.org/10.5281/zenodo.18334091)

10. **Unsupervised Topological Alignment Between Neural and Phenomenal Spaces via Gromov-Wasserstein Transport**  
    Reis, Éric. Zenodo (2026). DOI: [10.5281/zenodo.18334153](https://doi.org/10.5281/zenodo.18334153)

11. **Dimensional Efficiency Bounds for Embedding Hierarchical Metric Structures: A Regime-Dependent Analysis**  
    Reis, Éric. Zenodo (2026). DOI: [10.5281/zenodo.18378938](https://doi.org/10.5281/zenodo.18378938)

Additional foundational references:
- Yang et al. (2025). "HypLoRA: Hyperbolic Fine-Tuning for Large Language Models." NeurIPS 2025 Spotlight.
  https://arxiv.org/abs/2410.04010
- Lou et al. (2020). "Differentiating through the Fréchet Mean." ICML.
- Nickel & Kiela (2017). "Poincaré Embeddings for Learning Hierarchical Representations."
- Ganea et al. (2018). "Hyperbolic Neural Networks."
- Amari (1998). "Natural gradient works efficiently in learning." Neural Computation.

---

## 9. Related Publications by the Author

The following papers provide broader theoretical context. **These are NOT fully implemented in this repository.**

### Numerical Foundations

| Paper | DOI | Status |
|-------|-----|--------|
| Topological Bridges and Precision-Aware Geometry | [10.5281/zenodo.19362794](https://doi.org/10.5281/zenodo.19362794) | Implemented (`lorentz_hardened.py`) |

### Architectural Extensions

| Paper | DOI | Status |
|-------|-----|--------|
| Lorentz-Manifold Transformers: A Geometric–Dynamical Framework | [10.5281/zenodo.18334083](https://doi.org/10.5281/zenodo.18334083) | Theoretical |
| H-KAN: Hyperbolic Kuramoto Attention Networks | [10.5281/zenodo.18334106](https://doi.org/10.5281/zenodo.18334106) | Theoretical (Poincaré disk variant) |
| The Ψ-Former: Topological Downward Causation via Riemannian Optimization | [10.5281/zenodo.18334069](https://doi.org/10.5281/zenodo.18334069) | Theoretical |
| Resilient Cognitive Architectures via Anosov Flows on Lorentz Manifolds | [10.5281/zenodo.18334123](https://doi.org/10.5281/zenodo.18334123) | Theoretical |

### Theoretical Framework

| Paper | DOI | Status |
|-------|-----|--------|
| The Geometric Control Manifold Hypothesis | [10.5281/zenodo.18334059](https://doi.org/10.5281/zenodo.18334059) | Foundational hypothesis |
| A Unified Geometric Field Theory of Self-Organizing Intelligence | [10.5281/zenodo.18334098](https://doi.org/10.5281/zenodo.18334098) | UGFT framework |
| The Topological Signature of Consciousness: A GW Framework | [10.5281/zenodo.18334076](https://doi.org/10.5281/zenodo.18334076) | GW for neural-phenomenal alignment |
| Hyperbolic Semantic Communication: Geometrodynamics of Synthetic Intelligence | [10.5281/zenodo.18379242](https://doi.org/10.5281/zenodo.18379242) | Unified framework under causal constraints |

### Application Domains (XR / Semantic Communication)

| Paper | DOI | Status |
|-------|-----|--------|
| Hyperbolic Semantic Communication for Retinal-Resolution XR | [10.5281/zenodo.18380913](https://doi.org/10.5281/zenodo.18380913) | Theoretical framework |
| Geometric Dynamics and Semantic Communication for XR Video (v1) | [10.5281/zenodo.18383221](https://doi.org/10.5281/zenodo.18383221) | Systems-level analysis |
| Geometric Dynamics and Semantic Communication for XR Video (v2) | [10.5281/zenodo.18383250](https://doi.org/10.5281/zenodo.18383250) | Extended version |

**Note:** Claims made in theoretical papers should not be interpreted as empirically validated by this codebase.

---

## Citation

```bibtex
@misc{sena2026hydra,
  title={HyDRA v3: Hyperbolic Distillation with Riemannian Adaptation ---
         Attractor Invariance and Proof by Elimination},
  author={de Sena, {\'E}ric Gustavo Reis},
  year={2026},
  doi={10.5281/zenodo.19483206},
  note={v2: 10.5281/zenodo.19480998}
}

@misc{sena2026cgt,
  title={Contrastive Geometric Transfer: Efficient Sentence Embeddings via Hyperbolic Projection with 24× Compression},
  author={de Sena, {\'E}ric Gustavo Reis},
  year={2026},
  publisher={Zenodo},
  doi={10.5281/zenodo.18379741}
}
```

```bibtex
@inproceedings{yang2025hyplora,
  title     = {Hyperbolic Fine-Tuning for Large Language Models},
  author    = {Yang, Menglin and B B, Ram Samarth and Feng, Aosong
               and Xiong, Bo and Liu, Jiahong and King, Irwin and Ying, Rex},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025}
}
```

---

## License

[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)  
Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.

**IP Protected** — Patent Pending. Commercial licensing: eirikreisena@gmail.com

### AI Training Prohibition

**NOTICE TO AI/ML COMPANIES AND RESEARCHERS:**

This repository and all its contents are **explicitly excluded** from use in training, fine-tuning, or otherwise improving any artificial intelligence, machine learning, or large language model systems, including but not limited to:

- Foundation models (GPT, Claude, Gemini, LLaMA, etc.)
- Embedding models, code generation models, and any derivative or successor systems

**Prohibited uses include:** direct training on this codebase or documentation; inclusion in training datasets (Common Crawl, The Stack, etc.); use in RLHF, DPO, or other alignment procedures; extraction of patterns, architectures, or methodologies for model improvement.

This prohibition applies to all entities — AI companies, academic institutions, and any entity collecting data for AI training purposes. Violation constitutes copyright infringement. The CC BY-NC-SA 4.0 license does not grant permission for AI training use.

For authorized research collaborations, contact the author directly.

---

**Author:** Éric Gustavo Reis de Sena  
**Contact:** eirikreisena@gmail.com  
**LinkedIn:** [linkedin.com/in/eric-araya](https://linkedin.com/in/eric-araya)  
**YouTube:** [youtube.com/@ericreis-z3u](https://youtube.com/@ericreis-z3u)

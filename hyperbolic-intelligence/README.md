# Contrastive Geometric Transfer (CGT)

**Research Code for Hyperbolic Sentence Embedding Compression**

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![Status: Research Code](https://img.shields.io/badge/Status-Research%20Code-blueviolet.svg)](#research-philosophy)

---

## Research Philosophy

This repository represents **exploratory research**, not production-ready software. All code is provided as proof-of-concept implementations accompanying preprint publications.

**Why this exists:**

The dominant paradigm in AI scales parameters and compute, but rarely questions the geometric substrate. Scaling has been remarkably effective—but effectiveness does not imply exhaustiveness. This project asks a different question: *what if the bottleneck isn't size, but shape?*

These experiments aim to:
- **Explore alternative architectures** grounded in differential geometry and dynamical systems
- **Democratize frontier research** by making theoretical frameworks implementable and testable
- **Invite collaboration** from researchers who share curiosity about non-Euclidean approaches to intelligence

This is not a claim of superiority over existing methods. It is an invitation to investigate whether different mathematical foundations might unlock capabilities that scaling alone cannot reach.

*Contributions, critiques, and collaborations welcome.*

---

## 1. Project Overview

CGT (Contrastive Geometric Transfer) is a research framework exploring whether hyperbolic geometry can compress high-dimensional Euclidean sentence embeddings into lower-dimensional representations while preserving semantic structure. The core hypothesis: negatively curved spaces may offer a more natural substrate for hierarchical information than flat Euclidean space.

### The Question

Pre-trained sentence embedding models (e.g., MiniLM-384d, MPNet-768d) produce high-dimensional vectors that incur significant storage and computational costs at scale. But the deeper question is geometric: *can we exploit the exponential volume growth of negatively curved spaces to embed hierarchical semantic structures more compactly?* This project is an empirical investigation of that hypothesis.

### The Geometric Intuition

Hyperbolic space (specifically, the Lorentz/hyperboloid model) offers volume growth proportional to $e^r$ rather than the polynomial $r^n$ of Euclidean space. This property suggests that hierarchical structures—trees, taxonomies, semantic hierarchies—might embed with lower distortion in fewer dimensions. CGT tests this intuition by projecting from flat Euclidean embeddings to a curved hyperbolic manifold while attempting to preserve pairwise similarity structure.

### What This Repository Implements

- A student-teacher framework where a small hyperbolic encoder learns to preserve the similarity structure of a large Euclidean teacher model
- Lorentz manifold operations with numerical stability guarantees (exp/log maps, geodesic distances, parallel transport)
- Multi-objective training combining contrastive learning, distance distillation, and spectral alignment
- Evaluation protocols with falsification tests for geometric property validation
- Ablation studies comparing hyperbolic vs. Euclidean baselines under identical conditions

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

### 2.2 Contrastive Geometric Transfer

The core CGT method trains a student encoder to project Euclidean teacher embeddings onto the Lorentz manifold while preserving similarity structure. The loss function combines:

1. **Hyperbolic InfoNCE**: Contrastive loss using geodesic distances as similarity metric
2. **Power-law distillation**: Empirical distance scaling ($d_{target} = d_{teacher}^\alpha$) to address capacity mismatch
3. **Spectral alignment**: Graph Laplacian eigenvalue matching (first-order approximation)
4. **Topological regularization**: Differentiable proxy for Betti-0 (connectivity)

> **Reference:** The CGT methodology and experimental framework are documented in *Contrastive Geometric Transfer: Efficient Sentence Embeddings via Hyperbolic Projection with 24× Compression* (DOI: [10.5281/zenodo.18379741](https://doi.org/10.5281/zenodo.18379741)).

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

> **Reference:** The Hyperbolic Transformer specification is described in *Hyperbolic-Native Large Language Model: Complete Mathematical Specification for Direct Implementation* (DOI: [10.5281/zenodo.18383897](https://doi.org/10.5281/zenodo.18383897)). The H-AKORN attention mechanism integrating Kuramoto oscillatory dynamics is specified in *Geometric Dynamics in Neural Architectures: The Integration of Hyperbolic Adaptive Kuramoto Oscillators into Large Language Models* (DOI: [10.5281/zenodo.18394033](https://doi.org/10.5281/zenodo.18394033)).

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

> **Reference:** The H-NCA, H-AKOrN, and topological field constructs are described in:
> - *Hyperbolic Neural Cellular Automata: A Geometric Framework for Emergent Complexity* (DOI: [10.5281/zenodo.18334091](https://doi.org/10.5281/zenodo.18334091))
> - *Geometric Control Manifolds for Hyperbolic Self-Organizing Intelligence* (DOI: [10.5281/zenodo.18334132](https://doi.org/10.5281/zenodo.18334132))
> - *From Geometric Control Manifolds to Toy Ψ-SLM: A Classical, Variational Implementation on Hyperbolic Manifolds* (DOI: [10.5281/zenodo.18334140](https://doi.org/10.5281/zenodo.18334140))

### 2.4 Gromov-Wasserstein Transfer

An optional geometric alignment mechanism using entropic Gromov-Wasserstein optimal transport to compare metric spaces without requiring coordinate correspondence.

> **Reference:** The GW framework is detailed in *Unsupervised Topological Alignment Between Neural and Phenomenal Spaces via Gromov-Wasserstein Transport* (DOI: [10.5281/zenodo.18334153](https://doi.org/10.5281/zenodo.18334153)).

---

## 3. Code Architecture

### Directory Structure

```
src/cgt/
├── geometry/
│   ├── lorentz.py              # Lorentz manifold operations (exp/log maps, distances)
│   ├── lorentz_hardened.py     # Numerically hardened version with extra stability
│   └── frechet.py              # Fréchet mean (Lou et al. 2020)
│
├── models/
│   ├── cgt_hardened.py         # CGTStudentHardened encoder (main model)
│   ├── student.py              # Base student architecture
│   ├── hyperbolic_transformer.py  # H-LLM: Complete Hyperbolic Transformer
│   └── cgt_gw.py               # CGT with Gromov-Wasserstein loss
│
├── losses/
│   ├── core.py                 # HyperbolicInfoNCE, PowerLawDistillation, TopoLoss
│   ├── losses_hardened.py      # Hardened multi-objective loss
│   ├── hyperbolic_lm_losses.py # H-LLM losses: LM + distillation + manifold fidelity
│   ├── hybrid_active_loss.py   # Experimental active learning losses
│   └── topo_bootstrap.py       # Topological bootstrap regularization
│
├── evaluation/
│   └── metrics.py              # Falsification protocols (F1-F3), STS-B evaluation
│
├── regularization/
│   ├── lipschitz.py            # Lipschitz regularization
│   └── lipschitz_analysis.py   # Lipschitz constant analysis
│
├── psi_extensions/             # Experimental Ψ-SLM components
│   ├── dynamics/h_nca.py       # Hyperbolic Neural Cellular Automata
│   ├── binding/h_akorn.py      # Hyperbolic Kuramoto oscillators
│   ├── topology/topological_field.py  # Differentiable topology proxy
│   └── transfer/gw_transfer.py # Gromov-Wasserstein alignment
│
└── utils/
    └── helpers.py              # Utility functions

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
| `psi_extensions/dynamics/h_nca.py` | H-NCA | Discretized Riemannian flow (Euler) |
| `psi_extensions/binding/h_akorn.py` | H-AKOrN | Kuramoto dynamics with geodesic coupling |
| `psi_extensions/topology/topological_field.py` | Topological loss | Persistence landscape (differentiable proxy) |

---

## 4. Implemented Geometry and Constraints

### Hyperbolic Model

This project uses the **Lorentz (hyperboloid) model** exclusively:
- Points satisfy $\langle x, x \rangle_L = -1/K$ where $K > 0$ is the curvature parameter
- Sectional curvature is $-1/K$
- The Poincaré ball model is NOT implemented

### Numerical Guarantees

The implementation includes explicit stability measures:
- `safe_acosh()`: Taylor expansion near $x=1$ to avoid gradient explosion
- `safe_sqrt()`: Minimum clamp to prevent infinite gradients at zero
- `proj()`: "Paranoid projection" to ensure points remain on manifold after numerical operations
- `proj_tangent()`: Ensures vectors satisfy $\langle x, v \rangle_L = 0$

These are numerical corrections, not theoretical modifications. The code explicitly labels them as such.

### What Is NOT Implemented

- **Exact Betti numbers**: Only differentiable proxies (persistence landscapes) are computed. The code does not compute global topological invariants.
- **Optimal transport proofs**: The power-law distillation is an empirical heuristic, not OT-optimal.
- **Global Lipschitz bounds**: Only local estimates via linear regression.
- **Poincaré model**: All operations are in the Lorentz model.

---

## 5. Relationship to the Papers

### Directly Implemented

| Concept | Paper | Implementation |
|---------|-------|----------------|
| Lorentz manifold operations | DOI: 10.5281/zenodo.18382872 | `geometry/lorentz.py` |
| CGT student-teacher framework | DOI: 10.5281/zenodo.18379741 | `models/cgt_hardened.py` |
| Hyperbolic InfoNCE loss | DOI: 10.5281/zenodo.18379741 | `losses/core.py` |
| Euclidean ablation methodology | DOI: 10.5281/zenodo.18379741 | `experiments/ablations/euclidean_ablation.py` |
| Hyperbolic Transformer (H-LLM) | DOI: 10.5281/zenodo.18383897 | `models/hyperbolic_transformer.py` |
| H-LLM training losses | DOI: 10.5281/zenodo.18383897 | `losses/hyperbolic_lm_losses.py` |
| GPT-2 distillation for H-LLM | DOI: 10.5281/zenodo.18383897 | `notebooks/hyperbolic_llm_training.ipynb` |

### Partially Approximated

| Concept | Paper | Approximation |
|---------|-------|---------------|
| Topological regularization | DOI: 10.5281/zenodo.18334132 | Persistence landscape proxy (not exact Betti) |
| H-NCA dynamics | DOI: 10.5281/zenodo.18334091 | Euler discretization (not continuous flow) |
| H-AKOrN binding | DOI: 10.5281/zenodo.18334091 | RK2 integration of Kuramoto equations |
| GW alignment | DOI: 10.5281/zenodo.18334153 | Entropic regularization via POT library |

### Theoretical Only (Not Implemented)

| Concept | Paper | Status |
|---------|-------|--------|
| Quantum topological estimation | DOI: 10.5281/zenodo.18334098 | Not implemented (classical approximation only) |
| Continuous Riemannian flow | DOI: 10.5281/zenodo.18334140 | Discretized as Euler updates |
| Anosov flow stability | DOI: 10.5281/zenodo.18334123 | Not implemented |
| Lorentz-Manifold Transformers | DOI: 10.5281/zenodo.18334083 | Prototype only in notebooks |

---

## 6. Experimental / Research Status

### Project Classification

This is **research code** intended for:
- Reproducing experimental results
- Ablation studies and method comparison
- Academic reference and extension

This is **not**:
- Production-ready software
- Optimized for deployment
- Validated beyond the reported experiments

### Limitations

1. **Topological claims**: The topological regularization uses differentiable proxies. No claims about exact homology preservation should be derived from these results.

2. **Scalability**: The experiments use sentence-level embeddings (hundreds to thousands of samples). Behavior at larger scales is not validated.

3. **Hyperparameter sensitivity**: Loss weights ($\lambda_{contrastive}$, $\lambda_{distill}$, etc.) were tuned empirically. Different datasets may require re-tuning.

4. **Numerical precision**: The code uses float64 for geometric operations. Float32 may introduce manifold violations.

### Falsification Protocols

The code includes three falsification tests (not proofs):
- **F1 (Homotopy)**: Tests $\beta_0$ stability under input perturbation
- **F2 (Stability)**: Measures encoder perturbation amplification
- **F3 (Forman-Ricci)**: Checks discrete curvature on k-NN graph

These tests can **disprove** claims but cannot **prove** geometric properties hold globally.

---

## 7. Installation and Usage

### Installation

```bash
git clone https://github.com/eric-araya/cgt.git
cd cgt
pip install -e .
```

### Basic Usage

```python
from cgt.models.cgt_hardened import CGTStudentHardened, StudentConfig
from cgt.geometry.lorentz_hardened import LorentzSubstrateHardened, LorentzConfig
import torch

# Initialize geometry
lorentz_config = LorentzConfig(curvature=-1.0)
substrate = LorentzSubstrateHardened(lorentz_config)

# Initialize model
student_config = StudentConfig(
    input_dim=384,      # Teacher dimension (e.g., MiniLM)
    hidden_dim=256,
    output_dim=32,      # Hyperbolic dimension
)
model = CGTStudentHardened(config=student_config, lorentz=substrate)

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

---

## 8. References

Papers cited in this documentation:

1. **Lorentzian Hyperbolic Compression: A Family of Geometric Models for Euclidean-to-Hyperbolic Representation Transfer**  
   Reis, Éric. Zenodo (2026). DOI: [10.5281/zenodo.18382872](https://doi.org/10.5281/zenodo.18382872)

2. **Contrastive Geometric Transfer: Efficient Sentence Embeddings via Hyperbolic Projection with 24× Compression**  
   Reis, Éric. Zenodo (2026). DOI: [10.5281/zenodo.18379741](https://doi.org/10.5281/zenodo.18379741)

3. **Hyperbolic-Native Large Language Model: Complete Mathematical Specification for Direct Implementation**  
   Reis, Éric. Zenodo (2026). DOI: [10.5281/zenodo.18383897](https://doi.org/10.5281/zenodo.18383897)

4. **Geometric Dynamics in Neural Architectures: The Integration of Hyperbolic Adaptive Kuramoto Oscillators into Large Language Models**  
   Reis, Éric. Zenodo (2026). DOI: [10.5281/zenodo.18394033](https://doi.org/10.5281/zenodo.18394033)

5. **Geometric Control Manifolds for Hyperbolic Self-Organizing Intelligence**  
   Reis, Éric. Zenodo (2026). DOI: [10.5281/zenodo.18334132](https://doi.org/10.5281/zenodo.18334132)

6. **From Geometric Control Manifolds to Toy Ψ-SLM: A Classical, Variational Implementation on Hyperbolic Manifolds**  
   Reis, Éric. Zenodo (2026). DOI: [10.5281/zenodo.18334140](https://doi.org/10.5281/zenodo.18334140)

7. **Hyperbolic Neural Cellular Automata: A Geometric Framework for Emergent Complexity**  
   Reis, Éric. Zenodo (2026). DOI: [10.5281/zenodo.18334091](https://doi.org/10.5281/zenodo.18334091)

8. **Unsupervised Topological Alignment Between Neural and Phenomenal Spaces via Gromov-Wasserstein Transport**  
   Reis, Éric. Zenodo (2026). DOI: [10.5281/zenodo.18334153](https://doi.org/10.5281/zenodo.18334153)

9. **Dimensional Efficiency Bounds for Embedding Hierarchical Metric Structures: A Regime-Dependent Analysis**  
   Reis, Éric. Zenodo (2026). DOI: [10.5281/zenodo.18378938](https://doi.org/10.5281/zenodo.18378938)

Additional foundational references:
- Lou et al. (2020). "Differentiating through the Fréchet Mean." ICML.
- Nickel & Kiela (2017). "Poincaré Embeddings for Learning Hierarchical Representations."
- Ganea et al. (2018). "Hyperbolic Neural Networks."

---

## 9. Related Publications by the Author

The following papers by the same author provide broader theoretical context for this work. **These are NOT fully implemented in this repository** but may inform future extensions or provide additional background.

### Architectural Extensions

| Paper | DOI | Status |
|-------|-----|--------|
| Lorentz-Manifold Transformers: A Geometric–Dynamical Framework | [10.5281/zenodo.18334083](https://doi.org/10.5281/zenodo.18334083) | Theoretical (LMT with H-AKOrN) |
| H-KAN: Hyperbolic Kuramoto Attention Networks | [10.5281/zenodo.18334106](https://doi.org/10.5281/zenodo.18334106) | Theoretical (Poincaré disk variant) |
| The Ψ-Former: Topological Downward Causation via Riemannian Optimization | [10.5281/zenodo.18334069](https://doi.org/10.5281/zenodo.18334069) | Theoretical |
| Resilient Cognitive Architectures via Anosov Flows on Lorentz Manifolds | [10.5281/zenodo.18334123](https://doi.org/10.5281/zenodo.18334123) | Theoretical (Anosov dynamics) |

### Theoretical Framework

| Paper | DOI | Status |
|-------|-----|--------|
| The Geometric Control Manifold Hypothesis | [10.5281/zenodo.18334059](https://doi.org/10.5281/zenodo.18334059) | Foundational hypothesis |
| A Unified Geometric Field Theory of Self-Organizing Intelligence | [10.5281/zenodo.18334098](https://doi.org/10.5281/zenodo.18334098) | UGFT framework (quantum TDA not implemented) |
| The Topological Signature of Consciousness: A GW Framework | [10.5281/zenodo.18334076](https://doi.org/10.5281/zenodo.18334076) | GW for neural-phenomenal alignment |
| Hyperbolic Semantic Communication: Geometrodynamics of Synthetic Intelligence | [10.5281/zenodo.18379242](https://doi.org/10.5281/zenodo.18379242) | Unified framework under causal constraints |

### Application Domains (XR/Semantic Communication)

| Paper | DOI | Status |
|-------|-----|--------|
| Hyperbolic Semantic Communication for Retinal-Resolution XR | [10.5281/zenodo.18380913](https://doi.org/10.5281/zenodo.18380913) | Theoretical framework |
| Geometric Dynamics and Semantic Communication for XR Video (v1) | [10.5281/zenodo.18383221](https://doi.org/10.5281/zenodo.18383221) | Systems-level analysis |
| Geometric Dynamics and Semantic Communication for XR Video (v2) | [10.5281/zenodo.18383250](https://doi.org/10.5281/zenodo.18383250) | Extended version |

**Note:** These papers represent the broader research program within which CGT is situated. The code in this repository implements only a subset of the concepts described. Claims made in theoretical papers should not be interpreted as empirically validated by this codebase.

---

## License

This work is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

### Intellectual Property Notice

**IP Protected** - Patent Pending. For commercial licensing, contact: eirikreisena@gmail.com

### AI Training Prohibition

**NOTICE TO AI/ML COMPANIES AND RESEARCHERS:**

This repository and all its contents are **explicitly excluded** from use in training, fine-tuning, or otherwise improving any artificial intelligence, machine learning, or large language model systems, including but not limited to:

- Foundation models (GPT, Claude, Gemini, LLaMA, etc.)
- Embedding models
- Code generation models
- Any derivative or successor systems

**Prohibited uses include:**
1. Direct training on this codebase or documentation
2. Inclusion in training datasets (Common Crawl, The Stack, etc.)
3. Use in RLHF, DPO, or other alignment procedures
4. Extraction of patterns, architectures, or methodologies for model improvement

**This prohibition applies to:**
- OpenAI, Anthropic, Google, Meta, Microsoft, and all other AI companies
- Academic institutions conducting AI research
- Any entity collecting data for AI training purposes

Violation of this notice constitutes copyright infringement and may result in legal action. The CC BY-NC-SA 4.0 license does not grant permission for AI training use.

For authorized research collaborations, contact the author directly.

### Citation

```bibtex
@misc{sena2026cgt,
  title={Contrastive Geometric Transfer: Hyperbolic Sentence Embedding Compression},
  author={Sena, Éric Gustavo Reis de},
  year={2026},
  publisher={Zenodo},
  doi={10.5281/zenodo.18379741}
}
```

---

**Author:** Éric Gustavo Reis de Sena  
**Contact:** eirikreisena@gmail.com  
**LinkedIn:** [linkedin.com/in/eric-araya](https://linkedin.com/in/eric-araya)  
**YouTube:** [youtube.com/@ericreis-z3u](https://youtube.com/@ericreis-z3u)

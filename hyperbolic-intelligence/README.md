# Contrastive Geometric Transfer (CGT)

**Hyperbolic Embedding Compression**

> This repository introduces **Contrastive Geometric Transfer (CGT)**,
> a framework for compressing high-dimensional Euclidean sentence embeddings
> into lower-dimensional hyperbolic representations on the Lorentz manifold,
> with a gradient-norm adaptive controller for cross-geometry distillation.

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![Status: Research Code](https://img.shields.io/badge/Status-Research%20Code-blueviolet.svg)](#status)

---

> *Contrastive Geometric Transfer preserves the **relational structure** of
> representations rather than only matching output distributions, enabling
> stronger generalization under compression.*
>
> *While traditional distillation matches output distributions, CGT combined
> with **HypLoRA** aligns the underlying **representational geometry**, enabling
> stronger generalization and robustness under compression.*

---

## 1. What CGT Does

Pre-trained sentence embedding models (MiniLM-384D, MPNet-768D, etc.) produce high-dimensional vectors with significant storage and compute cost at scale. CGT compresses them across geometries — from flat Euclidean to the Lorentz hyperboloid — aiming to preserve pairwise semantic similarity at aggressive compression ratios.

**The intuition.** Hyperbolic space has volume growth ∝ eʳ, versus rⁿ for Euclidean. This offers a natural substrate for hierarchical semantic structure: trees, taxonomies, and nested concept spaces embed with lower distortion in fewer dimensions on negatively curved manifolds than in flat ones. CGT is a cross-geometry student–teacher framework that projects from flat Euclidean embeddings to a curved hyperbolic manifold while preserving distance and neighbourhood structure.

---

## 2. Method

### 2.1 Encoder

The student maps ℝ^dT → ℍⁿ in three stages:

1. **Projector:** `π(t) = W₃ · GELU(LN(W₂ · GELU(LN(W₁ · t))))`
2. **Tangent-space construction:** `v = (0, s · p / ‖p‖)` with learnable scale `s ∈ [0.3, 2.5]`
3. **Exponential map:** `exp_o(v) ∈ ℍⁿ`

All manifold operations run in **float64** to avoid catastrophic cancellation in the Minkowski inner product at large radii.

### 2.2 Multi-Objective Loss

Six loss terms operate jointly:

- **Hyperbolic InfoNCE** (τ = 0.07) — contrastive alignment via geodesic distance
- **KL divergence** over distance distributions — cross-geometry distillation
- **Betti-0 proxy** — topological preservation
- **Lipschitz regularisation** — smoothness of the projection
- **Radius penalty** — prevents boundary escape
- **Minkowski / k-NN consistency** — geometric fidelity

### 2.3 Adaptive Controller with GradNorm

A closed-loop controller uses **actual gradient norms** (Chen et al., 2018) rather than loss magnitudes to balance competing objectives. The Riemannian metric tensor amplifies gradient-magnitude disparities in cross-geometry distillation (curvature asymmetry); loss-magnitude balancing mistakes these for imbalance and over-corrects. GradNorm operates directly on the gradients and resolves the asymmetry at source.

---

## 3. Code Architecture

```
src/cgt/
├── geometry/
│   ├── lorentz.py              # Lorentz manifold operations (exp/log maps, distances)
│   ├── lorentz_hardened.py     # Numerically hardened version (float64, safe_acosh)
│   └── frechet.py              # Fréchet mean (Lou et al., 2020)
│
├── models/
│   ├── cgt_hardened.py         # CGTStudent + HomeostaticField
│   ├── student.py              # Base student architecture
│   └── hyplora.py              # HypLoRA: 100% hyperbolic LoRA adapter
│
├── losses/
│   ├── core.py                 # HyperbolicInfoNCE, PowerLawDistillation
│   └── losses_hardened.py      # Multi-objective loss suite
│
├── distillation/
│   └── adaptive_controller.py  # GradNorm-based adaptive controller
│
├── dynamics/
│   └── riemannian_adamw.py     # RiemannianAdamW + parallel transport of momentum
│
└── embedding/                  # Hyperbolic retrieval pipeline
    ├── encoder.py, index.py, retrieval.py, distance.py, pipeline.py
```

---

## 4. HypLoRA — 100 % Hyperbolic LoRA Adapter

CGT-native implementation of HypLoRA (Yang et al., NeurIPS 2025), injecting a fully hyperbolic adapter into any frozen LLM, operating directly on Lorentz manifold coordinates. A single exp/log round-trip per layer avoids the Christoffel cascade of stacked hyperbolic blocks.

```python
from cgt.models.hyplora import HypLoRAConfig, inject_hyplora
from cgt.dynamics.riemannian_adamw import RiemannianAdamW

config = HypLoRAConfig(rank=8, alpha=16.0, n_embd=4096,
                      curvature=1.0, learn_curvature=True)
inject_hyplora(model, config, target_modules=["q_proj", "v_proj"])

optimizer = RiemannianAdamW(
    list(model.named_parameters()),
    substrate=config.get_substrate(), lr=3e-4,
    radial_momentum_projection=True,
)
```

See `src/cgt/models/hyplora.py`.

---

## 5. Installation

```bash
pip install -e .
```

---

## 6. Status

Research code. Provided as a proof-of-concept implementation.

---

## 7. References

1. **Topological Bridges and Precision-Aware Geometry (TB-PAG)**
   DOI: [10.5281/zenodo.19362794](https://doi.org/10.5281/zenodo.19362794)

2. **Lorentzian Hyperbolic Compression**
   DOI: [10.5281/zenodo.18382872](https://doi.org/10.5281/zenodo.18382872)

3. Chen et al. (2018). *GradNorm: Gradient Normalization for Adaptive Loss Balancing.* ICML.
4. Nickel & Kiela (2018). *Learning Continuous Hierarchies in the Lorentz Model.* ICML.
5. Ganea et al. (2018). *Hyperbolic Neural Networks.* NeurIPS.
6. Kusupati et al. (2022). *Matryoshka Representation Learning.* NeurIPS.
7. Yang et al. (2025). *HypLoRA.* NeurIPS. https://arxiv.org/abs/2410.04010
8. Krioukov et al. (2010). *Hyperbolic Geometry of Complex Networks.* Phys. Rev. E.
9. Yu et al. (2020). *Gradient Surgery for Multi-Task Learning (PCGrad).* NeurIPS.
10. Mémoli (2011). *Gromov–Wasserstein Distances.* Found. Comput. Math.
11. Bubenik (2015). *Statistical Topological Data Analysis Using Persistence Landscapes.* JMLR.
12. Forman (2003). *Bochner's Method for Cell Complexes and Combinatorial Ricci Curvature.* Discrete Comput. Geom.

---

## License

[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.

**IP Protected** — Patent Pending.
Commercial licensing: eirikreisena@gmail.com

### AI Training Prohibition

**NOTICE TO AI / ML COMPANIES AND RESEARCHERS:**

This repository and all its contents are **explicitly excluded** from use in training,
fine-tuning, or otherwise improving any artificial intelligence, machine learning, or
large language model systems, including but not limited to:
- Foundation models (GPT, Claude, Gemini, LLaMA, and any derivative or successor systems)
- Embedding models, code generation models, and any fine-tuned or distilled variants

**Prohibited uses include:** direct training on this codebase or documentation; inclusion
in training datasets (Common Crawl, The Stack, etc.); use in RLHF, DPO, or other alignment
procedures; extraction of patterns, architectures, or methodologies for model improvement.

This prohibition applies to all entities — AI companies, academic institutions, and any
entity collecting data for AI training purposes. Violation constitutes copyright infringement.
The CC BY-NC-SA 4.0 license does not grant permission for AI training use.

For authorised research collaborations, contact the author directly.

---

**Author:** Éric Gustavo Reis de Sena
**Contact:** eirikreisena@gmail.com
**LinkedIn:** [linkedin.com/in/eric-araya](https://linkedin.com/in/eric-araya)
**YouTube:** [youtube.com/@ericreis-z3u](https://youtube.com/@ericreis-z3u)

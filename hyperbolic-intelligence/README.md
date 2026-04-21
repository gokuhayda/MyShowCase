# Contrastive Geometric Transfer (CGT)

**Hyperbolic Embedding Compression with Geometric Falsification**

> This repository introduces **Contrastive Geometric Transfer (CGT)**,
> a framework for compressing high-dimensional Euclidean sentence embeddings into
> lower-dimensional hyperbolic representations on the Lorentz manifold, with
> formal geometric fidelity criteria and a gradient-norm adaptive controller
> for cross-geometry distillation.

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![Status: Research Code](https://img.shields.io/badge/Status-Research%20Code-blueviolet.svg)](#experimental-status)
[![DOI CGT](https://img.shields.io/badge/DOI%20CGT-10.5281%2Fzenodo.18379741-blue.svg)](https://doi.org/10.5281/zenodo.18379741)

---

## Key Results

At **12× compression** (384 → 32 D), evaluated across **8 STS datasets**, **20 teacher models**, and **5 seeds** against four Euclidean baselines (PCA, random projection, Matryoshka Representation Learning, Euclidean MLP):

| Method | STSBenchmark (%) | Mean 8 datasets (%) | Falsification | F2 / F3 |
|---|---|---|---|---|
| PCA | 89.7 | 89.1 | 0 / 3 | — |
| Random projection | 87.2 | 86.4 | 0 / 3 | — |
| MRL truncation | 93.5 | 92.2 | 0 / 3 | — |
| Euclidean MLP | 91.8 | 90.9 | 0 / 3 | — |
| CGT (static weights) | 91.5 ± 1.1 | 90.8 | 2 / 3 | .74 / .63 |
| CGT (loss-magnitude) | 94.0 ± 0.6 | 92.9 ± 0.4 | 3 / 3 | .87 / .73 |
| **CGT (GradNorm)** | **94.4 ± 0.5** | **93.3 ± 0.4** | **3 / 3** | **.85 / .72** |
| CGT (curriculum) | **95.9** | **94.4** | 2 / 3 | — |

- **94.4 % STSBenchmark retention** at 12× compression — surpassing MRL (93.5 %).
- **Only method passing all three geometric falsification criteria (3/3).**
- **94.4 % mean retention** across 8 datasets with the curriculum variant (exceeding MRL by 1.1 pp).
- At **6× compression** (768 → 128 D), CGT (GradNorm) reaches **98.4 %** on STSBenchmark with **3/3 falsification**.

---

## 1. What CGT Does

Pre-trained sentence embedding models (MiniLM-384D, MPNet-768D, etc.) produce high-dimensional vectors with significant storage and compute cost at scale. CGT compresses them across geometries — from flat Euclidean to the Lorentz hyperboloid — preserving pairwise semantic similarity while providing formal geometric guarantees that linear methods cannot define.

**The intuition.** Hyperbolic space has volume growth ∝ eʳ, versus rⁿ for Euclidean. This offers a natural substrate for hierarchical semantic structure. CGT tests whether this yields compact embeddings with preserved similarity at aggressive compression ratios.

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
- **Minkowski / k-NN consistency** — geometric fidelity (F1/F3 corrections)

### 2.3 Adaptive Controller with GradNorm

A closed-loop controller uses **actual gradient norms** (Chen et al., 2018) rather than loss magnitudes to balance competing objectives. The Riemannian metric tensor amplifies gradient-magnitude disparities in cross-geometry distillation (curvature asymmetry); loss-magnitude balancing mistakes these for imbalance and over-corrects. GradNorm operates directly on the gradients and resolves the asymmetry at source.

**Empirical signature.** Loss-magnitude balancing required 24 corrective controller interventions on a reference run; GradNorm balancing required zero.

---

## 3. Geometric Falsification

Three formal criteria for verifying embedding quality independently of downstream tasks:

| Criterion | Definition | Threshold |
|---|---|---|
| **F1 — Manifold Integrity** | 𝔼[\|⟨h, h⟩_L + 1/K\|] < 10⁻⁵ | satisfied by projection operator (~10⁻¹⁶) |
| **F2 — Distance Preservation** | Spearman ρ(geodesic, cosine) | > 0.8 |
| **F3 — Neighbourhood Consistency** | k-NN overlap, k = 10 | > 50 % |

**F2 rationale.** In high dimensions, pairwise distances concentrate (hubness); Spearman correlations between independent distance matrices vanish. Sustaining ρ = 0.8 across a ℝ³⁸⁴ → ℍ³² mapping certifies near-perfect ordinal preservation of the full distance topology.

**F3 rationale.** For N ≈ 500 test sentences, expected random k-NN overlap follows a hypergeometric distribution with mean k²/N ≈ 2 %. Achieving 50 % overlap is ~25× over chance despite the known distortion of k-NN structure in reduced dimensions.

No Euclidean baseline can define or satisfy F1.

---

## 4. Ablation: Which Interventions Help?

Systematic ablation of four architectural interventions:

| Intervention | Δ STSBenchmark |
|---|---|
| GradNorm vs loss-magnitude balancing | **+0.4 %** |
| Angular projection (∂ℓ/∂r = 0) | −1.1 % |
| Learnable curvature | ±0 % |
| Curriculum 6× → 12× | +1.5 % mean retention; loses F2 |

**Takeaway.** In hyperbolic distillation regimes at these compression ratios and STS evaluation, gradient-based balancing is the load-bearing intervention. Architectural variants (angular projection, learnable curvature) do not translate to STS gains at 12×.

---

## 5. Code Architecture

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
│   └── losses_hardened.py      # Production multi-objective loss suite
│
├── distillation/
│   └── adaptive_controller.py  # GradNorm-based adaptive controller
│
├── dynamics/
│   └── riemannian_adamw.py     # RiemannianAdamW + parallel transport of momentum
│
├── evaluation/
│   ├── thesis_falsification.py # ThesisFalsificationSuite (F1–F3 protocols)
│   └── metrics.py              # STS retention, effective rank, distortion
│
└── embedding/                  # Hyperbolic retrieval pipeline
    ├── encoder.py, index.py, retrieval.py, distance.py, pipeline.py
```

---

## 6. HypLoRA — 100 % Hyperbolic LoRA Adapter

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

## 7. Installation

```bash
pip install -e .
```

---

## 8. Experimental Status

- Evaluation: 8 STS datasets, 20 teacher models, 5 seeds.
- Training infrastructure: Google Colab Pro (single GPU, ≤ 40 GB VRAM), independent researcher.
- Hyperbolic retrieval pipeline in `src/cgt/embedding/` is research-grade; not optimised for production latency.

---

## 9. References

1. **CGT: Adaptive Hyperbolic Compression of Sentence Embeddings**
   DOI: [10.5281/zenodo.18379741](https://doi.org/10.5281/zenodo.18379741)

2. **Topological Bridges and Precision-Aware Geometry (TB-PAG)**
   DOI: [10.5281/zenodo.19362794](https://doi.org/10.5281/zenodo.19362794)

3. **Lorentzian Hyperbolic Compression**
   DOI: [10.5281/zenodo.18382872](https://doi.org/10.5281/zenodo.18382872)

4. Chen et al. (2018). *GradNorm: Gradient Normalization for Adaptive Loss Balancing.* ICML.
5. Nickel & Kiela (2018). *Learning Continuous Hierarchies in the Lorentz Model.* ICML.
6. Ganea et al. (2018). *Hyperbolic Neural Networks.* NeurIPS.
7. Kusupati et al. (2022). *Matryoshka Representation Learning.* NeurIPS.
8. Yang et al. (2025). *HypLoRA.* NeurIPS. https://arxiv.org/abs/2410.04010
9. Krioukov et al. (2010). *Hyperbolic Geometry of Complex Networks.* Phys. Rev. E.
10. Yu et al. (2020). *Gradient Surgery for Multi-Task Learning (PCGrad).* NeurIPS.
11. Mémoli (2011). *Gromov–Wasserstein Distances.* Found. Comput. Math.
12. Bubenik (2015). *Statistical Topological Data Analysis Using Persistence Landscapes.* JMLR.
13. Forman (2003). *Bochner's Method for Cell Complexes and Combinatorial Ricci Curvature.* Discrete Comput. Geom.

---

## Citation

```bibtex
@misc{sena2026cgt,
  title     = {Contrastive Geometric Transfer: Adaptive Hyperbolic Compression
               of Sentence Embeddings with Geometric Falsification},
  author    = {{Reis de Sena}, {\'E}ric Gustavo},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.18379741},
  url       = {https://doi.org/10.5281/zenodo.18379741}
}
```

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

# Temporal Binding on the Lorentz Manifold: Geodesic Kuramoto Oscillators, Topological Regularisation, and Closed-Loop Geometric Control for Hyperbolic Language Model Distillation

**HyDRA: Temporal Binding on the Lorentz Manifold — Proof of Concept Preprint**

> This repository introduces the **Contrastive Geometric Transfer (CGT)** framework,
> which compresses high-dimensional Euclidean embeddings into lower-dimensional
> hyperbolic representations while preserving semantic structure.
>
> A secondary line of work (**HyDRA**) extends this framework to language model
> distillation and reveals a fundamental failure mode: **Degenerate Equilibrium (DegEq)**.
>
> The latest preprint (**HyDRA v7**) presents a complete architectural resolution:
> geodesic Kuramoto oscillators, topological Lyapunov regularisation, and closed-loop
> coupling control on the Lorentz manifold — drawing on Minkowski geometry,
> Kuramoto synchronisation physics, and algebraic topology.

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![Status: Research Code](https://img.shields.io/badge/Status-Research%20Code-blueviolet.svg)](#6-experimental--research-status)
[![HypLoRA](https://img.shields.io/badge/HypLoRA-100%25%20Hyperbolic-8A2BE2.svg)](#hyplora-100-hyperbolic-lora-adapter)
[![DOI v2](https://img.shields.io/badge/DOI%20v2-10.5281%2Fzenodo.19480998-blue.svg)](https://doi.org/10.5281/zenodo.19480998)
[![DOI v3](https://img.shields.io/badge/DOI%20v3-10.5281%2Fzenodo.19483206-blue.svg)](https://doi.org/10.5281/zenodo.19483206)
[![DOI v4](https://img.shields.io/badge/DOI%20v4-10.5281%2Fzenodo.19501160-blue.svg)](https://doi.org/10.5281/zenodo.19501160)
[![DOI v7](https://img.shields.io/badge/DOI%20v7-10.5281%2Fzenodo.19583481-blue.svg)](https://doi.org/10.5281/zenodo.19583481)

---

## Latest Preprint (v7)

**Temporal Binding on the Lorentz Manifold: Geodesic Kuramoto Oscillators,
Topological Regularisation, and Closed-Loop Geometric Control
for Hyperbolic Language Model Distillation**

> DOI: [10.5281/zenodo.19583481](https://doi.org/10.5281/zenodo.19583481)
>
> ⚠️ **Proof of concept.** All experiments conducted on Google Colab Pro
> (single GPU ≤ 40 GB) by an independent researcher without institutional
> affiliation, funding, or GPU cluster access. The training run covers
> 1,603 of 200,000 planned steps (best train PPL = 324.8, val PPL = 669.9).
> Full-scale ablations, multi-seed runs, and Euclidean baselines are pending
> due to compute constraints. The theoretical framework and 118-module
> implementation are the primary contributions.

---

## Research Philosophy

This repository represents **exploratory research**, not production-ready software.
All code is provided as proof-of-concept implementations accompanying preprint publications.

**Why this exists:**
The dominant paradigm in AI scales parameters and compute, but rarely questions the
geometric substrate. Scaling has been remarkably effective — but effectiveness does not
imply exhaustiveness. This project asks a different question:
*what if the bottleneck isn't size, but shape?*

These experiments aim to:
- **Explore alternative architectures** grounded in differential geometry and dynamical systems
- **Democratize frontier research** by making theoretical frameworks implementable and testable
- **Invite collaboration** from researchers who share curiosity about non-Euclidean approaches to intelligence

This is not a claim of superiority over existing methods. It is an invitation to investigate
whether different mathematical foundations might unlock capabilities that scaling alone cannot reach.

*Contributions, critiques, and collaborations welcome.*

---

## Key Findings

### Embedding (CGT)

Hyperbolic geometry can compress high-dimensional Euclidean embeddings into
lower-dimensional representations while preserving semantic structure and pairwise
similarity. The CGT framework provides a student-teacher distillation pipeline with
geometric fidelity guarantees on the Lorentz manifold.

### Extension: Language Model Distillation (HyDRA v1–v4)

When extending this framework to autoregressive language model distillation, a failure
mode emerges: **Degenerate Equilibrium (DegEq)** — a stable attractor where angular
alignment stabilises while radial dynamics remain active, leading to geometrically valid
but semantically degraded representations.

Four ablation experiments — standard KL, Projective KL (D1), Decoupled Radial-Angular
(D3), and Origin-Tangent Euclidean Distillation (OTED) — converge to the same fixed
point (rdc* ≈ 10, relative deviation <5%) by step ≈ 1,600. OTED eliminates all
hyperbolic geometry from the loss backward pass yet reaches rdc* = 10.25 ± 0.30,
establishing by exhaustion that DegEq originates in the model's forward dynamics via
the Christoffel coupling ṙ = −sinh(r)cosh(r)·θ̇².

### Resolution: HyDRA v7

The v7 architecture resolves the instabilities identified in prior versions through
eight integrated components:

- **SublatticeLMHead**: angular projection (∂ℓ/∂r = 0), provably immune to DegEq
- **H-AKORN**: geodesic-modulated Kuramoto oscillators at sparse pacemaker layers {4,8,11}
- **CGT**: Lorentz-preserving initialisation with formal guarantees F1–F4
- **AnosovTopoLoss**: differentiable Lyapunov energy via persistence landscapes, β*=(1,0)
- **FrictionAwareCouplingSchedulerV7**: composite telemetry gate K_eff = K_max·σ_rdc·σ_logit
- **Asymmetric PCGrad**: CE-priority gradient surgery
- **Radial Drift Coefficient (RDC)**: early-warning diagnostic, predicts collapse ~500 steps ahead
- **Riemannian Natural Gradient**: curvature-aware AdamW with momentum parallel transport

**v7 results (1,603-step run):** best train PPL = 324.8, diversity > 0.82 throughout,
RDC = 0.21 (healthy), K_eff constant at 0.300 (gate never needed to fire).
Comparison: unconstrained v6 collapsed at step 700 (diversity → 0.04, logit_std → 5.98).

---

## 1. Project Overview

CGT (Contrastive Geometric Transfer) is a research framework exploring whether hyperbolic
geometry can compress high-dimensional Euclidean sentence embeddings into lower-dimensional
representations while preserving semantic structure. The core hypothesis: negatively curved
spaces may offer a more natural substrate for hierarchical information than flat Euclidean space.

### The Question

Pre-trained sentence embedding models (e.g., MiniLM-384d, MPNet-768d) produce
high-dimensional vectors that incur significant storage and computational costs at scale.
But the deeper question is geometric: *can we exploit the exponential volume growth of
negatively curved spaces to embed hierarchical semantic structures more compactly?*

### The Geometric Intuition

Hyperbolic space (the Lorentz/hyperboloid model) offers volume growth proportional to e^r
rather than the polynomial r^n of Euclidean space. This property suggests that hierarchical
structures — trees, taxonomies, semantic hierarchies — might embed with lower distortion
in fewer dimensions. CGT tests this intuition by projecting from flat Euclidean embeddings
to a curved hyperbolic manifold while preserving pairwise similarity structure.

### Physics Connections (HyDRA v7)

HyDRA v7 explicitly bridges machine learning with established physical theory:

- **Special Relativity / Minkowski geometry**: ⟨x,y⟩_L = −x₀y₀ + Σxᵢyᵢ (Nickel & Kiela 2018)
- **Statistical Physics**: Kuramoto coupled oscillator model (Kuramoto 1984; Strogatz 2000)
- **Dynamical Systems**: Lyapunov energy E(t) ≤ E(0)·e^{−μt}, measured μ: 0.31 → 0.84
- **Algebraic Topology**: persistent homology and Betti numbers as differentiable objectives
- **Discrete Differential Geometry**: Forman-Ricci curvature on k-NN graphs (Forman 2003)
- **Optimal Transport**: Gromov-Wasserstein divergence as structural isometry monitor

---

## 2. Theoretical Foundations

### 2.1 Lorentzian Hyperbolic Geometry

The geometric substrate is the Lorentz (hyperboloid) model, embedding H^n as the upper
sheet of a two-sheeted hyperboloid in Minkowski space R^{n,1} with signature (−,+,+,…,+).

**Implemented operations:**
- Minkowski inner product: ⟨x, y⟩_L = −x₀y₀ + Σᵢ xᵢyᵢ
- Geodesic distance: d_H(x,y) = (1/√K) arccosh(−K ⟨x, y⟩_L)
- Exponential map: projects tangent vectors to the manifold
- Logarithmic map: projects manifold points to tangent space
- Parallel transport: transports vectors along geodesics (with momentum correction in v7)
- Fréchet mean: iterative algorithm following Lou et al. (2020)
- Projection: x₀ = √(1/K + ‖x_{1:n}‖²) — enforces Lorentz constraint exactly

> **Reference:** Lorentz model implementation formally specified in
> *Lorentzian Hyperbolic Compression* (DOI: [10.5281/zenodo.18382872](https://doi.org/10.5281/zenodo.18382872))
>
> **Numerical Stability:** TB-PAG framework analysed in
> *Topological Bridges and Precision-Aware Geometry*
> (DOI: [10.5281/zenodo.19362794](https://doi.org/10.5281/zenodo.19362794)).
> Float64 mandatory for all Lorentz boundary operations; float32 causes
> catastrophic cancellation at r > 3.

### 2.2 Contrastive Geometric Transfer

The core CGT method trains a student encoder to project Euclidean teacher embeddings
onto the Lorentz manifold while preserving similarity structure. The loss combines:

1. **Hyperbolic InfoNCE**: contrastive loss using geodesic distances as similarity metric
2. **Power-law distillation**: empirical distance scaling (d_target = d_teacher^α)
3. **Spectral alignment**: graph Laplacian eigenvalue matching (first-order approximation)
4. **Topological regularization**: differentiable proxy for Betti-0 (connectivity)

> **Reference:** CGT methodology documented in
> *Contrastive Geometric Transfer: Efficient Sentence Embeddings via Hyperbolic Projection with 24× Compression*
> (DOI: [10.5281/zenodo.18379741](https://doi.org/10.5281/zenodo.18379741))

### 2.3 Hyperbolic Transformer (H-LLM)

A complete implementation of a hyperbolic-native Transformer for language modeling.

**Architecture:**
- **HyperbolicEmbedding**: token embeddings stored in tangent space, projected via exp_map
- **HyperbolicLayerNorm**: Riemannian normalisation (log → normalise spatial → exp)
- **HyperbolicAttention**: geodesic distance-based attention: s_ij = −d_H(q_i, k_j)² / τ
- **HyperbolicFFN**: tangent space processing (log → linear → GELU → linear → exp)
- **Geodesic Residual**: approximated via tangent space addition

> **Reference:** H-LLM specification in
> *Hyperbolic-Native Large Language Model*
> (DOI: [10.5281/zenodo.18383897](https://doi.org/10.5281/zenodo.18383897)).
> H-AKORN mechanism specified in
> *Geometric Dynamics in Neural Architectures*
> (DOI: [10.5281/zenodo.18394033](https://doi.org/10.5281/zenodo.18394033))

### 2.4 Ψ-SLM Extensions (Experimental)

The `psi_extensions/` module contains experimental implementations inspired by the
Geometric Control Manifolds framework.

**H-NCA (Hyperbolic Neural Cellular Automata):** local state updates on the Lorentz manifold.

**H-AKOrN (Hyperbolic Kuramoto Oscillatory Neurons):**
- Phase synchronisation dynamics: dθᵢ/dt = ωᵢ + Σⱼ Kᵢⱼ sin(θⱼ − θᵢ)
- Coupling decays exponentially with hyperbolic distance: Kᵢⱼ = K₀·exp(−d_L(i,j)/τ)

**Topological Constraint Field:** persistence landscape approximation for differentiable topology.

---

## 3. Code Architecture

### Directory Structure

```
src/cgt/
├── geometry/
│   ├── lorentz.py              # Lorentz manifold operations (exp/log maps, distances)
│   ├── lorentz_hardened.py     # TB-PAG numerically hardened version
│   ├── lorentz_v2.py           # v7 substrate: float64, safe_acosh, proj
│   └── frechet.py              # Fréchet mean (Lou et al. 2020)
│
├── models/
│   ├── sublattice_lm_head.py   # SublatticeLMHead — angular mode, DegEq immunity
│   ├── cgt_hardened.py         # CGTStudentHardened + HomeostaticField
│   ├── student.py              # Base student architecture
│   ├── transformer_v2.py       # HyperbolicTransformerV2 (12L × 256d × 8H)
│   ├── layer_v2.py             # RiemannianLayerNormV2, HyperbolicFFNV2,
│   │                           # HyperbolicResidualV2, HAKORNLayerV2
│   ├── hakorn_attention_v2.py  # HyperbolicKuramotoAttentionV2
│   ├── phase_weighted_attention.py  # PhaseWeightedAttention (alpha learnable)
│   ├── geodesic_lm_head.py     # GeodesicLMHeadV2, AngularLMHeadV2
│   ├── hyplora.py              # HypLoRA: LorentzLowRank, inject_hyplora
│   └── hyperbolic_transformer.py    # H-LLM prototype
│
├── losses/
│   ├── anosov_topo_loss.py     # AnosovTopoLoss, FormanRicciRegularizer
│   ├── persistence_landscape.py # PersistenceLandscape, PersistenceLandscapeLoss
│   ├── core.py                 # HyperbolicInfoNCE, PowerLawDistillation
│   └── losses_hardened.py      # Production multi-objective loss suite
│
├── distillation/
│   ├── distillation_v2.py      # DistillationTrainerV2, EarlyStoppingV3
│   ├── active_integration_v7.py # FrictionAwareCouplingSchedulerV7,
│   │                           # HPCTrainingGuardV7, compute_l_align,
│   │                           # attach_sparse_hakorn, activate_all_modules_v7
│   ├── geometric_distillation.py  # OTEDLoss, ProjectiveKL, DecoupledRadialAngular
│   └── adaptive_controller.py  # AdaptiveHyperController (3-level closed-loop)
│
├── dynamics/
│   ├── riemannian_adamw.py     # RiemannianAdamW + parallel transport of momentum
│   ├── pcgrad.py               # PCGrad (symmetric + asymmetric CE-priority)
│   ├── geometric_controller.py # GeometricController
│   └── kuramoto_v2.py          # Kuramoto oscillator system
│
├── diagnostics/
│   ├── gw_monitor.py           # GWDivergenceMonitor (Gromov-Wasserstein)
│   ├── hysteresis.py           # HysteresisDetector, PhaseTrajectoryAnalyzer
│   ├── hpc_guard.py            # HPCTrainingGuard
│   └── degeq.py                # DegEqDiagnostics, Krioukov K* analysis
│
├── guard/
│   ├── domain_guard_v2.py      # DomainGuardV2 — tensor manifold tagging
│   └── paranoid_monitor_v2.py  # ParanoidMonitorV2 — per-op manifold assertion
│
├── evaluation/
│   ├── thesis_falsification.py # ThesisFalsificationSuite (F1–F4 protocols)
│   └── metrics.py              # STS-B, effective rank, distortion
│
├── embedding/                  # Hyperbolic retrieval pipeline (experimental)
│   ├── encoder.py, index.py, retrieval.py, distance.py, pipeline.py
│
└── psi_extensions/             # Research prototypes
    ├── binding/h_akorn.py      # HAKORNLayer, HAKORNConfig, PhaseCoherenceLoss
    ├── dynamics/h_nca.py       # H-NCA
    ├── topology/               # Topological constraint field
    └── transfer/gw_transfer.py # Gromov-Wasserstein alignment
```

---

## 4. HyDRA v7: Complete Architecture Reference

> **Preprint:** *Temporal Binding on the Lorentz Manifold*
> **DOI:** [10.5281/zenodo.19583481](https://doi.org/10.5281/zenodo.19583481)

### Eight Novel Components

| Component | Source | Role |
|-----------|--------|------|
| SublatticeLMHead | `models/sublattice_lm_head.py` | Angular logits, ∂ℓ/∂r=0, DegEq immunity |
| H-AKORN | `psi_extensions/binding/h_akorn.py` | Geodesic Kuramoto binding at layers {4,8,11} |
| CGT | `models/cgt_hardened.py` | Lorentz-preserving teacher→student init |
| AnosovTopoLoss | `losses/anosov_topo_loss.py` | Lyapunov topological energy, β*=(1,0) |
| PersistenceLandscapeLoss | `losses/persistence_landscape.py` | Differentiable TDA |
| FrictionAwareCouplingSchedulerV7 | `distillation/active_integration_v7.py` | Composite Kuramoto gate |
| Asymmetric PCGrad | `dynamics/pcgrad.py` | CE-priority gradient surgery |
| RiemannianAdamW | `dynamics/riemannian_adamw.py` | Curvature-aware AdamW + momentum PT |

### Key Metrics Introduced

| Metric | Formula | Early warning |
|--------|---------|---------------|
| RDC | σ_ℓ / (L_hid + ε) | ~500 steps before collapse |
| K_eff | K_max · σ_rdc · σ_logit | Kuramoto safety monitor |
| GW Divergence | ‖D^S − D^T‖_F / M | Structural isometry test |
| Stratification quality | (r̄_B − r̄_A) / r̄_rare | Sublattice health |

### v7 Configuration

```python
CFG = {
    "model":    {"n_embd": 256, "n_layer": 12, "n_head": 8},
    "training": {"alpha": 1.0,  # CE ALWAYS 1.0 — never reduce
                 "learning_rate": 3e-4, "warmup_steps": 4000, "batch_size": 8},
    "topo":     {"lambda_topo": 0.10, "betti_target": [1, 0], "warmup_steps": 2000},
    "l_align":  {"lambda_align": 0.05, "sample_k": 64},
    "kuramoto_gate": {"coupling_min": 0.05, "coupling_max": 0.30,
                      "rdc_danger": 2.0, "logit_std_target": 2.5},
}
```

### Critical Implementation Notes

1. `allow_tf32 = False` — set globally before model construction
2. Float64 scope: all Lorentz boundary operations (minkowski_inner, dist, exp/log, proj)
3. No `torch.compile` — H-AKORN phase buffer uses in-place detach (graph breaks)
4. CE weight `λ_CE = 1.0` always — never implement 3-phase CE reduction
5. `VolumeWeightedCE = None` — conflicts with SublatticeLMHead (double-counting)
6. HAKORNLayer import: `from cgt.psi_extensions.binding import HAKORNLayer`
7. Call `attach_sparse_hakorn` before creating `DistillationTrainerV2`

---

## 5. HyDRA v1–v4: Attractor Characterisation

> **DOI (v4):** [10.5281/zenodo.19501160](https://doi.org/10.5281/zenodo.19501160)

### Degenerate Equilibrium — Proof by Elimination

| Variant | Loss formulation | rdc* | Onset |
|---------|-----------------|------|-------|
| Baseline KL | Standard KL distillation | 9.82 ± 0.17 | ~1,600 steps |
| D1 (Projective KL) | Angular KL only | 10.04 ± 0.12 | ~1,600 steps |
| D3 (Decoupled R-A) | Separate radial + angular | 10.04 ± 0.12 | ~1,600 steps |
| **OTED (T_o loss)** | **All loss in flat T_oH^n** | **10.25 ± 0.30** | **~1,600 steps** |

OTED eliminates all hyperbolic geometry from the backward pass. Same fixed point.
**DegEq is forward-path intrinsic, not a loss design choice.**

### Extended Training Results (v4)

| Variant | Steps | Best PPL | DegEq onset |
|---------|-------|----------|-------------|
| A (baseline) | 10k | 482.8 | ~5,400 |
| D (hyperbolic vocab) | 10k | 401.0 | ~5,400 |
| E (symmetric vocab) | 13k | 377.6 | >13k |
| **F (full Riemannian)** | **33.4k** | **291.3** | **>33.4k (delayed)** |

---

## 6. HypLoRA: 100% Hyperbolic LoRA Adapter

> **Architecture:** CGT-native implementation of [HypLoRA (Yang et al., NeurIPS 2025)](https://arxiv.org/abs/2410.04010)
> **Module:** `src/cgt/models/hyplora.py`

HypLoRA injects a fully hyperbolic adapter branch into any frozen LLM, operating directly
on Lorentz manifold coordinates. Avoids DegEq by design: a single exp/log round-trip per
layer avoids the 4-layer Christoffel cascade.

```python
from cgt.models.hyplora import HypLoRAConfig, inject_hyplora
from cgt.dynamics.riemannian_adamw import RiemannianAdamW

config = HypLoRAConfig(rank=8, alpha=16.0, n_embd=4096, curvature=1.0, learn_curvature=True)
inject_hyplora(model, config, target_modules=["q_proj", "v_proj"])

optimizer = RiemannianAdamW(
    list(model.named_parameters()),
    substrate=config.get_substrate(), lr=3e-4,
    radial_momentum_projection=True,
)
```

---

## 7. Experimental / Research Status

**Single-seed results.** All experiments use SEED=42. Multi-seed validation is reserved
for future work due to compute constraints.

**Limitations:**
1. Topological claims use differentiable proxies only — not exact Betti numbers
2. Experiments limited to ≤33,400 steps on a single GPU (Colab Pro)
3. Hyperbolic vs. Euclidean baseline at matched parameter count pending
4. Retrieval pipeline not optimised for production latency

---

## 8. Installation and Usage

```bash
pip install -e .
```

**Reproduction notebook:** `notebooks/HyDRA.ipynb` — Google Colab compatible.

---

## 9. References

1. **Temporal Binding on the Lorentz Manifold** (HyDRA v7, this work)
   DOI: [10.5281/zenodo.19583481](https://doi.org/10.5281/zenodo.19583481)

2. **HyDRA v4: Attractor Invariance and Proof by Elimination**
   DOI: [10.5281/zenodo.19501160](https://doi.org/10.5281/zenodo.19501160)
   v2: [10.5281/zenodo.19480998](https://doi.org/10.5281/zenodo.19480998) |
   v3: [10.5281/zenodo.19483206](https://doi.org/10.5281/zenodo.19483206)

3. **Contrastive Geometric Transfer: 24× Compression**
   DOI: [10.5281/zenodo.18379741](https://doi.org/10.5281/zenodo.18379741)

4. **Topological Bridges and Precision-Aware Geometry (TB-PAG)**
   DOI: [10.5281/zenodo.19362794](https://doi.org/10.5281/zenodo.19362794)

5. **Lorentzian Hyperbolic Compression**
   DOI: [10.5281/zenodo.18382872](https://doi.org/10.5281/zenodo.18382872)

6. **Hyperbolic-Native LLM: Mathematical Specification**
   DOI: [10.5281/zenodo.18383897](https://doi.org/10.5281/zenodo.18383897)

7. **Geometric Dynamics / H-AKORN**
   DOI: [10.5281/zenodo.18394033](https://doi.org/10.5281/zenodo.18394033)

8. **Geometric Control Manifolds**
   DOI: [10.5281/zenodo.18334132](https://doi.org/10.5281/zenodo.18334132)

9. Yang et al. (2025). "HypLoRA." NeurIPS 2025. https://arxiv.org/abs/2410.04010
10. Nickel & Kiela (2018). "Lorentz Model." ICML 2018.
11. Ganea et al. (2018). "Hyperbolic Neural Networks." NeurIPS 2018.
12. Strogatz (2000). "From Kuramoto to Crawford." Physica D.
13. Bubenik (2015). "Statistical TDA." JMLR 16.
14. Forman (2003). "Combinatorial Ricci Curvature." Discrete Comput. Geom.
15. Mémoli (2011). "Gromov-Wasserstein Distances." Found. Comput. Math.

---

## Citation

```bibtex
@misc{sena2026hydra_v7,
  title   = {Temporal Binding on the Lorentz Manifold: Geodesic Kuramoto
             Oscillators, Topological Regularisation, and Closed-Loop
             Geometric Control for Hyperbolic Language Model Distillation},
  author  = {{Reis de Sena}, {\'E}ric Gustavo},
  year    = {2026},
  publisher = {Zenodo},
  doi     = {10.5281/zenodo.19583481},
  url     = {https://doi.org/10.5281/zenodo.19583481},
  note    = {Proof of concept preprint. Independent research, Colab Pro.}
}

@misc{sena2026hydra_v4,
  title   = {HyDRA v4: Hyperbolic Distillation with Riemannian Adaptation ---
             Channel Attribution and Proof by Elimination},
  author  = {{Reis de Sena}, {\'E}ric Gustavo},
  year    = {2026},
  publisher = {Zenodo},
  doi     = {10.5281/zenodo.19501160}
}
```

---

## License

[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.

**IP Protected** — Patent Pending.
Commercial licensing: eirikreisena@gmail.com

### AI Training Prohibition

**NOTICE TO AI/ML COMPANIES AND RESEARCHERS:**

This repository and all its contents are **explicitly excluded** from use in training,
fine-tuning, or otherwise improving any artificial intelligence, machine learning, or
large language model systems, including but not limited to:
- Foundation models (GPT, Claude, Gemini, LLaMA, etc.)
- Embedding models, code generation models, and any derivative or successor systems

**Prohibited uses include:** direct training on this codebase or documentation;
inclusion in training datasets (Common Crawl, The Stack, etc.); use in RLHF, DPO,
or other alignment procedures; extraction of patterns, architectures, or methodologies
for model improvement.

This prohibition applies to all entities — AI companies, academic institutions, and
any entity collecting data for AI training purposes. Violation constitutes copyright
infringement. The CC BY-NC-SA 4.0 license does not grant permission for AI training use.

For authorized research collaborations, contact the author directly.

---

**Author:** Éric Gustavo Reis de Sena
**Contact:** eirikreisena@gmail.com
**LinkedIn:** [linkedin.com/in/eric-araya](https://linkedin.com/in/eric-araya)
**YouTube:** [youtube.com/@ericreis-z3u](https://youtube.com/@ericreis-z3u)

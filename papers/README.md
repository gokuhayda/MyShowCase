# Geometric Consciousness & Neural Dynamics Research

**Author:** Éric Reis  
**Contact:** eirikreisena@gmail.com  
**Status:** Active Research (2025–2026)

---

## 🌌 Overview

This repository contains the theoretical frameworks, mathematical proofs, and experimental code implementations for a series of papers investigating the **Phenomenal Manifold Hypothesis (PMH)**.

This research program explores the intersection of **Hyperbolic Geometry**, **Riemannian Optimization**, **Topological Data Analysis**, and **Oscillatory Neural Dynamics** to model the structural organization of conscious experience and emergent complexity in artificial systems.

---

## 📂 Repository Structure

```
.
├── papers/
    ├── 01_Phenomenal_Manifold_Hypothesis.pdf
    ├── 02_Topological_Signature_Consciousness.pdf
    ├── 03_Lorentz_Manifold_Transformers.pdf
    ├── 04_Psi_Former_Architecture.pdf
    └── 05_Hyperbolic_NCA.pdf
    ├── src/
    │   ├── common/             # Shared geometric & dynamical libs (Lorentz, Kuramoto)
    │   ├── pmh_reconstruction/ # Code for Paper 1 (Manifold Reconstruction)
    │   ├── gw_topology/        # Code for Paper 2 (Color Ring Toy Model)
    │   ├── lmt_model/          # Code for Paper 3 (Transformer Implementation)
    │   ├── psi_former/         # Code for Paper 4 (K-FAC & Scaling)
    │   └── hyperbolic_nca/     # Code for Paper 5 (Pentagrid Automata)
    └── README.md
```

---

## 📄 Research Papers & Implementations

### 1. The Phenomenal Manifold Hypothesis (PMH)
**Subtitle:** A Geometric Framework Induced by Informational Dynamics  
**📅 Date:** November 2025 (Revised)

- **Abstract:** Proposes that conscious experience is a low-dimensional Riemannian manifold (Ψ) projected from high-dimensional neural parameter space. Defines the "Hybrid Metric" combining information geometry with three neural invariants: Integration (ℐ), Coherence (Γ), and Differentiation (Δ).

- **Key Contributions:**
  - Formal definition of the projection π: 𝒫(M₄) → Ψ
  - Differential predictions vs. IIT, GNWT, and Predictive Processing
  - Geometric interpretation of altered states (meditation, psychedelics)

- **💻 Code Status:** Coming Soon (Manifold learning pipeline & reconstruction procedures)

---

### 2. The Topological Signature of Consciousness
**Subtitle:** A Gromov-Wasserstein Framework for Neural-Phenomenal Alignment  
**📅 Date:** January 2026

- **Abstract:** Introduces a methodology using Gromov-Wasserstein (GW) Optimal Transport to align the metric space of neural states with the metric space of phenomenal distinctions. Uses Topological Data Analysis (Persistent Homology) to distinguish genuine structure from high-dimensional noise.

- **Key Experiment:** Color Ring Toy Model — Recovering circular topology (β₁ = 1) from noisy neural spike trains.

- **💻 Code Status:** Available (Simulation code for Color Ring model & GW Alignment)
  - **Path:** `/src/gw_topology`

---

### 3. Lorentz-Manifold Transformers (LMT)
**Subtitle:** A Geometric-Dynamical Framework for Hierarchical Representation Learning  
**📅 Date:** January 2026

- **Abstract:** Addresses the "Geometric Capacity Bottleneck" in standard Transformers. Integrates Hyperbolic Geometry (Lorentz model) for exponential capacity with H-AKOrN (Hyperbolic Artificial Kuramoto Oscillatory Neurons) for temporal binding.

- **Key Contributions:**
  - Manifold Capacity bounds proving exponential advantage (Ω(eʳ))
  - Geometric Frustration (ℱ) as a misalignment metric
  - Phase-transition failure modes in structural coherence

- **💻 Code Status:** In Progress (Synthetic validations & H-AKOrN dynamics)
  - **Path:** `/src/lmt_model`

---

### 4. The Ψ-Former
**Subtitle:** Topological Downward Causation via Riemannian Optimization  
**📅 Date:** January 2026

- **Abstract:** A deep learning architecture designed to explicitly instantiate PMH constraints. Features Topological Downward Causation, where the phenomenal manifold exerts causal influence on neural weights via Riemannian optimization (K-FAC).

- **Key Contributions:**
  - **Phenomenal Risk Score (PRS):** Ethical framework for AI phenomenology
  - **Architecture:** Hyperbolic Embeddings + Kuramoto Oscillators + Transformer-XL + K-FAC
  - **Scalability:** Inverse-free natural gradients and mean-field approximations

- **💻 Code Status:** In Progress (WikiText-103 training scripts & K-FAC implementation)
  - **Path:** `/src/psi_former`

---

### 5. Hyperbolic Neural Cellular Automata (H-NCA)
**Subtitle:** A Geometric Framework for Emergent Complexity  
**📅 Date:** January 7, 2026

- **Abstract:** Transposes Neural Cellular Automata to hyperbolic tessellations (Pentagrid {5,4}) to solve the temporal binding problem and capacity bottlenecks in distributed systems.

- **Key Experiment:** Emergent phase synchronization and geometric structure preservation in 61-cell hyperbolic grids.

- **💻 Code Status:** Available (Pentagrid setup & H-AKOrN dynamics)
  - **Path:** `/src/hyperbolic_nca`

---

## 🛠️ Core Technologies & Mathematical Basis

This research unifies several advanced mathematical frameworks. The shared codebase (`/src/common`) includes utilities for:

### 📐 Hyperbolic Geometry (Lorentz Model)
Implementation of the hyperboloid model ℍⁿ for numerical stability:

```
⟨x, y⟩_L = -x₀y₀ + x₁y₁ + ... + xₙyₙ
```

Includes Exponential/Logarithmic maps and covariant gradients.

### ⏱️ Oscillatory Dynamics (H-AKOrN)
Generalization of the Kuramoto model to curved manifolds for feature binding:

```
dθᵢ/dt = ωᵢ + (K/N) Σⱼ sin(θⱼ - θᵢ)
```

Used for enforcing Global Coherence (Γ).

### 🧬 Topological Data Analysis (TDA)
Tools for computing Persistent Homology (Betti numbers βₖ) and Gromov-Wasserstein distances to validate structural isomorphism between neural and phenomenal spaces.

### 📉 Riemannian Optimization
Implementation of Natural Gradient Descent and K-FAC approximations to ensure learning trajectories follow geodesic flows on the statistical manifold:

```
θₜ₊₁ = Expθₜ(-ηG(θₜ)⁻¹∇ℒ(θₜ))
```

---

## 🚀 Getting Started

Instructions for running the specific models will be located in their respective subdirectories.

### Prerequisites
- Python 3.9+
- PyTorch / JAX
- Geoopt (Manifold optimization)
- Gudhi / Ripser (TDA)
- NetworkX

### Installation
```bash
git clone https://github.com/your-username/Hyperbolic-NCA.git
cd Hyperbolic-NCA
pip install -r requirements.txt
```

---

## ⚖️ Citation

If you use this code or these papers in your research, please cite the specific work:

```bibtex
@article{reis2026pmh,
  title={The Phenomenal Manifold Hypothesis: A Geometric Framework Induced by Informational Dynamics},
  author={Reis, Éric},
  year={2026}
}
```

(See individual paper folders for specific BibTeX entries)

---

## ⚠️ Disclaimer

This is a research repository. The Ψ-Former and H-NCA are theoretical architectures designed to investigate the structural correlates of consciousness. The Phenomenal Risk Score (PRS) is a proposed ethical heuristic, not a definitive test for sentience.

---

**© 2026 Éric Reis. All rights reserved.**

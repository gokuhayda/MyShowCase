# Theoretical Foundations

This document describes the mathematical and theoretical foundations of the UGFT Simulator.

## Table of Contents

1. [Unified Geometric Field Theory](#unified-geometric-field-theory)
2. [Hyperbolic Geometry](#hyperbolic-geometry)
3. [H-NCA Architecture](#h-nca-architecture)
4. [H-AKORN Dynamics](#h-akorn-dynamics)
5. [Topological Analysis](#topological-analysis)
6. [Falsifiable Predictions](#falsifiable-predictions)

---

## 1. Unified Geometric Field Theory

### 1.1 Core Hypothesis

The **Unified Geometric Field Theory (UGFT)** proposes that cognitive organization emerges from the minimization of a geometric action functional:

$$\mathcal{S} = \mathcal{L}_{task} + \mathcal{L}_{geometry} + \mathcal{L}_{topology}$$

Where:
- **L_task**: Task-specific loss (e.g., synchronization objective)
- **L_geometry**: Curvature-weighted frustration penalty
- **L_topology**: Topological complexity regularization

### 1.2 Phenomenal Manifold Hypothesis

The **Phenomenal Manifold Hypothesis (PMH)** states that:

> A system's informational dynamics induce an effective Riemannian manifold Ψ whose metric combines local information (Fisher metric) with global topological invariants.

Key invariants:
- **Integration (I)**: Degree of information integration
- **Coherence (Γ)**: Phase synchronization measure
- **Differentiation (Δ)**: Repertoire of distinguishable states

### 1.3 Topological Downward Causation

The theory predicts **topological downward causation** — global topological properties (Betti numbers, connectivity) influence local dynamics through a feedback mechanism:

$$\frac{d\theta_i}{dt} = f_{local}(\theta_i, \theta_{N(i)}) \cdot g_{global}(R, \beta_0, \beta_1)$$

This is testable via counterfactual analysis: systems with vs. without topological feedback should exhibit measurably different trajectories.

---

## 2. Hyperbolic Geometry

### 2.1 Why Hyperbolic Space?

**The Geometric Capacity Bottleneck**: Euclidean spaces cannot embed hierarchical structures without exponential distortion.

| Space | Volume growth | Embedding capacity |
|-------|--------------|-------------------|
| Euclidean ℝⁿ | V(r) ~ rⁿ | Polynomial |
| Hyperbolic ℍⁿ | V(r) ~ e^{(n-1)r} | Exponential |

Hierarchical data (trees, taxonomies, neural architectures) grows exponentially with depth, matching hyperbolic geometry.

### 2.2 Hyperboloid Model (Lorentz Model)

We use the **Lorentz model** for computation:

$$\mathbb{H}^n = \{x \in \mathbb{R}^{n+1} : \langle x, x \rangle_{\mathcal{L}} = -1, x_0 > 0\}$$

The **Lorentz inner product**:

$$\langle u, v \rangle_{\mathcal{L}} = -u_0 v_0 + u_1 v_1 + \cdots + u_n v_n$$

**Geodesic distance**:

$$d_{\mathbb{H}}(p, q) = \text{arcosh}(-\langle p, q \rangle_{\mathcal{L}})$$

### 2.3 Poincaré Disk Model

For visualization, we project to the **Poincaré disk** 𝔻² = {z ∈ ℂ : |z| < 1}:

**Hyperboloid → Poincaré**:
$$\pi: (t, x, y) \mapsto \left(\frac{x}{1+t}, \frac{y}{1+t}\right)$$

**Poincaré → Hyperboloid**:
$$\pi^{-1}: (u, v) \mapsto \left(\frac{1+r^2}{1-r^2}, \frac{2u}{1-r^2}, \frac{2v}{1-r^2}\right)$$

where $r^2 = u^2 + v^2$.

### 2.4 Geodesics in Poincaré Disk

Geodesics appear as **circular arcs orthogonal to the boundary** (or diameters through origin).

For points p, q in the disk, the geodesic arc is computed by finding the unique circle passing through both points and intersecting the unit circle at right angles.

**Arc center and radius**:

Given p = (p₁, p₂) and q = (q₁, q₂):

$$c_x = \frac{(|p|^2 + 1)q_2 - (|q|^2 + 1)p_2}{2(p_1 q_2 - q_1 p_2)}$$

$$c_y = \frac{(|q|^2 + 1)p_1 - (|p|^2 + 1)q_1}{2(p_1 q_2 - q_1 p_2)}$$

$$r = \sqrt{(c_x - p_1)^2 + (c_y - p_2)^2}$$

---

## 3. H-NCA Architecture

### 3.1 Pentagrid Tessellation

The simulator uses a **{5,4} pentagrid tessellation** — a regular tessellation of the hyperbolic plane by pentagons with 4 meeting at each vertex.

**Properties**:
- Vertex degree: 4
- Face type: Regular pentagon
- Exponential growth: ~5 × 2^(layer-1) cells per layer

### 3.2 Cell State

Each cell maintains:

```
Cell {
    position: ℍ² point (hyperboloid coordinates)
    poincare: 𝔻² point (for visualization)
    phase: θ ∈ [0, 2π)
    omega: ω ∈ ℝ (natural frequency)
    neighbors: list of adjacent cell indices
}
```

### 3.3 Adjacency

Two cells are adjacent if their hyperbolic distance is below a threshold:

$$A_{ij} = \mathbb{1}[d_{\mathbb{H}}(p_i, p_j) < \tau]$$

The threshold τ is chosen to approximate the {5,4} tessellation connectivity.

---

## 4. H-AKORN Dynamics

### 4.1 Kuramoto Model

The classical **Kuramoto model** describes coupled oscillators:

$$\frac{d\theta_i}{dt} = \omega_i + \frac{K}{N}\sum_{j=1}^{N} \sin(\theta_j - \theta_i)$$

### 4.2 Hyperbolic Attention

**H-AKORN** (Hyperbolic Attentive Kuramoto Oscillator Recurrent Network) extends this with **geometry-aware coupling**:

$$\frac{d\theta_i}{dt} = \omega_i + \frac{K}{|N(i)|}\sum_{j \in N(i)} A_{ij}^{hyp} \sin(\theta_j - \theta_i)$$

The **hyperbolic attention weight**:

$$A_{ij}^{hyp} = \exp(-d_{\mathbb{H}}(p_i, p_j) \cdot |\kappa|)$$

This creates distance-dependent coupling strength, with nearby cells (in hyperbolic metric) influencing each other more.

### 4.3 Order Parameter

The **Kuramoto order parameter** measures global synchronization:

$$R = \left|\frac{1}{N}\sum_{k=1}^{N} e^{i\theta_k}\right| \in [0, 1]$$

- R ≈ 0: Incoherent (random phases)
- R ≈ 1: Fully synchronized

### 4.4 Critical Coupling

For a Lorentzian frequency distribution g(ω), the **critical coupling** for synchronization onset is:

$$K_c = \frac{2}{\pi g(0)} \approx 1.27$$

Below K_c: Incoherent state
Above K_c: Partial synchronization emerges

### 4.5 Topological Feedback (Optional)

When enabled, global coherence modulates local dynamics:

$$\frac{d\theta_i}{dt} = \omega_i + \frac{K}{|N(i)|}\sum_{j \in N(i)} A_{ij}^{hyp} \sin(\theta_j - \theta_i) \cdot (1 + \alpha(R - 0.5))$$

This implements **downward causation** from global order (R) to local evolution.

---

## 5. Topological Analysis

### 5.1 Cluster Detection

**Phase-coherent clusters** are detected via breadth-first search:

Two cells belong to the same cluster if:
1. They are adjacent (in the tessellation graph)
2. Their phase difference is small: min(|θᵢ - θⱼ|, 2π - |θᵢ - θⱼ|) < τ_phase

### 5.2 Betti Numbers

**Betti numbers** characterize the topology of the phase-coherent structure:

- **β₀**: Number of connected components (clusters)
- **β₁**: Number of "holes" (cycles in the network)
- **β₂**: Number of voids (0 for 2D manifolds)

**Euler characteristic approximation**:
$$\chi = V - E + F = \beta_0 - \beta_1 + \beta_2$$

For a graph: β₁ ≈ E - V + β₀

### 5.3 Integrated Information Proxy

**Φ_proxy** estimates integrated information:

$$\Phi_{proxy} = R \times \ln(N + 1)$$

This combines synchronization (R) with system size (N) as a simple proxy for information integration.

---

## 6. Falsifiable Predictions

### 6.1 Counterfactual Test

**Prediction**: Systems with topological feedback will exhibit different long-term dynamics than systems without feedback, given identical initial conditions.

**Test**: 
1. Initialize two systems with same random seed
2. Run one with feedback enabled, one without
3. Measure divergence: ΔR(t), ΔΦ(t), ΔClusters(t)
4. Predict: |ΔR| should increase over time

### 6.2 Geometric Capacity Test

**Prediction**: Hyperbolic embedding should maintain lower distortion than Euclidean embedding for hierarchical data.

**Metric**: Average distortion D = mean(|d_embedded - d_original| / d_original)

### 6.3 Phase Transition

**Prediction**: At K = K_c ≈ 1.27, the system should exhibit:
- Discontinuity in dR/dK
- Power-law cluster size distribution
- Critical slowing down (increased correlation time)

---

## References

1. Kuramoto, Y. (1984). Chemical Oscillations, Waves, and Turbulence.
2. Nickel, M., & Kiela, D. (2017). Poincaré Embeddings for Learning Hierarchical Representations.
3. Margenstern, M. (2007). Cellular Automata in Hyperbolic Spaces.
4. Tononi, G. (2008). Consciousness as Integrated Information: A Provisional Manifesto.
5. Chami, I., et al. (2019). Hyperbolic Graph Convolutional Neural Networks.

---

*Document version: 1.0.0 | Last updated: January 2026*

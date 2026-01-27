# Ψ UGFT Simulator

**Unified Geometric Field Theory • Hyperbolic Neural Cellular Automata**


[![WebGPU](https://img.shields.io/badge/WebGPU-Enabled-cyan.svg)](https://www.w3.org/TR/webgpu/)
[![Demo](https://img.shields.io/badge/Live-Demo-green.svg)](https://erickreis.github.io/ugft-simulator/)

<p align="center">
  <img src="assets/preview.gif" alt="UGFT Simulator Preview" width="600">
</p>

An interactive scientific simulator demonstrating **Geometric Intelligence** through Hyperbolic Neural Cellular Automata (H-NCA) with H-AKORN phase dynamics. This project provides a falsifiable implementation of the Unified Geometric Field Theory (UGFT) for self-organizing intelligence.

---

## 🎯 What This Simulator Does

| ✅ Does | ❌ Does Not |
|---------|-------------|
| Simulate geometric field dynamics | Simulate quantum hardware |
| Test stability conditions | Model physical particles |
| Expose topological feedback | Claim empirical consciousness |
| Demonstrate downward causation | Replace rigorous theory |

---

## 🧠 Scientific Foundation

This simulator implements concepts from:

- **Phenomenal Manifold Hypothesis (PMH)**: Cognitive states as structural invariants in negatively-curved geometry
- **H-NCA Architecture**: Neural Cellular Automata on {5,4} pentagrid tessellation
- **H-AKORN Dynamics**: Hyperbolic Attentive Kuramoto Oscillator Recurrent Networks
- **Topological Data Analysis**: Persistent homology for global structure detection

### Core Equation

The system minimizes a geometric action functional:

```
S = L_task + L_geometry + L_topology
```

Where:
- `L_task`: Synchronization loss (Kuramoto order parameter)
- `L_geometry`: Curvature-weighted frustration penalty
- `L_topology`: Topological complexity (Betti numbers)

---

## ✨ Features

### 🔬 Scientific Visualization
- **Real-time H-AKORN dynamics** on hyperbolic tessellation
- **Kuramoto order parameter R(t)** with temporal evolution
- **Betti numbers (β₀, β₁, β₂)** for topological analysis
- **Phase distribution histogram**
- **Φ_proxy** integrated information estimate

### 🎨 Visualization Modes
- **Phase θᵢ**: Color-coded phase angles
- **Clusters**: Detected coherent phase groups
- **Attention Aᵢⱼ**: Local synchronization strength
- **Geodesics**: True hyperbolic geodesic arcs

### ⚡ Counterfactual Analysis
- **Split-view comparison**: With vs Without topological feedback
- **Causal divergence metrics**: ΔR, ΔΦ, ΔClusters
- **Instability detection**: Automatic identification of divergence point
- Demonstrates **topological downward causation**

### 🖱️ Interactivity
- **Hover tooltips** with cell-specific metrics
- **Geodesic highlighting** on mouse over
- **Real-time parameter adjustment**
- **Responsive design** for different screen sizes

---

## 🚀 Quick Start

### Option 1: Direct Browser

Simply open `index.html` in a modern browser (Chrome 113+, Edge 113+, Firefox Nightly):

```bash
git clone https://github.com/erickreis/ugft-simulator.git
cd ugft-simulator
open index.html  # or: start index.html (Windows)
```

### Option 2: Local Server

```bash
# Using Python
python -m http.server 8000

# Using Node.js
npx serve .

# Then open http://localhost:8000
```

### Option 3: GitHub Pages

Visit the live demo: [https://erickreis.github.io/ugft-simulator/](https://erickreis.github.io/ugft-simulator/)

---

## 📁 Project Structure

```
ugft-simulator/
├── index.html              # Main simulator (State of the Art version)
├── README.md               # This file
├── LICENSE                 # MIT License
├── CITATION.cff            # Citation metadata
├── .gitignore
│
├── license/                # Research license pages
│   ├── index.html          # License purchase page
│   └── success/
│       └── index.html      # Post-purchase confirmation
│
├── versions/               # Alternative implementations
│   ├── webgpu-compute.html # WebGPU compute shader version
│   └── canvas-fallback.html # Pure Canvas 2D fallback
│
├── assets/
│   ├── preview.gif         # Demo animation
│   ├── screenshot.png      # Static preview
│   └── og-image.png        # Social media preview
│
└── docs/
    ├── THEORY.md           # Mathematical foundations
    ├── ARCHITECTURE.md     # Technical implementation details
    └── API.md              # JavaScript API reference
```

---

## ⚙️ Parameters

| Parameter | Symbol | Range | Default | Description |
|-----------|--------|-------|---------|-------------|
| Coupling | K | 0-10 | 2.5 | Kuramoto coupling strength |
| Time Step | ε | 0.01-0.2 | 0.05 | Integration step size |
| Curvature | κ | -2 to -0.1 | -1.0 | Hyperbolic curvature |
| Layers | - | 2-7 | 5 | Tessellation depth |

### Critical Values

- **K_c ≈ 1.27**: Critical coupling for synchronization transition (K_c = 2/πg(0) for Lorentzian distribution)
- **κ = -1**: Standard hyperbolic plane (Poincaré disk model)

---

## 🧪 Running Experiments

### 1. Phase Transition Study

1. Set K = 0.5 (subcritical)
2. Press Play and observe chaotic dynamics
3. Gradually increase K while running
4. Observe transition at K ≈ 1.3
5. Note cluster coalescence and R(t) jump

### 2. Counterfactual Causation Test

1. Reset the simulation
2. Click "⚡ Run Counterfactual"
3. Observe split-view comparison
4. Watch for divergence in ΔR and ΔΦ
5. Note instability point in "zombie" system

### 3. Topology Exploration

1. Set visualization to "Clusters"
2. Run simulation at K = 2.0
3. Watch β₀ decrease as clusters merge
4. Switch to "Geodesics" mode
5. Hover cells to see attention topology

---

## 📚 Related Publications

1. **The Phenomenal Manifold Hypothesis** - Geometric approach to consciousness modeling
2. **H-NCA: Hyperbolic Neural Cellular Automata** - Architecture foundations
3. **Unified Geometric Field Theory of Self-Organizing Intelligence** - Theoretical framework

---

## 🔧 Browser Requirements

| Browser | Version | Status |
|---------|---------|--------|
| Chrome | 113+ | ✅ Full support |
| Edge | 113+ | ✅ Full support |
| Firefox | Nightly | ⚠️ WebGPU flag required |
| Safari | 18+ | ⚠️ Limited WebGPU |

The simulator includes automatic Canvas 2D fallback for unsupported browsers.

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas for Contribution

- [ ] WebGPU render pipeline (eliminate CPU readback)
- [ ] Additional tessellations ({7,3}, {4,5})
- [ ] Lyapunov exponent calculation
- [ ] VR/WebXR visualization
- [ ] Performance benchmarks
- [ ] Accessibility improvements

---

## 📖 Citation

If you use this simulator in your research, please cite:

```bibtex
@software{reis2026ugft,
  author       = {Reis, Eric},
  title        = {UGFT Simulator: Unified Geometric Field Theory Implementation},
  year         = {2026},
  url          = {https://github.com/erickreis/ugft-simulator},
  version      = {1.0.0}
}
```

---

## 🙏 Acknowledgments

- Margenstern's pentagrid tessellation algorithms
- Kuramoto model foundations
- Poincaré disk model geometry
- WebGPU working group

---

<p align="center">
  <b>Built with 🧠 by <a href="https://github.com/erickreis">Eric Reis</a></b>
  <br>
  <i>Independent Researcher | Data Scientist | Geometric Intelligence</i>
</p>

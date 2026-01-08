# Hyperbolic Neural Cellular Automata (H-NCA)

[![arXiv](https://img.shields.io/badge/arXiv-2501.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2501.XXXXX)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/hyperbolic-nca/blob/main/notebooks/HNCA_Demo.ipynb)

> **A framework for simulating neural cellular automata on hyperbolic geometries with emergent synchronization dynamics**

## 🎯 Overview

H-NCA combines three powerful mathematical frameworks to model collective behavior in networks with hierarchical structure:

- **Hyperbolic Geometry** (Poincaré disk) for natural hierarchical embeddings
- **Cellular Automata** for local interaction rules
- **Kuramoto Dynamics** for emergent synchronization

This repository contains the complete implementation described in our paper, including interactive visualizations and reproducible experiments.

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/hyperbolic-nca.git
cd hyperbolic-nca

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Run Your First Simulation

```python
from src.hnca import HNCA
from src.pentagrid import build_pentagrid
import matplotlib.pyplot as plt

# Build hyperbolic lattice
G, coords = build_pentagrid(layers=3)

# Create H-NCA model
model = HNCA(G, coords, coupling_strength=0.3)

# Run synchronization
history = model.run(num_steps=500)

# Visualize results
model.plot_kuramoto_order(history)
plt.show()
```

**Expected output:**
- Kuramoto order parameter evolves from ~0.02 → 0.95
- Phases synchronize following hyperbolic distance hierarchy
- Generates publication-quality figures

## 📊 Simulation Examples

### 1. Phase Synchronization

The core simulation demonstrates emergent collective behavior in hyperbolic space:

```python
# Example: Synchronization on pentagrid lattice
python src/run_toy_model.py
```

**Results:**
- Initial disorder: R ≈ 0.02 (random phases)
- Final coherence: R ≈ 0.95 (synchronized)
- Convergence time: ~200-300 steps
- Critical coupling: K_c ≈ 0.15

![Synchronization Evolution](paper/figures/fig1_sync.pdf)

### 2. Hyperbolic Embedding

Visualize how the hierarchical structure affects dynamics:

```python
from src.visualization import plot_hyperbolic_embedding

# Plot Poincaré disk with node states
fig = plot_hyperbolic_embedding(
    coords, 
    phases=model.phases,
    title="H-NCA Hyperbolic Embedding"
)
```

![Hyperbolic Embedding](paper/figures/fig2_embed.pdf)

### 3. Custom Experiments

Explore parameter space with the interactive notebook:

```python
# Open in Colab (click badge above) or locally:
jupyter notebook notebooks/HNCA_Demo.ipynb
```

**Adjustable parameters:**
- Lattice depth (layers: 1-5)
- Coupling strength (K: 0.0-1.0)
- Natural frequencies (ω: uniform/normal)
- Initial conditions (random/clustered)

## 📖 Documentation

### Project Structure

```
hyperbolic-nca/
├── src/                        # Core implementation
│   ├── hnca.py                 # Main H-NCA class
│   ├── pentagrid.py            # Hyperbolic lattice generation
│   ├── lorentz.py              # Lorentz model geometry
│   ├── kuramoto.py             # Synchronization dynamics
│   └── visualization.py        # Plotting utilities
│
├── notebooks/                  # Interactive tutorials
│   ├── HNCA_Demo.ipynb         # ⭐ Start here!
│   ├── Paper_Figures.ipynb     # Reproduce paper results
│   └── Advanced_Examples.ipynb # Extended experiments
│
├── paper/                      # Academic paper
│   ├── hnca_paper_FINAL.tex    # LaTeX source
│   ├── hnca_paper_FINAL.pdf    # Compiled PDF
│   └── figures/                # High-quality figures
│
└── tests/                      # Unit tests
    ├── test_hnca.py
    ├── test_pentagrid.py
    └── test_lorentz.py
```

### Key Components

#### `HNCA` Class
Main simulation engine for hyperbolic cellular automata:

```python
model = HNCA(
    graph=G,              # NetworkX graph
    coords=coords,        # Hyperbolic coordinates (Nx2)
    coupling=0.3,         # Kuramoto coupling strength
    natural_freq=None     # Natural frequencies (optional)
)

# Run simulation
history = model.run(num_steps=500, dt=0.05)

# Access results
print(f"Final order: {model.order_parameter():.3f}")
print(f"Phases: {model.phases}")
```

#### `build_pentagrid()`
Generate hyperbolic pentagonal lattice:

```python
from src.pentagrid import build_pentagrid

G, coords = build_pentagrid(
    layers=3,           # Depth of lattice (1-5)
    return_coords=True  # Return Poincaré coordinates
)

print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
```

**Lattice sizes:**
- Layers 1: 6 nodes
- Layers 2: 16 nodes
- Layers 3: 61 nodes
- Layers 4: 151 nodes
- Layers 5: 306 nodes

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=html

# Specific test module
pytest tests/test_hnca.py -v
```

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{reis2025hnca,
  title={Hyperbolic Neural Cellular Automata: Emergent Synchronization in Hierarchical Spaces},
  author={Reis, Eric},
  journal={arXiv preprint arXiv:2501.XXXXX},
  year={2025}
}
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Areas for contribution:**
- Additional lattice geometries (hexagonal, {7,3} tiling)
- Alternative dynamics (Ising, voter models)
- Performance optimization (GPU acceleration)
- Extended documentation and tutorials

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This work builds upon:
- Mordvintsev et al. (2020) - Growing Neural Cellular Automata
- Nickel & Kiela (2017) - Poincaré Embeddings
- Kuramoto (1975) - Chemical Oscillations, Waves, and Turbulence

## 📧 Contact

**Author:** Eric Reis  
**Email:** eireikreisena@gmail.com  
**GitHub:** [@yourusername](https://github.com/yourusername)

---

⭐ **Star this repo if you find it useful!**

📚 **Read the paper:** [arXiv:2501.XXXXX](https://arxiv.org/abs/2501.XXXXX)

💻 **Try it now:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/hyperbolic-nca/blob/main/notebooks/HNCA_Demo.ipynb)

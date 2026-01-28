# 🚀 H-AKORN Integration - What's New

This version of the CGT project includes the **H-AKORN (Hyperbolic Attention with Kuramoto Oscillator Regularized Networks)** transformer architecture.

## New Features

### 🌀 H-AKORN Transformer

Complete implementation of a novel transformer architecture that combines:
- **Hyperbolic Geometry**: Attention based on geodesic distances
- **Kuramoto Oscillators**: Phase synchronization dynamics for attention heads
- **Adaptive Coupling**: Dynamic interaction between oscillators

**Location**: `src/cgt/models/hakorn/`

### 📦 New Modules

```
src/cgt/models/hakorn/
├── __init__.py              # Module exports
├── phase_dynamics.py        # Kuramoto phase evolution (189 lines)
├── coupling.py              # Adaptive coupling matrices (251 lines)
├── attention.py             # Hyperbolic attention (361 lines)
├── layer.py                 # Complete transformer layer (287 lines)
├── model.py                 # Full model (421 lines)
└── losses.py                # H-AKORN specific losses (293 lines)
```

### 🚂 Training & Examples

- `scripts/train_hakorn.py` - Complete training script
- `examples/hakorn_cgt_integration.py` - Integration example
- `notebooks/hakorn_notebook.ipynb` - Interactive demo with visualizations

### 🧪 Tests

- `tests/test_installation_hakorn.py` - Installation and import tests

---

## Quick Start with H-AKORN

### Installation

```bash
pip install -e .
```

### Basic Usage

```python
from cgt.geometry.lorentz_hardened import LorentzSubstrateHardened
from cgt.models.hakorn import HAKORNTransformer, HAKORNLoss

# Create hyperbolic substrate
substrate = LorentzSubstrateHardened(
    manifold_dim=768,
    curvature=-1.0,
)

# Create H-AKORN model
model = HAKORNTransformer(
    vocab_size=50257,
    d_model=768,
    num_layers=12,
    num_heads=12,
    substrate=substrate,
    coupling_strength=1.0,
)

# Forward pass
import torch
input_ids = torch.randint(0, 50257, (4, 128))
output = model(input_ids, return_dict=True)

print(f"Logits: {output['logits'].shape}")
print(f"Order parameter: {output['all_order_params'][0].mean():.4f}")
```

### Training

```bash
python scripts/train_hakorn.py \
    --vocab_size 50257 \
    --d_model 768 \
    --num_layers 12 \
    --batch_size 32 \
    --num_epochs 10
```

---

## Mathematical Framework

### Kuramoto Dynamics

Phase evolution for each attention head:

```
dθ_i/dt = ω_i + (K/N) Σ_j A_ij sin(θ_j - θ_i)
```

Where:
- θ_i: Phase of oscillator i
- ω_i: Natural frequency (learnable)
- K: Coupling strength
- A_ij: Adaptive coupling matrix
- N: Number of heads

### Order Parameter

Measures synchronization:

```
r = |⟨e^(iθ)⟩| = |(1/N) Σ_j e^(iθ_j)| ∈ [0, 1]
```

### Hyperbolic Attention

Attention scores based on hyperbolic distance:

```
Score_ij = exp(-d_H(q_i, k_j)/τ) · cos(θ_i - θ_j)
```

### Loss Function

Combined H-AKORN loss:

```
L_total = L_LM + λ_sync·(1-r)² + λ_var·(-Var(r_l))
```

---

## Architecture Overview

```
Input Tokens
    ↓
Embeddings (Token + Position)
    ↓
┌─────────────────────────────────┐
│      H-AKORN Transformer        │
│                                 │
│  ┌──────────────────────────┐  │
│  │   H-AKORN Layer          │  │
│  │  ┌────────────────────┐  │  │
│  │  │ Hyperbolic Attention│  │  │
│  │  │  + Kuramoto Phase   │  │  │
│  │  └────────────────────┘  │  │
│  │  ┌────────────────────┐  │  │
│  │  │  Feed-Forward      │  │  │
│  │  └────────────────────┘  │  │
│  └──────────────────────────┘  │
│               ⋮                 │
│  [Additional Layers...]         │
└─────────────────────────────────┘
    ↓
Language Modeling Head
    ↓
Output Logits
```

---

## Key Hyperparameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `coupling_strength` | Kuramoto coupling K | 1.0 | [0.1, 5.0] |
| `use_phase_modulation` | Enable phase modulation | True | bool |
| `curvature` | Hyperbolic curvature | -1.0 | K < 0 |
| `lambda_sync` | Sync regularization | 0.1 | [0.01, 1.0] |
| `lambda_variance` | Variance regularization | 0.05 | [0.01, 0.5] |

---

## Examples

### Example 1: Basic Model

```python
from cgt.models.hakorn import HAKORNTransformer

model = HAKORNTransformer(vocab_size=50257, d_model=768)
```

### Example 2: With Substrate

```python
from cgt.geometry.lorentz_hardened import LorentzSubstrateHardened
from cgt.models.hakorn import HAKORNTransformer

substrate = LorentzSubstrateHardened(manifold_dim=768, curvature=-1.0)
model = HAKORNTransformer(vocab_size=50257, substrate=substrate)
```

### Example 3: Complete Training

```python
from cgt.models.hakorn import HAKORNTransformer, HAKORNLoss
from torch.optim import AdamW

model = HAKORNTransformer(vocab_size=50257)
criterion = HAKORNLoss(lambda_sync=0.1, lambda_variance=0.05)
optimizer = AdamW(model.parameters(), lr=1e-4)

for batch in dataloader:
    output = model(batch['input_ids'], labels=batch['labels'], return_dict=True)
    loss_dict = criterion(output['loss'], output['all_order_params'])
    
    optimizer.zero_grad()
    loss_dict['total'].backward()
    optimizer.step()
```

---

## Documentation

- **Original CGT Documentation**: See `README.md` (original) for core CGT functionality
- **H-AKORN Notebook**: `notebooks/hakorn_notebook.ipynb` for interactive demos
- **Integration Example**: `examples/hakorn_cgt_integration.py` for complete integration

---

## Testing

```bash
# Test H-AKORN installation
pytest tests/test_installation_hakorn.py -v

# Test all CGT components
pytest tests/ -v
```

---

## Performance Metrics

### Model Sizes

| Configuration | Parameters | Memory (FP32) |
|--------------|------------|---------------|
| H-AKORN Small | ~30M | ~2GB |
| H-AKORN Medium | ~124M | ~8GB |
| H-AKORN Large | ~350M | ~20GB |

### Order Parameter Evolution

- **Initial**: r ≈ 0.3-0.5 (random phases)
- **During Training**: r → 0.7-0.9 (partial sync)
- **Deeper Layers**: Higher r values (more synchronization)

---

## Citation

If you use H-AKORN in your research, please cite:

```bibtex
@software{hakorn2026,
  author = {Éric Gustavo Reis de Sena},
  title = {H-AKORN: Hyperbolic Attention with Kuramoto Oscillator Regularized Networks},
  year = {2026},
  url = {https://github.com/eric-araya/cgt}
}
```

---

## What's New - Summary

✅ **H-AKORN Transformer** - Complete implementation (7 modules, ~2000 lines)  
✅ **Kuramoto Dynamics** - Phase synchronization for attention heads  
✅ **Adaptive Coupling** - Dynamic oscillator interactions  
✅ **Training Scripts** - Full pipeline with H-AKORN losses  
✅ **Examples** - Integration example + interactive notebook  
✅ **Tests** - Installation and functionality tests  
✅ **Documentation** - This file + notebook + examples  

---

## Original CGT Documentation

For the original CGT framework documentation (contrastive geometric transfer, Lorentz geometry, etc.), see the sections below or the original `README.md`.

---

**Author**: Éric Gustavo Reis de Sena  
**Contact**: eirikreisena@gmail.com  
**Version**: 0.3.0 (with H-AKORN)  
**License**: CC-BY-NC-SA-4.0

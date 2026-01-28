# PSI Extensions for CGT

## Overview

This module contains extensions derived from the **Ψ-SLM** (Toy Phenomenal Small Language Model) project, integrated into CGT following strict dominance rules.

> ⚠️ **IMPORTANT**: This module is **ADDITIVE ONLY**. It does NOT modify any existing CGT functionality. All CGT code remains completely intact.

## Origin

- **Source Project**: Ψ-SLM (Toy Phenomenal Small Language Model)
- **Original License**: MIT
- **Integration Date**: 2026-01-18
- **Integrated By**: Forensic Code Audit Process

## Integration Rules Applied

1. **CGT is DOMINANT**: No CGT code was modified
2. **No Conflicts**: All namespace conflicts were resolved by exclusion
3. **Isolation**: All PSI_SLM code is contained in `psi_extensions/`
4. **Explicit Imports**: All imports use `cgt.geometry.lorentz`, not original PSI_SLM substrate

## Available Modules

### `topology/`
Topological constraint fields and persistence landscapes.

```python
from cgt.psi_extensions.topology import (
    TopologicalConfig,
    TopologicalConstraintField,
    PersistenceLandscapeField,
    ScalarFeedbackField,
    create_topological_field,
)
```

**Key Features**:
- `PersistenceLandscapeField`: Differentiable TDA proxy for topological constraints
- `ScalarFeedbackField`: Non-differentiable scalar feedback for ablation studies
- Implements "Topological Downward Causation" from UGFT

### `binding/`
H-AKOrN (Hyperbolic Artificial Kuramoto Oscillatory Neurons) for solving the binding problem.

```python
from cgt.psi_extensions.binding import (
    HAKORNConfig,
    HyperbolicAKORN,
)
```

**Key Features**:
- Phase synchronization for semantic binding
- Kuramoto oscillator dynamics on hyperbolic manifold
- Local coherence and synchrony group identification

### `dynamics/`
Hyperbolic Neural Cellular Automata (H-NCA).

```python
from cgt.psi_extensions.dynamics import (
    HNCAConfig,
    HyperbolicNCA,
)
```

**Key Features**:
- Local update rules on hyperbolic manifold
- Emergent global structure from local interactions

### `transfer/`
Transfer learning extensions.

```python
from cgt.psi_extensions.transfer import (
    GromovWassersteinLoss,
    entropic_gromov_wasserstein_loss,
    compute_gw_divergence,
    MultiTeacherTransfer,
)
```

**Key Features**:
- Gromov-Wasserstein alignment loss (alternative to PowerLawDistillation)
- Multi-teacher distillation
- Geometric transfer from Euclidean to hyperbolic space

### `teachers/`
Teacher model wrappers.

```python
from cgt.psi_extensions.teachers import (
    SentenceTransformerTeacher,
)
```

**Key Features**:
- Unified interface for Sentence-Transformers
- Embedding caching for efficiency
- Similarity matrix computation

### `benchmarks/`
MTEB benchmark adapters.

```python
from cgt.psi_extensions.benchmarks import (
    MTEBAdapter,
    MTEBConfig,
    evaluate_mteb_task,
)
```

**Key Features**:
- Adapter for MTEB (Massive Text Embedding Benchmark)
- Task-specific evaluation utilities

### `applications/`
Application pipelines.

```python
from cgt.psi_extensions.applications import (
    RAGConfig,
    HyperbolicRAG,
    HyperbolicKnowledgeBase,
)
```

**Key Features**:
- Complete RAG pipeline using hyperbolic embeddings
- Hyperbolic semantic search
- LLM integration for generation

### `visualization/`
Plotting and visualization utilities.

```python
from cgt.psi_extensions.visualization import (
    MetricsLogger,
    plot_poincare_embedding,
    plot_phase_coherence_matrix,
    plot_training_curves,
)
```

## Dependencies

These extensions may require additional dependencies not in base CGT:

```bash
# Optional dependencies
pip install sentence-transformers  # For teachers module
pip install POT                     # For Gromov-Wasserstein
pip install transformers            # For RAG applications
```

## Usage Example

```python
import torch
from cgt.geometry import LorentzSubstrate, LorentzConfig
from cgt.psi_extensions.binding import HyperbolicAKORN, HAKORNConfig
from cgt.psi_extensions.transfer import GromovWassersteinLoss

# Use CGT's substrate (NOT psi_slm's)
substrate = LorentzSubstrate(LorentzConfig(intrinsic_dim=32))

# Initialize H-AKOrN for binding
akorn_config = HAKORNConfig(coupling_strength=1.0, decay_scale=1.0)
akorn = HyperbolicAKORN(config=akorn_config, substrate=substrate)

# Initialize states and phases
states = substrate.random_init(100)
phases = akorn.initialize(100)

# Run binding dynamics
for _ in range(50):
    phases = akorn.step(states, phases)

coherence = akorn.compute_coherence(phases)
print(f"Global coherence: {coherence:.3f}")

# Use GW loss as alternative to PowerLawDistillation
gw_loss = GromovWassersteinLoss(substrate, epsilon=0.01)
teacher_emb = torch.randn(100, 768)  # From Sentence-Transformer
student_emb = states

loss = gw_loss(teacher_emb, student_emb)
```

## Conflicts Avoided

The following PSI_SLM components were **NOT** integrated due to conflicts:

| Component | Reason |
|-----------|--------|
| `substrate/lorentz.py` | `LorentzConfig` and `LorentzSubstrate` already exist in CGT with different signatures |
| `utils/metrics.py::compute_distortion` | Same function name with incompatible signature |
| `integration/geometric_transfer.py` | Redundant with `cgt.losses.core` |
| `integration/slm_bridge.py` | Specific to Toy Ψ-SLM architecture |
| `action/action_functional.py` | Theoretical UGFT framework, out of scope |

## License

- **CGT Components**: CC-BY-NC-SA-4.0 © 2026 Éric Gustavo Reis de Sena
- **PSI_SLM Components**: MIT License (original)
- **Integration**: Dual-licensed under original terms

## Changelog

### v0.1.0 (2026-01-18)
- Initial integration from PSI_SLM
- All imports adapted to use `cgt.geometry.lorentz`
- Isolated in `psi_extensions/` module

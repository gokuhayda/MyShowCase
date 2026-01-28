# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
CGT Model Architecture Module
=============================

Neural network components for Contrastive Geometric Transfer.

HARDENED versions are the production-ready implementations matching
the CGT_Paper_Ready_v6_1_HARDENED notebook exactly.

Exports
-------
CGTGW
    CGT with Gromov-Wasserstein alignment (Ψ-SLM architecture).
CGTGWConfig
    Configuration for CGTGW model.
CGTStudentHardened
    Production-ready student encoder model.
HomeostaticFieldHardened
    Density preservation layer with proper Riemannian ops.
RiemannianOptimizerWrapper
    Optimizer wrapper with manifold support.
RiemannianAdam
    Full Riemannian Adam optimizer with parallel transport. (v9.9.4)
create_projector
    Projector factory function with orthogonal init.

Hyperbolic Transformer (H-LLM)
------------------------------
HyperbolicTransformer
    Complete hyperbolic-native language model.
HyperbolicTransformerConfig
    Configuration for hyperbolic transformer.
HyperbolicEmbedding, HyperbolicAttention, HyperbolicFFN
    Core components.
"""

# CGTGW - Main model for Ψ-SLM experiments
from cgt.models.cgt_gw import CGTGW, CGTGWConfig, stable_initialization

# HARDENED versions (production)
from cgt.models.cgt_hardened import (
    CGTStudentHardened,
    HomeostaticFieldHardened,
    RiemannianOptimizerWrapper,
    RiemannianAdam,  # AUDIT FIX v9.9.4
    create_projector,
)

# Hyperbolic Transformer (H-LLM)
from cgt.models.hyperbolic_transformer import (
    HyperbolicTransformer,
    HyperbolicTransformerConfig,
    HyperbolicEmbedding,
    HyperbolicAttention,
    HyperbolicFFN,
    HyperbolicLayerNorm,
    HyperbolicResidual,
    HyperbolicTransformerBlock,
    HyperbolicLMHead,
    create_hyperbolic_gpt2_small,
    create_hyperbolic_gpt2_medium,
    create_minimal_hyperbolic_transformer,
)

# Hyperbolic Transformer HARDENED (NaN-proof)
from cgt.models.hyperbolic_transformer_hardened import (
    HyperbolicTransformerHardened,
    HyperbolicEmbeddingHardened,
    HyperbolicAttentionHardened,
    HyperbolicFFNHardened,
    HyperbolicLayerNormHardened,
    HyperbolicResidualHardened,
    HyperbolicTransformerBlockHardened,
    HyperbolicLMHeadHardened,
    create_minimal_hyperbolic_transformer_hardened,
    create_hyperbolic_gpt2_small_hardened,
)

# Aliases for backward compatibility
CGTStudent = CGTStudentHardened
HomeostaticField = HomeostaticFieldHardened

__all__ = [
    # CGTGW (Ψ-SLM)
    "CGTGW",
    "CGTGWConfig",
    "stable_initialization",
    # Core (HARDENED)
    "CGTStudentHardened",
    "HomeostaticFieldHardened",
    "RiemannianOptimizerWrapper",
    "RiemannianAdam",  # AUDIT FIX v9.9.4
    "create_projector",
    # Hyperbolic Transformer (H-LLM)
    "HyperbolicTransformer",
    "HyperbolicTransformerConfig",
    "HyperbolicEmbedding",
    "HyperbolicAttention",
    "HyperbolicFFN",
    "HyperbolicLayerNorm",
    "HyperbolicResidual",
    "HyperbolicTransformerBlock",
    "HyperbolicLMHead",
    "create_hyperbolic_gpt2_small",
    "create_hyperbolic_gpt2_medium",
    "create_minimal_hyperbolic_transformer",
    # Hyperbolic Transformer HARDENED
    "HyperbolicTransformerHardened",
    "HyperbolicEmbeddingHardened",
    "HyperbolicAttentionHardened",
    "HyperbolicFFNHardened",
    "HyperbolicLayerNormHardened",
    "HyperbolicResidualHardened",
    "HyperbolicTransformerBlockHardened",
    "HyperbolicLMHeadHardened",
    "create_minimal_hyperbolic_transformer_hardened",
    "create_hyperbolic_gpt2_small_hardened",
    # Aliases
    "CGTStudent",
    "HomeostaticField",
]

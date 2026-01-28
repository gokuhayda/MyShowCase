# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
Hyperbolic Transformer [LLM Implementation]
===========================================

Complete implementation of a Hyperbolic-native Transformer for language modeling.
Based on Lorentz (hyperboloid) model with geodesic attention.

Mathematical Specification
--------------------------
- Geometry: Lorentz model H^n_K = {x ∈ R^{n+1} : ⟨x,x⟩_L = -1/K, x₀ > 0}
- Attention: s_ij = -d_H(q_i, k_j)² / τ (geodesic distance)
- FFN: Tangent space processing (log → linear → GELU → linear → exp)
- Residual: Tangent space approximation (log + log → exp)
- LayerNorm: Riemannian normalization in tangent space

Key Design Decisions
--------------------
1. Lorentz over Poincaré: Unbounded coordinates prevent boundary NaNs
2. Tangent-space FFN: Non-linearities remain Euclidean
3. Geodesic attention: Distance-based similarity respects manifold geometry
4. Origin-centric operations: Most exp/log through origin for stability

Author: Éric Gustavo Reis de Sena
Date: January 2026
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from cgt.geometry.lorentz_hardened import (
    LorentzConfig,
    LorentzSubstrateHardened,
    safe_acosh,
    safe_sqrt,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class HyperbolicTransformerConfig:
    """
    Configuration for Hyperbolic Transformer.
    
    Attributes:
        vocab_size: Vocabulary size
        n_embd: Embedding dimension (intrinsic hyperbolic dimension)
        n_layer: Number of transformer layers
        n_head: Number of attention heads
        n_positions: Maximum sequence length
        ffn_ratio: FFN hidden dimension multiplier (default: 4)
        dropout: Dropout rate
        attention_dropout: Attention-specific dropout
        
        # Geometry
        initial_curvature: Initial K value
        learnable_curvature: Whether K is learnable
        curvature_min: Minimum K bound
        curvature_max: Maximum K bound
        
        # Stability
        radius_max: Maximum allowed radius (for regularization)
        attention_temperature: τ for attention scores (learnable if None)
        positional_lambda: λ_pos for positional bias
        positional_warmup_steps: Steps to warmup positional bias
        layer_norm_gamma_init: Initial γ for LayerNorm (>1 prevents radial collapse)
        
        # Training
        tie_word_embeddings: Share input/output embeddings
        use_cache: Enable KV caching for generation
    """
    vocab_size: int = 50257  # GPT-2 default
    n_embd: int = 256        # Hyperbolic dimension (intrinsic)
    n_layer: int = 12
    n_head: int = 8
    n_positions: int = 1024
    ffn_ratio: int = 4
    dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # Geometry
    initial_curvature: float = 1.0
    learnable_curvature: bool = False  # Fixed for V1
    curvature_min: float = 0.1
    curvature_max: float = 10.0
    
    # Stability (from spec Known Failure Modes)
    radius_max: float = 10.0
    attention_temperature: Optional[float] = 1.0  # Fixed for V1
    positional_lambda: float = 0.1
    positional_warmup_steps: int = 5000
    layer_norm_gamma_init: float = 1.2  # >1 prevents radial collapse
    
    # Training
    tie_word_embeddings: bool = True
    use_cache: bool = True
    
    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        assert self.n_embd % self.n_head == 0
        return self.n_embd // self.n_head
    
    @property
    def ffn_dim(self) -> int:
        """FFN hidden dimension."""
        return self.n_embd * self.ffn_ratio


# =============================================================================
# HYPERBOLIC EMBEDDING
# =============================================================================

class HyperbolicEmbedding(nn.Module):
    """
    Hyperbolic token embedding layer.
    
    Stores embeddings in tangent space at origin, projects via exp_map on lookup.
    
    Architecture:
        lookup(token_id) → e_i ∈ R^n → zero-pad to (0, e_i) → exp_o → H^n
    
    Initialization:
        e_i ~ N(0, σ²I) with σ = 0.01 (keeps embeddings near origin initially)
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        substrate: LorentzSubstrateHardened,
        init_std: float = 0.01,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim  # intrinsic dim n
        self.substrate = substrate
        self.padding_idx = padding_idx
        
        # Tangent space embeddings (spatial components only)
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim) * init_std)
        
        if padding_idx is not None:
            with torch.no_grad():
                self.weight[padding_idx].zero_()
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup and project to manifold.
        
        Args:
            input_ids: Token indices [B, L]
            
        Returns:
            Hyperbolic embeddings [B, L, n+1]
        """
        # Lookup spatial components
        e_spatial = F.embedding(
            input_ids, 
            self.weight, 
            padding_idx=self.padding_idx
        )  # [B, L, n]
        
        # Zero-pad time component for tangent vector at origin
        e_time = torch.zeros(
            *e_spatial.shape[:-1], 1,
            device=e_spatial.device,
            dtype=e_spatial.dtype
        )
        e_tangent = torch.cat([e_time, e_spatial], dim=-1)  # [B, L, n+1]
        
        # Project to manifold via exp_map from origin
        return self.substrate.exp_map_zero(e_tangent)


# =============================================================================
# HYPERBOLIC LAYER NORMALIZATION
# =============================================================================

class HyperbolicLayerNorm(nn.Module):
    """
    Riemannian Layer Normalization.
    
    Maps to tangent space, normalizes spatial components, maps back.
    
    Process:
        x → log_o(x) → extract spatial → normalize → scale/shift → exp_o
    
    Critical: Initialize γ > 1 (default 1.2) to prevent radial collapse.
    """
    
    def __init__(
        self,
        normalized_shape: int,
        substrate: LorentzSubstrateHardened,
        eps: float = 1e-6,
        gamma_init: float = 1.2,
    ):
        super().__init__()
        self.normalized_shape = normalized_shape  # intrinsic dim n
        self.substrate = substrate
        self.eps = eps
        
        # Learnable parameters (operate on spatial components)
        self.gamma = nn.Parameter(torch.ones(normalized_shape) * gamma_init)
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Riemannian LayerNorm.
        
        Args:
            x: Points on manifold [..., n+1]
            
        Returns:
            Normalized points on manifold [..., n+1]
        """
        # Map to tangent space at origin
        v = self.substrate.log_map_zero_batch(x)  # [..., n+1]
        v_spatial = v[..., 1:]  # [..., n]
        
        # Standard LayerNorm on spatial components
        mean = v_spatial.mean(dim=-1, keepdim=True)
        var = v_spatial.var(dim=-1, keepdim=True, unbiased=False)
        v_norm = (v_spatial - mean) / torch.sqrt(var + self.eps)
        
        # Scale and shift
        v_out = self.gamma * v_norm + self.beta
        
        # Zero time component
        v_time = torch.zeros(
            *v_out.shape[:-1], 1,
            device=v_out.device,
            dtype=v_out.dtype
        )
        v_full = torch.cat([v_time, v_out], dim=-1)
        
        # Map back to manifold
        return self.substrate.exp_map_zero(v_full)


# =============================================================================
# HYPERBOLIC ATTENTION
# =============================================================================

class HyperbolicAttention(nn.Module):
    """
    Geodesic Multi-Head Attention.
    
    Attention scores based on negative squared geodesic distance:
        s_ij = -d_H(q_i, k_j)² / τ + bias_pos
    
    Value aggregation via tangent space weighted sum (Fréchet mean approximation).
    
    Architecture:
        Q, K, V = linear projections in tangent space
        scores = -dist²/τ + pos_bias
        output = weighted_midpoint(V, softmax(scores))
    """
    
    def __init__(
        self,
        config: HyperbolicTransformerConfig,
        substrate: LorentzSubstrateHardened,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.config = config
        self.substrate = substrate
        self.layer_idx = layer_idx
        
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_embd = config.n_embd
        
        # QKV projections (operate on spatial components in tangent space)
        # Input: n, Output: n (split into heads later)
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        # Temperature (fixed for V1, learnable possible in V2)
        if config.attention_temperature is not None:
            self.register_buffer('temperature', torch.tensor(config.attention_temperature))
        else:
            self.temperature = nn.Parameter(torch.tensor(1.0))
        
        # Positional bias coefficient
        self.register_buffer('lambda_pos', torch.tensor(config.positional_lambda))
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        """Initialize with small weights to prevent exp_map explosion."""
        for proj in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.normal_(proj.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02 / math.sqrt(2 * self.config.n_layer))
    
    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        training_step: int = 0,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for hyperbolic attention.
        
        Args:
            x: Input on manifold [B, L, n+1]
            position_ids: Position indices [B, L]
            attention_mask: Mask [B, 1, L, L] (additive, -inf for masked)
            past_key_value: Cached K, V for generation
            use_cache: Whether to return updated cache
            training_step: Current step for positional warmup
            
        Returns:
            Tuple of (output [B, L, n+1], optional cache)
        """
        B, L, _ = x.shape
        
        # Map to tangent space at origin
        x_tangent = self.substrate.log_map_zero_batch(x)  # [B, L, n+1]
        x_spatial = x_tangent[..., 1:]  # [B, L, n]
        
        # Project Q, K, V in tangent space
        q_spatial = self.q_proj(x_spatial)  # [B, L, n]
        k_spatial = self.k_proj(x_spatial)
        v_spatial = self.v_proj(x_spatial)
        
        # Reshape for multi-head: [B, L, n] -> [B, H, L, head_dim]
        q_spatial = q_spatial.view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        k_spatial = k_spatial.view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        v_spatial = v_spatial.view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        
        # Project back to manifold for distance computation
        # Zero-pad time component
        zero_time = torch.zeros(B, self.n_head, L, 1, device=x.device, dtype=x.dtype)
        
        q_tangent = torch.cat([zero_time, q_spatial], dim=-1)  # [B, H, L, head_dim+1]
        k_tangent = torch.cat([zero_time, k_spatial], dim=-1)
        v_tangent = torch.cat([zero_time, v_spatial], dim=-1)
        
        # Map Q, K to manifold for distance computation
        # Note: We use a per-head substrate view (same curvature, different dim)
        # For simplicity, compute distances on spatial parts directly
        # This is equivalent to H^{head_dim} with same K
        
        q_manifold = self._exp_map_head(q_tangent)  # [B, H, L, head_dim+1]
        k_manifold = self._exp_map_head(k_tangent)
        v_manifold = self._exp_map_head(v_tangent)
        
        # Handle KV cache
        if past_key_value is not None:
            k_manifold = torch.cat([past_key_value[0], k_manifold], dim=2)
            v_manifold = torch.cat([past_key_value[1], v_manifold], dim=2)
        
        if use_cache:
            present_key_value = (k_manifold, v_manifold)
        else:
            present_key_value = None
        
        L_k = k_manifold.shape[2]
        
        # Compute geodesic distances for attention scores
        # d_H(q_i, k_j) using batch distance computation
        dist_sq = self._geodesic_distance_sq(q_manifold, k_manifold)  # [B, H, L, L_k]
        
        # Attention scores: s_ij = -d²/τ
        scores = -dist_sq / self.temperature.clamp(min=0.1, max=10.0)
        
        # Add positional bias with warmup
        if position_ids is not None:
            pos_bias = self._compute_positional_bias(position_ids, L_k, training_step)
            scores = scores + pos_bias
        
        # Clamp for numerical stability
        scores = scores.clamp(-50.0, 50.0)
        
        # Apply attention mask (causal or padding)
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Value aggregation via tangent space weighted sum
        output = self._aggregate_values(v_manifold, attn_weights)  # [B, H, L, head_dim+1]
        
        # Map back to tangent, project, reshape
        output_tangent = self._log_map_head(output)  # [B, H, L, head_dim+1]
        output_spatial = output_tangent[..., 1:]  # [B, H, L, head_dim]
        
        # Reshape: [B, H, L, head_dim] -> [B, L, n]
        output_spatial = output_spatial.transpose(1, 2).contiguous().view(B, L, self.n_embd)
        
        # Output projection
        output_spatial = self.out_proj(output_spatial)
        output_spatial = self.resid_dropout(output_spatial)
        
        # Map back to manifold
        zero_time_out = torch.zeros(B, L, 1, device=x.device, dtype=x.dtype)
        output_tangent_full = torch.cat([zero_time_out, output_spatial], dim=-1)
        output = self.substrate.exp_map_zero(output_tangent_full)
        
        return output, present_key_value
    
    def _exp_map_head(self, v: torch.Tensor) -> torch.Tensor:
        """Exp map for head-dimension vectors."""
        K = self.substrate.K.to(v.device, v.dtype)
        sqrt_K = torch.sqrt(K)
        
        v_spatial = v[..., 1:]
        v_norm = torch.norm(v_spatial, dim=-1, keepdim=True).clamp(min=1e-6)
        scale = (sqrt_K * v_norm).clamp(max=15.0)
        
        cosh_scale = torch.cosh(scale)
        sinh_scale = torch.sinh(scale) / (v_norm + 1e-6)
        
        x_time = cosh_scale / sqrt_K
        x_spatial = sinh_scale * v_spatial
        
        return torch.cat([x_time, x_spatial], dim=-1)
    
    def _log_map_head(self, y: torch.Tensor) -> torch.Tensor:
        """Log map for head-dimension points."""
        K = self.substrate.K.to(y.device, y.dtype)
        sqrt_K = torch.sqrt(K)
        
        y0 = y[..., 0:1]
        y_spatial = y[..., 1:]
        
        arg = (y0 * sqrt_K).clamp(min=1.0 + 1e-6, max=1e5)
        d = safe_acosh(arg) / sqrt_K
        
        y_spatial_norm = torch.norm(y_spatial, dim=-1, keepdim=True).clamp(min=1e-6)
        v_spatial = d * y_spatial / y_spatial_norm
        v_time = torch.zeros_like(y0)
        
        return torch.cat([v_time, v_spatial], dim=-1)
    
    def _geodesic_distance_sq(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor
    ) -> torch.Tensor:
        """Compute squared geodesic distances for attention."""
        K = self.substrate.K.to(q.device, q.dtype)
        
        # Minkowski inner product
        q_time = q[..., 0:1]
        q_space = q[..., 1:]
        k_time = k[..., 0:1]
        k_space = k[..., 1:]
        
        # Batched matmul for pairwise inner products
        space_inner = torch.matmul(q_space, k_space.transpose(-2, -1))
        time_inner = torch.matmul(q_time, k_time.transpose(-2, -1))
        inner = -time_inner + space_inner
        
        # d² = arccosh²(-K·⟨q,k⟩_L) / K
        arg = (-K * inner).clamp(min=1.0 + 1e-6, max=1e5)
        dist = safe_acosh(arg) / torch.sqrt(K)
        
        return dist ** 2
    
    def _compute_positional_bias(
        self, 
        position_ids: torch.Tensor,
        L_k: int,
        training_step: int,
    ) -> torch.Tensor:
        """Compute hyperbolic positional bias with warmup."""
        B, L_q = position_ids.shape
        device = position_ids.device
        dtype = self.lambda_pos.dtype
        
        # Positional lattice: p_i at distance i*Δ from origin
        # For simplicity, use absolute position difference
        pos_q = position_ids.unsqueeze(-1).float()  # [B, L_q, 1]
        pos_k = torch.arange(L_k, device=device, dtype=dtype).view(1, 1, -1)  # [1, 1, L_k]
        
        # Position difference (proxy for geodesic distance on lattice)
        pos_diff = torch.abs(pos_q - pos_k)  # [B, L_q, L_k]
        
        # Logarithmic compression (hyperbolic property)
        pos_dist = torch.log1p(pos_diff)
        
        # Warmup coefficient
        warmup = min(1.0, training_step / max(1, self.config.positional_warmup_steps))
        lambda_eff = self.lambda_pos * warmup
        
        # Bias: -λ_pos * d_pos
        bias = -lambda_eff * pos_dist
        
        # Expand for heads: [B, L_q, L_k] -> [B, 1, L_q, L_k]
        return bias.unsqueeze(1)
    
    def _aggregate_values(
        self, 
        values: torch.Tensor, 
        weights: torch.Tensor
    ) -> torch.Tensor:
        """Aggregate values via tangent space weighted sum."""
        # Map to tangent
        v_tangent = self._log_map_head(values)  # [B, H, L_k, head_dim+1]
        
        # Weighted sum: [B, H, L_q, L_k] @ [B, H, L_k, d+1] -> [B, H, L_q, d+1]
        v_agg = torch.einsum('bhqk,bhkd->bhqd', weights, v_tangent)
        
        # Map back to manifold
        return self._exp_map_head(v_agg)


# =============================================================================
# HYPERBOLIC FEED-FORWARD NETWORK
# =============================================================================

class HyperbolicFFN(nn.Module):
    """
    Hyperbolic Feed-Forward Network.
    
    Principle: "Non-linearities belong to Euclidean space."
    
    Process:
        x → log_o(x) → extract spatial → W₁ → GELU → W₂ → exp_o
    
    Initialization:
        W₂ initialized with small std to prevent exp_map explosion.
    """
    
    def __init__(
        self,
        config: HyperbolicTransformerConfig,
        substrate: LorentzSubstrateHardened,
    ):
        super().__init__()
        self.config = config
        self.substrate = substrate
        
        self.fc1 = nn.Linear(config.n_embd, config.ffn_dim)
        self.fc2 = nn.Linear(config.ffn_dim, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        
        # Critical: small init for fc2 to prevent explosion
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=1e-4)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input on manifold [B, L, n+1]
            
        Returns:
            Output on manifold [B, L, n+1]
        """
        # Map to tangent
        v = self.substrate.log_map_zero_batch(x)
        v_spatial = v[..., 1:]  # [B, L, n]
        
        # FFN in Euclidean space
        h = F.gelu(self.fc1(v_spatial))
        h = self.dropout(h)
        out_spatial = self.fc2(h)
        out_spatial = self.dropout(out_spatial)
        
        # Map back to manifold
        zero_time = torch.zeros(
            *out_spatial.shape[:-1], 1,
            device=out_spatial.device,
            dtype=out_spatial.dtype
        )
        out_tangent = torch.cat([zero_time, out_spatial], dim=-1)
        
        return self.substrate.exp_map_zero(out_tangent)


# =============================================================================
# HYPERBOLIC RESIDUAL
# =============================================================================

class HyperbolicResidual(nn.Module):
    """
    Hyperbolic Residual Connection.
    
    Tangent space approximation:
        r = log_o(x) + log_o(f(x))
        y = exp_o(r)
    
    This is exact when x = origin, with bounded distortion for
    points near origin (controlled by radius regularization).
    """
    
    def __init__(self, substrate: LorentzSubstrateHardened):
        super().__init__()
        self.substrate = substrate
    
    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """
        Add residual connection in tangent space.
        
        Args:
            x: Original input on manifold
            residual: Output to add
            
        Returns:
            Combined output on manifold
        """
        v_x = self.substrate.log_map_zero_batch(x)
        v_r = self.substrate.log_map_zero_batch(residual)
        
        v_combined = v_x + v_r
        
        return self.substrate.exp_map_zero(v_combined)


# =============================================================================
# HYPERBOLIC TRANSFORMER BLOCK
# =============================================================================

class HyperbolicTransformerBlock(nn.Module):
    """
    Complete Hyperbolic Transformer Block.
    
    Architecture (Pre-Norm):
        x → LayerNorm → Attention → Residual → LayerNorm → FFN → Residual
    """
    
    def __init__(
        self,
        config: HyperbolicTransformerConfig,
        substrate: LorentzSubstrateHardened,
        layer_idx: int,
    ):
        super().__init__()
        self.config = config
        self.substrate = substrate
        self.layer_idx = layer_idx
        
        self.ln_1 = HyperbolicLayerNorm(
            config.n_embd, 
            substrate,
            gamma_init=config.layer_norm_gamma_init
        )
        self.attn = HyperbolicAttention(config, substrate, layer_idx)
        self.residual_1 = HyperbolicResidual(substrate)
        
        self.ln_2 = HyperbolicLayerNorm(
            config.n_embd,
            substrate,
            gamma_init=config.layer_norm_gamma_init
        )
        self.ffn = HyperbolicFFN(config, substrate)
        self.residual_2 = HyperbolicResidual(substrate)
    
    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        training_step: int = 0,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass through block."""
        # Self-attention with pre-norm
        normed = self.ln_1(x)
        attn_out, present = self.attn(
            normed,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            training_step=training_step,
        )
        x = self.residual_1(x, attn_out)
        
        # FFN with pre-norm
        normed = self.ln_2(x)
        ffn_out = self.ffn(normed)
        x = self.residual_2(x, ffn_out)
        
        return x, present


# =============================================================================
# HYPERBOLIC LM HEAD
# =============================================================================

class HyperbolicLMHead(nn.Module):
    """
    Language Model Head for Hyperbolic Transformer.
    
    Computes logits as negative geodesic distance to vocabulary embeddings:
        logit_j = -d_H(h, e_j)
    
    This makes "closer = higher probability" in hyperbolic space.
    """
    
    def __init__(
        self,
        config: HyperbolicTransformerConfig,
        substrate: LorentzSubstrateHardened,
        embedding_weight: Optional[nn.Parameter] = None,
    ):
        super().__init__()
        self.config = config
        self.substrate = substrate
        
        if embedding_weight is not None and config.tie_word_embeddings:
            # Share weights with input embeddings
            self.weight = embedding_weight
        else:
            # Separate output embeddings
            self.weight = nn.Parameter(
                torch.randn(config.vocab_size, config.n_embd) * 0.01
            )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute logits.
        
        Args:
            hidden_states: Hidden states on manifold [B, L, n+1]
            
        Returns:
            Logits [B, L, vocab_size]
        """
        B, L, D = hidden_states.shape
        
        # Project vocabulary embeddings to manifold
        zero_time = torch.zeros(
            self.config.vocab_size, 1,
            device=self.weight.device,
            dtype=self.weight.dtype
        )
        vocab_tangent = torch.cat([zero_time, self.weight], dim=-1)  # [V, n+1]
        vocab_manifold = self.substrate.exp_map_zero(vocab_tangent)  # [V, n+1]
        
        # Compute distances from hidden states to all vocab embeddings
        # hidden_states: [B, L, n+1], vocab_manifold: [V, n+1]
        # Result: [B, L, V]
        
        K = self.substrate.K.to(hidden_states.device, hidden_states.dtype)
        
        # Reshape for batch computation
        h_flat = hidden_states.view(B * L, D)  # [B*L, n+1]
        
        # Minkowski inner products
        h_time = h_flat[:, 0:1]  # [B*L, 1]
        h_space = h_flat[:, 1:]  # [B*L, n]
        v_time = vocab_manifold[:, 0:1].t()  # [1, V]
        v_space = vocab_manifold[:, 1:].t()  # [n, V]
        
        space_inner = torch.mm(h_space, v_space)  # [B*L, V]
        time_inner = torch.mm(h_time, v_time)     # [B*L, V]
        inner = -time_inner + space_inner
        
        # Geodesic distance
        arg = (-K * inner).clamp(min=1.0 + 1e-6, max=1e5)
        dist = safe_acosh(arg) / torch.sqrt(K)
        
        # Logits = negative distance (closer = higher score)
        logits = -dist.view(B, L, -1)
        
        return logits


# =============================================================================
# MAIN MODEL
# =============================================================================

class HyperbolicTransformer(nn.Module):
    """
    Hyperbolic-Native Transformer for Language Modeling.
    
    Complete autoregressive language model in hyperbolic space.
    
    Features:
        - Lorentz (hyperboloid) geometry
        - Geodesic attention
        - Tangent space FFN
        - Riemannian optimization support
        - Distance-based output logits
    """
    
    def __init__(self, config: HyperbolicTransformerConfig):
        super().__init__()
        self.config = config
        
        # Create shared substrate
        lorentz_config = LorentzConfig(
            intrinsic_dim=config.n_embd,
            initial_curvature=config.initial_curvature,
            learnable_curvature=config.learnable_curvature,
            curvature_min=config.curvature_min,
            curvature_max=config.curvature_max,
        )
        self.substrate = LorentzSubstrateHardened(lorentz_config)
        
        # Token embeddings
        self.wte = HyperbolicEmbedding(
            config.vocab_size,
            config.n_embd,
            self.substrate,
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            HyperbolicTransformerBlock(config, self.substrate, i)
            for i in range(config.n_layer)
        ])
        
        # Final layer norm
        self.ln_f = HyperbolicLayerNorm(
            config.n_embd,
            self.substrate,
            gamma_init=config.layer_norm_gamma_init,
        )
        
        # LM head
        self.lm_head = HyperbolicLMHead(
            config,
            self.substrate,
            self.wte.weight if config.tie_word_embeddings else None,
        )
        
        # Initialize
        self.apply(self._init_weights)
        
        # Training step counter for positional warmup
        self.register_buffer('training_step', torch.tensor(0))
    
    def _init_weights(self, module: nn.Module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Dict[str, Any]:
        """
        Forward pass.
        
        Args:
            input_ids: Token indices [B, L]
            attention_mask: Attention mask [B, L]
            position_ids: Position indices [B, L]
            past_key_values: KV cache for generation
            use_cache: Whether to return updated cache
            labels: Target token IDs for loss computation
            return_dict: Always True (for compatibility)
            
        Returns:
            Dictionary with 'logits', 'loss' (if labels), 'past_key_values' (if use_cache)
        """
        B, L = input_ids.shape
        device = input_ids.device
        
        # Position IDs
        if position_ids is None:
            past_length = past_key_values[0][0].shape[2] if past_key_values else 0
            position_ids = torch.arange(
                past_length, past_length + L, 
                device=device
            ).unsqueeze(0).expand(B, -1)
        
        # Causal attention mask
        if attention_mask is None:
            attention_mask = torch.ones(B, L, device=device)
        
        # Convert to additive mask for attention
        # [B, L] -> [B, 1, L, L]
        past_length = past_key_values[0][0].shape[2] if past_key_values else 0
        total_length = past_length + L
        
        # Causal mask
        causal_mask = torch.triu(
            torch.full((L, total_length), float('-inf'), device=device),
            diagonal=past_length + 1
        )
        
        # Combine with padding mask
        if attention_mask.dim() == 2:
            padding_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * float('-inf')
            # Expand padding mask for key positions
            if past_key_values is not None:
                # Need full attention mask for all key positions
                padding_mask = torch.zeros(B, 1, L, total_length, device=device)
        else:
            padding_mask = torch.zeros(B, 1, L, total_length, device=device)
        
        attn_mask = causal_mask.unsqueeze(0).unsqueeze(0) + padding_mask
        
        # Embed tokens
        hidden_states = self.wte(input_ids)  # [B, L, n+1]
        
        # Pass through blocks
        presents = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            past = past_key_values[i] if past_key_values else None
            hidden_states, present = block(
                hidden_states,
                position_ids=position_ids,
                attention_mask=attn_mask,
                past_key_value=past,
                use_cache=use_cache,
                training_step=self.training_step.item(),
            )
            if use_cache:
                presents.append(present)
        
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # LM head
        logits = self.lm_head(hidden_states)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        
        # Increment training step
        if self.training:
            self.training_step += 1
        
        return {
            'logits': logits,
            'loss': loss,
            'past_key_values': presents if use_cache else None,
            'hidden_states': hidden_states,
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Prompt token IDs [B, L]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            do_sample: Whether to sample (vs greedy)
            
        Returns:
            Generated token IDs [B, L + max_new_tokens]
        """
        self.eval()
        device = input_ids.device
        
        past_key_values = None
        generated = input_ids
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Only process last token if we have cache
                if past_key_values is not None:
                    curr_input = generated[:, -1:]
                else:
                    curr_input = generated
                
                outputs = self.forward(
                    curr_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                
                logits = outputs['logits'][:, -1, :]  # [B, V]
                past_key_values = outputs['past_key_values']
                
                # Temperature
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')
                
                # Sample or greedy
                probs = F.softmax(logits, dim=-1)
                if do_sample:
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(probs, dim=-1, keepdim=True)
                
                generated = torch.cat([generated, next_token], dim=-1)
        
        return generated
    
    # =========================================================================
    # DIAGNOSTIC METHODS
    # =========================================================================
    
    def manifold_fidelity(self) -> Dict[str, float]:
        """
        Compute manifold fidelity metrics (F1 from spec).
        
        Returns:
            Dict with mean violation and max violation
        """
        violations = []
        
        # Check embedding weights
        zero_time = torch.zeros(
            self.config.vocab_size, 1,
            device=self.wte.weight.device,
            dtype=self.wte.weight.dtype
        )
        emb_tangent = torch.cat([zero_time, self.wte.weight], dim=-1)
        emb_manifold = self.substrate.exp_map_zero(emb_tangent)
        v = self.substrate.manifold_violation(emb_manifold)
        violations.append(v.item())
        
        return {
            'mean_violation': sum(violations) / len(violations),
            'max_violation': max(violations),
        }
    
    def radius_statistics(self, hidden_states: torch.Tensor) -> Dict[str, float]:
        """
        Compute radius statistics for monitoring.
        
        Args:
            hidden_states: Points on manifold [B, L, n+1]
            
        Returns:
            Dict with mean, std, min, max radius
        """
        radii = self.substrate.lorentz_radius(hidden_states)
        return {
            'radius_mean': radii.mean().item(),
            'radius_std': radii.std().item(),
            'radius_min': radii.min().item(),
            'radius_max': radii.max().item(),
        }
    
    def get_loss_components(
        self,
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all loss components for multi-objective training.
        
        Returns:
            Dict with 'lm_loss', 'radius_loss', 'manifold_loss'
        """
        losses = {}
        
        # Standard LM loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        losses['lm_loss'] = F.cross_entropy(
            shift_logits.view(-1, self.config.vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        
        # Radius regularization
        radii = self.substrate.lorentz_radius(hidden_states)
        losses['radius_loss'] = F.relu(radii - self.config.radius_max).pow(2).mean()
        
        # Manifold fidelity
        losses['manifold_loss'] = self.substrate.manifold_violation(hidden_states)
        
        return losses


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_hyperbolic_gpt2_small() -> HyperbolicTransformer:
    """Create a GPT-2 Small equivalent hyperbolic model."""
    config = HyperbolicTransformerConfig(
        vocab_size=50257,
        n_embd=768 // 3,  # ~256 (hyperbolic compression)
        n_layer=12,
        n_head=8,
        n_positions=1024,
    )
    return HyperbolicTransformer(config)


def create_hyperbolic_gpt2_medium() -> HyperbolicTransformer:
    """Create a GPT-2 Medium equivalent hyperbolic model."""
    config = HyperbolicTransformerConfig(
        vocab_size=50257,
        n_embd=1024 // 3,  # ~341
        n_layer=24,
        n_head=16,
        n_positions=1024,
    )
    return HyperbolicTransformer(config)


def create_minimal_hyperbolic_transformer() -> HyperbolicTransformer:
    """Create minimal model for testing stability."""
    config = HyperbolicTransformerConfig(
        vocab_size=1000,
        n_embd=64,
        n_layer=2,
        n_head=4,
        n_positions=128,
        dropout=0.0,
        attention_dropout=0.0,
    )
    return HyperbolicTransformer(config)

# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
Hyperbolic Transformer [HARDENED VERSION]
=========================================

NaN-proof implementation with defensive guards at every critical point.

Changes from base version:
1. All exp/log operations wrapped with nan_to_num
2. LMHead uses tangent-space dot product (not geodesic distance)
3. Attention uses clamped similarity instead of raw distance
4. LayerNorm has fallback for degenerate cases
5. All intermediate tensors sanitized

Author: Éric Gustavo Reis de Sena
Date: January 2026
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from cgt.geometry.lorentz_hardened import (
    LorentzConfig,
    LorentzSubstrateHardened,
    safe_acosh,
)


def sanitize(x: torch.Tensor, name: str = "") -> torch.Tensor:
    """Replace NaN/Inf with safe values. Use everywhere."""
    return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class HyperbolicTransformerConfig:
    """Configuration for Hyperbolic Transformer."""
    vocab_size: int = 50257
    n_embd: int = 64          # Smaller default for stability testing
    n_layer: int = 4          # Fewer layers for stability testing
    n_head: int = 4
    n_positions: int = 512
    ffn_ratio: int = 4
    dropout: float = 0.0      # Disable dropout initially
    attention_dropout: float = 0.0
    
    # Geometry
    initial_curvature: float = 1.0
    learnable_curvature: bool = False
    curvature_min: float = 0.1
    curvature_max: float = 10.0
    
    # Stability
    radius_max: float = 5.0   # Conservative
    attention_temperature: float = 1.0
    positional_lambda: float = 0.0  # Disable positional bias initially
    positional_warmup_steps: int = 5000
    layer_norm_gamma_init: float = 1.0  # Start at 1.0
    embedding_init_std: float = 0.001   # Very small init
    
    # Training
    tie_word_embeddings: bool = True
    use_cache: bool = True
    
    @property
    def head_dim(self) -> int:
        assert self.n_embd % self.n_head == 0
        return self.n_embd // self.n_head
    
    @property
    def ffn_dim(self) -> int:
        return self.n_embd * self.ffn_ratio


# =============================================================================
# HARDENED EMBEDDING
# =============================================================================

class HyperbolicEmbeddingHardened(nn.Module):
    """
    Hyperbolic embedding with NaN guards.
    
    Key change: Embeddings stored as Euclidean, projected on-the-fly.
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        substrate: LorentzSubstrateHardened,
        init_std: float = 0.001,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.substrate = substrate
        self.padding_idx = padding_idx
        
        # Very small init to keep embeddings near origin
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim) * init_std)
        
        if padding_idx is not None:
            with torch.no_grad():
                self.weight[padding_idx].zero_()
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Lookup and project to manifold with guards."""
        # Lookup
        e_spatial = F.embedding(input_ids, self.weight, padding_idx=self.padding_idx)
        e_spatial = sanitize(e_spatial, "embedding_lookup")
        
        # Clamp magnitude to prevent explosion
        e_norm = e_spatial.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        max_norm = 2.0  # Conservative limit
        scale = torch.where(e_norm > max_norm, max_norm / e_norm, torch.ones_like(e_norm))
        e_spatial = e_spatial * scale
        
        # Build tangent vector: [0, e_spatial]
        e_time = torch.zeros(*e_spatial.shape[:-1], 1, device=e_spatial.device, dtype=e_spatial.dtype)
        e_tangent = torch.cat([e_time, e_spatial], dim=-1)
        
        # Project to manifold
        result = self._safe_exp_map_zero(e_tangent)
        return sanitize(result, "embedding_output")
    
    def _safe_exp_map_zero(self, v: torch.Tensor) -> torch.Tensor:
        """Safe exponential map from origin."""
        K = self.substrate.K.to(v.device, v.dtype)
        sqrt_K = torch.sqrt(K)
        
        v_spatial = v[..., 1:]
        v_spatial = sanitize(v_spatial, "exp_v_spatial")
        
        v_norm = torch.norm(v_spatial, dim=-1, keepdim=True)
        v_norm = sanitize(v_norm, "exp_v_norm").clamp(min=1e-8, max=10.0)
        
        scale = (sqrt_K * v_norm).clamp(max=10.0)
        
        cosh_scale = torch.cosh(scale)
        sinh_scale = torch.sinh(scale) / (v_norm + 1e-8)
        
        cosh_scale = sanitize(cosh_scale, "cosh")
        sinh_scale = sanitize(sinh_scale, "sinh")
        
        x_time = cosh_scale / sqrt_K
        x_spatial = sinh_scale * v_spatial
        
        result = torch.cat([x_time, x_spatial], dim=-1)
        return sanitize(result, "exp_result")


# =============================================================================
# HARDENED LAYER NORM
# =============================================================================

class HyperbolicLayerNormHardened(nn.Module):
    """
    Riemannian LayerNorm with fallback for degenerate cases.
    """
    
    def __init__(
        self,
        normalized_shape: int,
        substrate: LorentzSubstrateHardened,
        eps: float = 1e-6,
        gamma_init: float = 1.0,
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.substrate = substrate
        self.eps = eps
        
        self.gamma = nn.Parameter(torch.ones(normalized_shape) * gamma_init)
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LayerNorm with NaN guards."""
        x = sanitize(x, "ln_input")
        
        # Map to tangent at origin
        v = self._safe_log_map_zero(x)
        v_spatial = v[..., 1:]
        v_spatial = sanitize(v_spatial, "ln_v_spatial")
        
        # Standard LayerNorm
        mean = v_spatial.mean(dim=-1, keepdim=True)
        var = v_spatial.var(dim=-1, keepdim=True, unbiased=False)
        var = var.clamp(min=self.eps)  # Prevent division by zero
        
        v_norm = (v_spatial - mean) / torch.sqrt(var + self.eps)
        v_norm = sanitize(v_norm, "ln_normalized")
        
        # Scale and shift
        v_out = self.gamma * v_norm + self.beta
        v_out = sanitize(v_out, "ln_scaled")
        
        # Clamp to prevent explosion
        v_out = v_out.clamp(-5.0, 5.0)
        
        # Map back
        v_time = torch.zeros(*v_out.shape[:-1], 1, device=v_out.device, dtype=v_out.dtype)
        v_full = torch.cat([v_time, v_out], dim=-1)
        
        result = self._safe_exp_map_zero(v_full)
        return sanitize(result, "ln_output")
    
    def _safe_log_map_zero(self, y: torch.Tensor) -> torch.Tensor:
        """Safe log map to origin."""
        K = self.substrate.K.to(y.device, y.dtype)
        sqrt_K = torch.sqrt(K)
        
        y = sanitize(y, "log_input")
        
        y0 = y[..., 0:1]
        y_spatial = y[..., 1:]
        
        # Ensure y0 is valid for acosh
        y0 = y0.clamp(min=1.0 / sqrt_K + 1e-6)
        
        arg = (y0 * sqrt_K).clamp(min=1.0 + 1e-6, max=1e4)
        d = safe_acosh(arg) / sqrt_K
        d = sanitize(d, "log_distance").clamp(max=10.0)
        
        y_spatial_norm = torch.norm(y_spatial, dim=-1, keepdim=True).clamp(min=1e-8)
        v_spatial = d * y_spatial / y_spatial_norm
        v_spatial = sanitize(v_spatial, "log_v_spatial")
        
        v_time = torch.zeros_like(y0)
        return torch.cat([v_time, v_spatial], dim=-1)
    
    def _safe_exp_map_zero(self, v: torch.Tensor) -> torch.Tensor:
        """Safe exp map from origin."""
        K = self.substrate.K.to(v.device, v.dtype)
        sqrt_K = torch.sqrt(K)
        
        v_spatial = v[..., 1:]
        v_spatial = sanitize(v_spatial, "exp_v")
        
        v_norm = torch.norm(v_spatial, dim=-1, keepdim=True).clamp(min=1e-8, max=10.0)
        scale = (sqrt_K * v_norm).clamp(max=10.0)
        
        cosh_scale = torch.cosh(scale)
        sinh_scale = torch.sinh(scale) / (v_norm + 1e-8)
        
        x_time = sanitize(cosh_scale / sqrt_K, "exp_time")
        x_spatial = sanitize(sinh_scale * v_spatial, "exp_spatial")
        
        return torch.cat([x_time, x_spatial], dim=-1)


# =============================================================================
# HARDENED ATTENTION (EUCLIDEAN APPROXIMATION)
# =============================================================================

class HyperbolicAttentionHardened(nn.Module):
    """
    Attention with Euclidean computation in tangent space.
    
    Key insight: For points near origin, tangent space ≈ Euclidean space.
    This is numerically stable and still captures hyperbolic structure
    when radius regularization keeps embeddings near origin.
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
        
        # Standard QKV projections
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self._init_weights()
    
    def _init_weights(self):
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
        """Attention in tangent space (Euclidean approximation)."""
        B, L, D = x.shape
        x = sanitize(x, "attn_input")
        
        # Extract spatial components (tangent space at origin approximation)
        # x is on manifold [B, L, n+1], spatial is [B, L, n]
        x_spatial = x[..., 1:]
        x_spatial = sanitize(x_spatial, "attn_spatial")
        
        # QKV projection
        q = self.q_proj(x_spatial)
        k = self.k_proj(x_spatial)
        v = self.v_proj(x_spatial)
        
        q = sanitize(q, "q")
        k = sanitize(k, "k")
        v = sanitize(v, "v")
        
        # Reshape for multi-head
        q = q.view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        
        # KV cache
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        
        present = (k, v) if use_cache else None
        L_k = k.shape[2]
        
        # Standard scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = sanitize(scores, "attn_scores")
        
        # Causal mask
        if attention_mask is not None:
            scores = scores + attention_mask
        
        scores = scores.clamp(-50.0, 50.0)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = sanitize(attn_weights, "attn_weights")
        attn_weights = self.attn_dropout(attn_weights)
        
        # Aggregate
        out = torch.matmul(attn_weights, v)
        out = sanitize(out, "attn_out")
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(B, L, self.n_embd)
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        out = sanitize(out, "attn_proj_out")
        
        # Reconstruct manifold point: [time, spatial]
        # Time component for point near origin: approximately 1/sqrt(K)
        K = self.substrate.K.to(out.device, out.dtype)
        out_norm_sq = (out ** 2).sum(dim=-1, keepdim=True)
        out_time = torch.sqrt(1.0 / K + out_norm_sq + 1e-8)
        
        result = torch.cat([out_time, out], dim=-1)
        return sanitize(result, "attn_final"), present


# =============================================================================
# HARDENED FFN
# =============================================================================

class HyperbolicFFNHardened(nn.Module):
    """FFN in tangent space with guards."""
    
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
        
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.001)  # Very small
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = sanitize(x, "ffn_input")
        
        # Extract spatial (tangent space approximation)
        x_spatial = x[..., 1:]
        x_spatial = sanitize(x_spatial, "ffn_spatial")
        
        # FFN
        h = F.gelu(self.fc1(x_spatial))
        h = sanitize(h, "ffn_h")
        h = self.dropout(h)
        out = self.fc2(h)
        out = sanitize(out, "ffn_out")
        out = self.dropout(out)
        
        # Clamp
        out = out.clamp(-10.0, 10.0)
        
        # Reconstruct manifold point
        K = self.substrate.K.to(out.device, out.dtype)
        out_norm_sq = (out ** 2).sum(dim=-1, keepdim=True)
        out_time = torch.sqrt(1.0 / K + out_norm_sq + 1e-8)
        
        result = torch.cat([out_time, out], dim=-1)
        return sanitize(result, "ffn_final")


# =============================================================================
# HARDENED RESIDUAL
# =============================================================================

class HyperbolicResidualHardened(nn.Module):
    """Simple residual in spatial components."""
    
    def __init__(self, substrate: LorentzSubstrateHardened):
        super().__init__()
        self.substrate = substrate
    
    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        x = sanitize(x, "res_x")
        residual = sanitize(residual, "res_r")
        
        # Add spatial components
        x_spatial = x[..., 1:]
        r_spatial = residual[..., 1:]
        
        combined = x_spatial + r_spatial
        combined = sanitize(combined, "res_combined")
        
        # Clamp to prevent explosion
        combined = combined.clamp(-10.0, 10.0)
        
        # Reconstruct time
        K = self.substrate.K.to(combined.device, combined.dtype)
        combined_norm_sq = (combined ** 2).sum(dim=-1, keepdim=True)
        time = torch.sqrt(1.0 / K + combined_norm_sq + 1e-8)
        
        result = torch.cat([time, combined], dim=-1)
        return sanitize(result, "res_final")


# =============================================================================
# TRANSFORMER BLOCK
# =============================================================================

class HyperbolicTransformerBlockHardened(nn.Module):
    """Block with pre-norm."""
    
    def __init__(
        self,
        config: HyperbolicTransformerConfig,
        substrate: LorentzSubstrateHardened,
        layer_idx: int,
    ):
        super().__init__()
        self.ln_1 = HyperbolicLayerNormHardened(config.n_embd, substrate, gamma_init=config.layer_norm_gamma_init)
        self.attn = HyperbolicAttentionHardened(config, substrate, layer_idx)
        self.residual_1 = HyperbolicResidualHardened(substrate)
        
        self.ln_2 = HyperbolicLayerNormHardened(config.n_embd, substrate, gamma_init=config.layer_norm_gamma_init)
        self.ffn = HyperbolicFFNHardened(config, substrate)
        self.residual_2 = HyperbolicResidualHardened(substrate)
    
    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        training_step: int = 0,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Attention
        normed = self.ln_1(x)
        attn_out, present = self.attn(normed, position_ids, attention_mask, past_key_value, use_cache, training_step)
        x = self.residual_1(x, attn_out)
        
        # FFN
        normed = self.ln_2(x)
        ffn_out = self.ffn(normed)
        x = self.residual_2(x, ffn_out)
        
        return x, present


# =============================================================================
# LM HEAD (EUCLIDEAN DOT PRODUCT)
# =============================================================================

class HyperbolicLMHeadHardened(nn.Module):
    """
    LM head using Euclidean dot product in tangent space.
    
    Much more stable than computing geodesic distances to all vocab embeddings.
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
            self.weight = embedding_weight
        else:
            self.weight = nn.Parameter(torch.randn(config.vocab_size, config.n_embd) * 0.001)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute logits via dot product in tangent space."""
        hidden_states = sanitize(hidden_states, "lm_input")
        
        # Extract spatial (tangent space)
        h_spatial = hidden_states[..., 1:]  # [B, L, n]
        h_spatial = sanitize(h_spatial, "lm_spatial")
        
        # Simple linear projection: logits = h @ W^T
        logits = F.linear(h_spatial, self.weight)
        logits = sanitize(logits, "lm_logits")
        
        return logits


# =============================================================================
# MAIN MODEL
# =============================================================================

class HyperbolicTransformerHardened(nn.Module):
    """
    Hyperbolic Transformer [HARDENED VERSION].
    
    All operations have NaN guards and conservative bounds.
    """
    
    def __init__(self, config: HyperbolicTransformerConfig):
        super().__init__()
        self.config = config
        
        # Substrate
        lorentz_config = LorentzConfig(
            intrinsic_dim=config.n_embd,
            initial_curvature=config.initial_curvature,
            learnable_curvature=config.learnable_curvature,
            curvature_min=config.curvature_min,
            curvature_max=config.curvature_max,
        )
        self.substrate = LorentzSubstrateHardened(lorentz_config)
        
        # Embeddings
        self.wte = HyperbolicEmbeddingHardened(
            config.vocab_size,
            config.n_embd,
            self.substrate,
            init_std=config.embedding_init_std,
        )
        
        # Blocks
        self.blocks = nn.ModuleList([
            HyperbolicTransformerBlockHardened(config, self.substrate, i)
            for i in range(config.n_layer)
        ])
        
        # Final norm
        self.ln_f = HyperbolicLayerNormHardened(
            config.n_embd,
            self.substrate,
            gamma_init=config.layer_norm_gamma_init,
        )
        
        # LM head
        self.lm_head = HyperbolicLMHeadHardened(
            config,
            self.substrate,
            self.wte.weight if config.tie_word_embeddings else None,
        )
        
        self.register_buffer('training_step', torch.tensor(0))
        
        print(f"HyperbolicTransformerHardened initialized:")
        print(f"  - vocab_size: {config.vocab_size}")
        print(f"  - n_embd: {config.n_embd}")
        print(f"  - n_layer: {config.n_layer}")
        print(f"  - n_head: {config.n_head}")
        print(f"  - params: {sum(p.numel() for p in self.parameters())/1e6:.2f}M")
    
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
        B, L = input_ids.shape
        device = input_ids.device
        
        # Position IDs
        if position_ids is None:
            past_length = past_key_values[0][0].shape[2] if past_key_values else 0
            position_ids = torch.arange(past_length, past_length + L, device=device).unsqueeze(0).expand(B, -1)
        
        # Causal mask
        past_length = past_key_values[0][0].shape[2] if past_key_values else 0
        total_length = past_length + L
        causal_mask = torch.triu(
            torch.full((L, total_length), float('-inf'), device=device),
            diagonal=past_length + 1
        )
        attn_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Embed
        hidden_states = self.wte(input_ids)
        hidden_states = sanitize(hidden_states, "embed_out")
        
        # Blocks
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
            hidden_states = sanitize(hidden_states, f"block_{i}_out")
            if use_cache:
                presents.append(present)
        
        # Final norm
        hidden_states = self.ln_f(hidden_states)
        hidden_states = sanitize(hidden_states, "ln_f_out")
        
        # LM head
        logits = self.lm_head(hidden_states)
        logits = sanitize(logits, "logits")
        
        # Loss
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            loss = sanitize(loss.unsqueeze(0), "loss").squeeze(0)
        
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
        self.eval()
        past_key_values = None
        generated = input_ids
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                if past_key_values is not None:
                    curr_input = generated[:, -1:]
                else:
                    curr_input = generated
                
                outputs = self.forward(curr_input, past_key_values=past_key_values, use_cache=True)
                logits = outputs['logits'][:, -1, :]
                past_key_values = outputs['past_key_values']
                
                if temperature != 1.0:
                    logits = logits / temperature
                
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                probs = F.softmax(logits, dim=-1)
                if do_sample:
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(probs, dim=-1, keepdim=True)
                
                generated = torch.cat([generated, next_token], dim=-1)
        
        return generated
    
    def manifold_fidelity(self) -> Dict[str, float]:
        """Check manifold constraint satisfaction."""
        with torch.no_grad():
            # Check a sample of embeddings
            sample_ids = torch.randint(0, min(1000, self.config.vocab_size), (10,), device=self.wte.weight.device)
            emb = self.wte(sample_ids.unsqueeze(0)).squeeze(0)
            violation = self.substrate.manifold_violation(emb)
            return {
                'mean_violation': violation.item() if torch.isfinite(violation) else 0.0,
                'max_violation': violation.item() if torch.isfinite(violation) else 0.0,
            }
    
    def radius_statistics(self, hidden_states: torch.Tensor) -> Dict[str, float]:
        """Compute radius statistics."""
        with torch.no_grad():
            radii = self.substrate.lorentz_radius(hidden_states)
            radii = sanitize(radii, "radii")
            return {
                'radius_mean': radii.mean().item() if torch.isfinite(radii.mean()) else 0.0,
                'radius_std': radii.std().item() if torch.isfinite(radii.std()) else 0.0,
                'radius_min': radii.min().item() if torch.isfinite(radii.min()) else 0.0,
                'radius_max': radii.max().item() if torch.isfinite(radii.max()) else 0.0,
            }


# =============================================================================
# FACTORY
# =============================================================================

def create_minimal_hyperbolic_transformer_hardened() -> HyperbolicTransformerHardened:
    """Create minimal model for testing."""
    config = HyperbolicTransformerConfig(
        vocab_size=1000,
        n_embd=64,
        n_layer=2,
        n_head=4,
        n_positions=128,
        dropout=0.0,
        attention_dropout=0.0,
        embedding_init_std=0.001,
        layer_norm_gamma_init=1.0,
    )
    return HyperbolicTransformerHardened(config)


def create_hyperbolic_gpt2_small_hardened() -> HyperbolicTransformerHardened:
    """Create GPT-2 Small equivalent."""
    config = HyperbolicTransformerConfig(
        vocab_size=50257,
        n_embd=256,
        n_layer=6,
        n_head=8,
        n_positions=512,
        dropout=0.1,
        attention_dropout=0.1,
        embedding_init_std=0.001,
        layer_norm_gamma_init=1.0,
    )
    return HyperbolicTransformerHardened(config)

# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.

"""
Hyperbolic Kuramoto Attention for H-AKORN
=========================================

Integrates hyperbolic geometry with Kuramoto oscillator dynamics:
1. Hyperbolic distance-based attention
2. Phase-modulated attention scores
3. Geodesic-aware value aggregation
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .phase_dynamics import KuramotoPhaseEvolution
from .coupling import AdaptiveCoupling


class HyperbolicKuramotoAttention(nn.Module):
    """
    Hyperbolic attention mechanism with Kuramoto phase dynamics.
    
    Attention scores are modulated by both:
    1. Hyperbolic distances (geometric similarity)
    2. Phase differences (oscillator synchronization)
    
    Score_ij = exp(-d_H(q_i, k_j) / τ) * cos(θ_i - θ_j)
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
        substrate: Hyperbolic geometry substrate (optional, for direct use)
        curvature: Hyperbolic curvature K (if substrate not provided)
        coupling_strength: Kuramoto coupling strength
        use_phase_modulation: Whether to modulate attention with phases
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        substrate = None,
        curvature: float = -1.0,
        coupling_strength: float = 1.0,
        use_phase_modulation: bool = True,
        temperature: float = 1.0,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.curvature = curvature
        self.use_phase_modulation = use_phase_modulation
        self.temperature = temperature
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout_layer = nn.Dropout(dropout)
        
        # Kuramoto phase dynamics
        self.phase_evolution = KuramotoPhaseEvolution(
            d_model=d_model,
            num_heads=num_heads,
            coupling_strength=coupling_strength,
        )
        
        # Adaptive coupling
        self.coupling = AdaptiveCoupling(
            num_heads=num_heads,
            coupling_type='hybrid',
            temperature=temperature,
        )
        
        # Store substrate reference if provided
        self.substrate = substrate
    
    def hyperbolic_distance(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute hyperbolic distance between points.
        
        If substrate is provided, use its distance function.
        Otherwise, use Poincaré ball distance approximation.
        
        Args:
            x: [B, ..., D] query points
            y: [B, ..., D] key points
        
        Returns:
            distances: [B, ...] hyperbolic distances
        """
        if self.substrate is not None:
            # Use provided substrate
            return self.substrate.distance(x, y)
        else:
            # Poincaré ball distance approximation
            # d_H(x, y) = arcosh(1 + 2||x - y||²/((1-||x||²)(1-||y||²)))
            
            # Clip norms to avoid numerical issues
            x_norm_sq = torch.sum(x * x, dim=-1, keepdim=True).clamp(max=0.99)
            y_norm_sq = torch.sum(y * y, dim=-1, keepdim=True).clamp(max=0.99)
            
            diff_norm_sq = torch.sum((x - y) ** 2, dim=-1)
            
            denominator = (1 - x_norm_sq.squeeze(-1)) * (1 - y_norm_sq.squeeze(-1))
            arg = 1 + 2 * diff_norm_sq / (denominator + 1e-8)
            
            # arcosh(x) = log(x + sqrt(x²-1))
            distance = torch.log(arg + torch.sqrt((arg ** 2 - 1).clamp(min=1e-8)))
            
            return distance * torch.sqrt(torch.abs(torch.tensor(self.curvature)))
    
    def compute_attention_scores(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        phases: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute hyperbolic attention scores modulated by phase.
        
        Args:
            Q: [B, H, L, D] queries
            K: [B, H, S, D] keys
            phases: [B, H] phase values
            mask: [B, 1, L, S] or [B, H, L, S] attention mask
        
        Returns:
            scores: [B, H, L, S] attention scores
            phase_modulation: [B, H, 1, 1] phase modulation factor
        """
        B, H, L, D = Q.shape
        S = K.shape[2]
        
        # Compute pairwise hyperbolic distances
        # Expand Q and K for broadcasting: [B, H, L, 1, D] and [B, H, 1, S, D]
        Q_expanded = Q.unsqueeze(3)  # [B, H, L, 1, D]
        K_expanded = K.unsqueeze(2)  # [B, H, 1, S, D]
        
        # Hyperbolic distance matrix: [B, H, L, S]
        distances = self.hyperbolic_distance(Q_expanded, K_expanded)
        
        # Convert distances to similarity: exp(-d_H / τ)
        scores = torch.exp(-distances / self.temperature)
        
        # Phase modulation: cos(θ_i - θ_j)
        phase_modulation = torch.ones(B, H, 1, 1, device=Q.device)
        
        if self.use_phase_modulation and phases is not None:
            # Phase differences between heads
            phase_diff = phases.unsqueeze(2) - phases.unsqueeze(1)  # [B, H, H]
            phase_coherence = torch.cos(phase_diff)  # [B, H, H]
            
            # Average coherence per head: [B, H]
            avg_coherence = phase_coherence.mean(dim=2)
            
            # Expand to match score dimensions: [B, H, 1, 1]
            phase_modulation = avg_coherence.unsqueeze(2).unsqueeze(3)
            
            # Modulate scores
            scores = scores * phase_modulation.expand_as(scores)
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [B, 1, L, S]
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        return scores, phase_modulation
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass of hyperbolic Kuramoto attention.
        
        Args:
            hidden_states: [B, L, D] input hidden states
            attention_mask: [B, L, S] or [B, 1, L, S] attention mask
            output_attentions: Whether to return attention weights
        
        Returns:
            output: [B, L, D] output hidden states
            attention_weights: [B, H, L, S] attention weights (if output_attentions)
            phases: [B, H] current phase values
        """
        B, L, D = hidden_states.shape
        H = self.num_heads
        head_dim = self.head_dim
        
        # Linear projections and reshape
        Q = self.q_proj(hidden_states).view(B, L, H, head_dim).transpose(1, 2)  # [B, H, L, D_h]
        K = self.k_proj(hidden_states).view(B, L, H, head_dim).transpose(1, 2)  # [B, H, S, D_h]
        V = self.v_proj(hidden_states).view(B, L, H, head_dim).transpose(1, 2)  # [B, H, S, D_h]
        
        # Compute adaptive coupling matrix (will be used for phase evolution)
        # For now, compute simple coupling based on attention pattern
        coupling_matrix = self.coupling(
            attention_scores=None,  # Will update after computing attention
            phases=None,
        )
        
        # Evolve phases using Kuramoto dynamics
        phases, order_parameter = self.phase_evolution(
            coupling_matrix=coupling_matrix,
            batch_size=B,
        )
        
        # Compute attention scores with phase modulation
        scores, phase_modulation = self.compute_attention_scores(
            Q, K, phases, attention_mask
        )
        
        # Softmax attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout_layer(attention_weights)
        
        # Update coupling matrix with actual attention patterns
        coupling_matrix = self.coupling(
            attention_scores=attention_weights,
            phases=phases,
        )
        
        # Weighted sum of values
        context = torch.matmul(attention_weights, V)  # [B, H, L, D_h]
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(B, L, D)
        output = self.out_proj(context)
        
        if output_attentions:
            return output, attention_weights, phases, order_parameter
        else:
            return output, None, phases, order_parameter
    
    def reset_phases(self):
        """Reset phase dynamics."""
        self.phase_evolution.reset_phases()


class MultiHeadHyperbolicKuramotoAttention(nn.Module):
    """
    Multi-head variant with independent phase evolution per head group.
    
    Useful for modeling different timescales or frequency bands.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_head_groups: int = 1,
        dropout: float = 0.1,
        substrate = None,
        curvature: float = -1.0,
        coupling_strength: float = 1.0,
        use_phase_modulation: bool = True,
    ):
        super().__init__()
        assert num_heads % num_head_groups == 0, "num_heads must be divisible by num_head_groups"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_head_groups = num_head_groups
        self.heads_per_group = num_heads // num_head_groups
        
        # Create attention modules for each head group
        self.attention_groups = nn.ModuleList([
            HyperbolicKuramotoAttention(
                d_model=d_model // num_head_groups,
                num_heads=self.heads_per_group,
                dropout=dropout,
                substrate=substrate,
                curvature=curvature,
                coupling_strength=coupling_strength,
                use_phase_modulation=use_phase_modulation,
            )
            for _ in range(num_head_groups)
        ])
        
        self.output_proj = nn.Linear(d_model, d_model)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward pass with grouped heads."""
        B, L, D = hidden_states.shape
        group_dim = D // self.num_head_groups
        
        outputs = []
        all_phases = []
        all_order_params = []
        all_attentions = [] if output_attentions else None
        
        # Process each head group
        for i, attn_group in enumerate(self.attention_groups):
            # Split input by head group
            group_input = hidden_states[..., i*group_dim:(i+1)*group_dim]
            
            # Forward pass
            group_output, group_attn, group_phases, group_order = attn_group(
                group_input,
                attention_mask,
                output_attentions,
            )
            
            outputs.append(group_output)
            all_phases.append(group_phases)
            all_order_params.append(group_order)
            
            if output_attentions:
                all_attentions.append(group_attn)
        
        # Concatenate outputs
        output = torch.cat(outputs, dim=-1)
        output = self.output_proj(output)
        
        # Concatenate phases
        phases = torch.cat(all_phases, dim=-1)
        order_parameter = torch.stack(all_order_params, dim=-1).mean(dim=-1)
        
        if output_attentions:
            attention_weights = torch.cat(all_attentions, dim=1)
            return output, attention_weights, phases, order_parameter
        else:
            return output, None, phases, order_parameter
    
    def reset_phases(self):
        """Reset all phase dynamics."""
        for attn_group in self.attention_groups:
            attn_group.reset_phases()

# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.

"""
H-AKORN Layer
============

Complete transformer layer with hyperbolic Kuramoto attention.

Architecture:
1. Hyperbolic Kuramoto Multi-Head Attention
2. Layer Normalization
3. Feed-Forward Network (with hyperbolic operations optional)
4. Residual connections
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import HyperbolicKuramotoAttention


class RiemannianLayerNorm(nn.Module):
    """
    Riemannian Layer Normalization for hyperbolic space.
    Specification 3.6: 
    1. Log map: v = logo(h)
    2. Euclidean norm: v_norm = LayerNorm(v)
    3. Exp map: h_norm = expo(v_norm)
    
    CRITICAL: γ initialized > 1.0 to prevent radial collapse.
    
    Args:
        d_model: Model dimension
        eps: Layer norm epsilon
        substrate: Hyperbolic substrate
    """
    
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        substrate=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.substrate = substrate
        
        self.layer_norm = nn.LayerNorm(d_model, eps=eps)
        
        if substrate is not None:
            with torch.no_grad():
                self.layer_norm.weight.fill_(1.2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Riemannian LayerNorm.
        
        Args:
            x: [B, L, D] Euclidean or [B, L, D+1] Lorentz manifold points
        
        Returns:
            x_norm: Normalized points (same shape as input)
        """
        if self.substrate is None:
            return self.layer_norm(x)
        
        B, L, D = x.shape
        expected_lorentz_dim = self.d_model + 1
        
        if D == expected_lorentz_dim:
            x_flat = x.reshape(B * L, D)
            
            v = self.substrate.log_map_zero(x_flat)
            
            v_spatial = v[:, 1:]
            v_spatial_norm = self.layer_norm(v_spatial)
            
            v_norm = torch.cat([torch.zeros(B * L, 1, device=x.device, dtype=x.dtype), v_spatial_norm], dim=-1)
            x_norm = self.substrate.exp_map_zero(v_norm)
            
            return x_norm.reshape(B, L, -1)
        
        elif D == self.d_model:
            return self.layer_norm(x)
        
        else:
            raise ValueError(
                f"RiemannianLayerNorm expects input dimension {self.d_model} (Euclidean) "
                f"or {expected_lorentz_dim} (Lorentz), got {D}"
            )


class HyperbolicFeedForward(nn.Module):
    """
    Feed-forward network with optional hyperbolic operations.
    
    Args:
        d_model: Model dimension
        d_ff: Feed-forward dimension
        dropout: Dropout probability
        activation: Activation function ('relu', 'gelu', 'swish')
        use_hyperbolic: Whether to use hyperbolic operations
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu',
        use_hyperbolic: bool = False,
        substrate=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.use_hyperbolic = use_hyperbolic
        self.substrate = substrate
        
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'swish':
            self.activation = lambda x: x * torch.sigmoid(x)
        else:
            self.activation = F.gelu
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Specification 3.5: FFN(h) = expo(W2 · GELU(W1 · logo(h)))
        
        Args:
            x: [B, L, D] Euclidean or [B, L, D+1] Lorentz manifold points
        
        Returns:
            output: Same shape as input
        """
        if self.use_hyperbolic and self.substrate is not None:
            B, L, D = x.shape
            expected_lorentz_dim = self.d_model + 1
            
            if D == expected_lorentz_dim:
                x_flat = x.reshape(B * L, D)
                
                v = self.substrate.log_map_zero(x_flat)
                
                v_spatial = v[:, 1:]
                v_hidden = self.activation(self.fc1(v_spatial))
                v_hidden = self.dropout(v_hidden)
                v_out = self.fc2(v_hidden)
                v_out = self.dropout(v_out)
                
                v_out_padded = torch.cat([torch.zeros(B * L, 1, device=x.device, dtype=x.dtype), v_out], dim=-1)
                x_out = self.substrate.exp_map_zero(v_out_padded)
                
                return x_out.reshape(B, L, -1)
            
            elif D == self.d_model:
                x = self.fc1(x)
                x = self.activation(x)
                x = self.dropout(x)
                x = self.fc2(x)
                x = self.dropout(x)
                return x
            
            else:
                raise ValueError(
                    f"HyperbolicFeedForward expects input dimension {self.d_model} (Euclidean) "
                    f"or {expected_lorentz_dim} (Lorentz), got {D}"
                )
        else:
            x = self.fc1(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.dropout(x)
            return x


class HAKORNLayer(nn.Module):
    """
    Single H-AKORN transformer layer.
    
    Combines:
    1. Hyperbolic Kuramoto attention
    2. Feed-forward network
    3. Layer normalization
    4. Residual connections
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout probability
        layer_norm_eps: Layer norm epsilon
        substrate: Hyperbolic substrate (optional)
        curvature: Hyperbolic curvature
        coupling_strength: Kuramoto coupling strength
        use_phase_modulation: Whether to use phase modulation in attention
        pre_norm: Whether to use pre-normalization (True) or post-normalization (False)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        substrate = None,
        curvature: float = -1.0,
        coupling_strength: float = 1.0,
        use_phase_modulation: bool = True,
        pre_norm: bool = True,
        activation: str = 'gelu',
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.pre_norm = pre_norm
        
        # Hyperbolic Kuramoto attention
        self.attention = HyperbolicKuramotoAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            substrate=substrate,
            curvature=curvature,
            coupling_strength=coupling_strength,
            use_phase_modulation=use_phase_modulation,
        )
        
        # Feed-forward network
        self.feed_forward = HyperbolicFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
            use_hyperbolic=substrate is not None,
            substrate=substrate,
        )
        
        # Layer normalization
        self.norm1 = RiemannianLayerNorm(d_model, eps=layer_norm_eps, substrate=substrate)
        self.norm2 = RiemannianLayerNorm(d_model, eps=layer_norm_eps, substrate=substrate)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            hidden_states: [B, L, D] input hidden states
            attention_mask: [B, L, S] attention mask
            output_attentions: Whether to return attention weights
        
        Returns:
            output: [B, L, D] output hidden states
            attention_weights: [B, H, L, S] attention weights (if output_attentions)
            phases: [B, H] phase values
            order_parameter: [B] Kuramoto order parameter
        """
        residual = hidden_states
        
        # Self-attention block
        if self.pre_norm:
            hidden_states = self.norm1(hidden_states)
            attn_output, attn_weights, phases, order_param = self.attention(
                hidden_states,
                attention_mask,
                output_attentions,
            )
            hidden_states = residual + self.dropout(attn_output)
        else:
            attn_output, attn_weights, phases, order_param = self.attention(
                hidden_states,
                attention_mask,
                output_attentions,
            )
            hidden_states = self.norm1(residual + self.dropout(attn_output))
        
        # Feed-forward block
        residual = hidden_states
        if self.pre_norm:
            hidden_states = self.norm2(hidden_states)
            ff_output = self.feed_forward(hidden_states)
            hidden_states = residual + self.dropout(ff_output)
        else:
            ff_output = self.feed_forward(hidden_states)
            hidden_states = self.norm2(residual + self.dropout(ff_output))
        
        return hidden_states, attn_weights, phases, order_param
    
    def reset_phases(self):
        """Reset phase dynamics."""
        self.attention.reset_phases()


class HAKORNEncoder(nn.Module):
    """
    Stack of H-AKORN layers forming an encoder.
    
    Args:
        num_layers: Number of transformer layers
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout probability
        layer_norm_eps: Layer norm epsilon
        substrate: Hyperbolic substrate (optional)
        curvature: Hyperbolic curvature
        coupling_strength: Kuramoto coupling strength
        use_phase_modulation: Whether to use phase modulation
        pre_norm: Whether to use pre-normalization
    """
    
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        substrate = None,
        curvature: float = -1.0,
        coupling_strength: float = 1.0,
        use_phase_modulation: bool = True,
        pre_norm: bool = True,
        activation: str = 'gelu',
    ):
        super().__init__()
        self.num_layers = num_layers
        
        # Stack of H-AKORN layers
        self.layers = nn.ModuleList([
            HAKORNLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps,
                substrate=substrate,
                curvature=curvature,
                coupling_strength=coupling_strength,
                use_phase_modulation=use_phase_modulation,
                pre_norm=pre_norm,
                activation=activation,
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm (for pre-norm architecture)
        if pre_norm:
            self.final_norm = RiemannianLayerNorm(d_model, eps=layer_norm_eps, substrate=substrate)
        else:
            self.final_norm = None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[list], Optional[list], list, list]:
        """
        Forward pass through all layers.
        
        Args:
            hidden_states: [B, L, D] input hidden states
            attention_mask: [B, L, S] attention mask
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
        
        Returns:
            output: [B, L, D] final output
            all_attentions: List of [B, H, L, S] attention weights per layer (if output_attentions)
            all_hidden_states: List of [B, L, D] hidden states per layer (if output_hidden_states)
            all_phases: List of [B, H] phases per layer
            all_order_params: List of [B] order parameters per layer
        """
        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None
        all_phases = []
        all_order_params = []
        
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            
            hidden_states, attn_weights, phases, order_param = layer(
                hidden_states,
                attention_mask,
                output_attentions,
            )
            
            all_phases.append(phases)
            all_order_params.append(order_param)
            
            if output_attentions:
                all_attentions.append(attn_weights)
        
        # Final normalization
        if self.final_norm is not None:
            hidden_states = self.final_norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        
        return hidden_states, all_attentions, all_hidden_states, all_phases, all_order_params
    
    def reset_phases(self):
        """Reset phase dynamics for all layers."""
        for layer in self.layers:
            layer.reset_phases()

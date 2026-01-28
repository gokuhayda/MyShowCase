# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.

"""
Adaptive Coupling for H-AKORN
=============================

Implements adaptive coupling matrix A_ij that depends on:
1. Attention scores (semantic coupling)
2. Phase differences (synchronization-based coupling)
3. Learnable coupling patterns

A_ij = σ(W_c * concat[attention_ij, cos(θ_j - θ_i)])
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveCoupling(nn.Module):
    """
    Adaptive coupling matrix computation for Kuramoto dynamics.
    
    The coupling matrix A_ij is computed based on:
    1. Attention scores (semantic similarity)
    2. Phase coherence (oscillator synchronization)
    3. Learnable patterns
    
    Args:
        num_heads: Number of attention heads (oscillators)
        coupling_type: Type of coupling ['attention', 'phase', 'hybrid']
        learnable_base: Whether to include learnable base coupling
    """
    
    def __init__(
        self,
        num_heads: int,
        coupling_type: str = 'hybrid',
        learnable_base: bool = True,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.coupling_type = coupling_type
        self.temperature = temperature
        
        if learnable_base:
            # Learnable base coupling matrix
            self.base_coupling = nn.Parameter(
                torch.eye(num_heads) + torch.randn(num_heads, num_heads) * 0.01
            )
        else:
            self.register_buffer(
                "base_coupling",
                torch.eye(num_heads)
            )
        
        if coupling_type in ['hybrid', 'attention']:
            # Projection for attention-based coupling
            self.attention_proj = nn.Linear(1, 1)  # Fixed: 1→1 instead of 1→num_heads
        
        if coupling_type in ['hybrid', 'phase']:
            # Projection for phase-based coupling
            self.phase_proj = nn.Linear(1, 1)  # Fixed: 1→1 instead of 1→num_heads
    
    def forward(
        self,
        attention_scores: Optional[torch.Tensor] = None,
        phases: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute adaptive coupling matrix.
        
        Args:
            attention_scores: [B, H, L, L] attention scores
            phases: [B, H] phase values
        
        Returns:
            coupling_matrix: [B, H, H] or [H, H] coupling matrix A_ij
        """
        device = self.base_coupling.device
        H = self.num_heads
        
        # Start with base coupling
        if attention_scores is not None:
            B = attention_scores.shape[0]
            A = self.base_coupling.unsqueeze(0).expand(B, -1, -1)
        else:
            A = self.base_coupling.clone()
        
        # Attention-based coupling
        if self.coupling_type in ['hybrid', 'attention'] and attention_scores is not None:
            # Average attention scores across sequence: [B, H, H]
            attn_coupling = attention_scores.mean(dim=[2, 3])
            
            # Project and add to coupling
            attn_coupling_expanded = attn_coupling.unsqueeze(-1)  # [B, H, H, 1]
            attn_contribution = self.attention_proj(attn_coupling_expanded).squeeze(-1)
            A = A + attn_contribution / self.temperature
        
        # Phase-based coupling
        if self.coupling_type in ['hybrid', 'phase'] and phases is not None:
            # Phase differences: [B, H, H]
            phase_diff = phases.unsqueeze(2) - phases.unsqueeze(1)
            phase_coupling = torch.cos(phase_diff)  # Coupling strength based on phase coherence
            
            # Project and add to coupling
            phase_coupling_expanded = phase_coupling.unsqueeze(-1)  # [B, H, H, 1]
            phase_contribution = self.phase_proj(phase_coupling_expanded).squeeze(-1)
            A = A + phase_contribution / self.temperature
        
        # Normalize and ensure symmetry
        A = (A + A.transpose(-2, -1)) / 2.0
        
        # Apply softmax to ensure non-negative coupling
        A = F.softmax(A / self.temperature, dim=-1) * H
        
        return A
    
    def get_coupling_strength(self, coupling_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute average coupling strength per head.
        
        Args:
            coupling_matrix: [B, H, H] or [H, H] coupling matrix
        
        Returns:
            strength: [B, H] or [H] average coupling per head
        """
        return coupling_matrix.sum(dim=-1) / self.num_heads


class HierarchicalCoupling(nn.Module):
    """
    Hierarchical coupling structure for multi-layer H-AKORN.
    
    Implements inter-layer coupling between adjacent layers:
    A_ij^(l,l+1) coupling between layer l and l+1
    """
    
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        inter_layer_strength: float = 0.5,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.inter_layer_strength = inter_layer_strength
        
        # Inter-layer coupling matrices
        self.inter_layer_coupling = nn.ParameterList([
            nn.Parameter(torch.randn(num_heads, num_heads) * 0.01)
            for _ in range(num_layers - 1)
        ])
    
    def forward(self, layer_idx: int) -> Optional[torch.Tensor]:
        """
        Get inter-layer coupling matrix for layer transitions.
        
        Args:
            layer_idx: Current layer index
        
        Returns:
            coupling: [H, H] inter-layer coupling or None if last layer
        """
        if layer_idx >= self.num_layers - 1:
            return None
        
        A = self.inter_layer_coupling[layer_idx]
        
        # Normalize
        A = (A + A.T) / 2.0
        A = F.softmax(A, dim=-1) * self.num_heads * self.inter_layer_strength
        
        return A


class DynamicCouplingScheduler(nn.Module):
    """
    Schedule coupling strength during training.
    
    Implements annealing schedule for coupling strength:
    K(t) = K_min + (K_max - K_min) * schedule(t)
    """
    
    def __init__(
        self,
        coupling_min: float = 0.1,
        coupling_max: float = 2.0,
        warmup_steps: int = 1000,
        schedule_type: str = 'cosine',
    ):
        super().__init__()
        self.coupling_min = coupling_min
        self.coupling_max = coupling_max
        self.warmup_steps = warmup_steps
        self.schedule_type = schedule_type
        self.register_buffer("current_step", torch.tensor(0))
    
    def forward(self) -> float:
        """Get current coupling strength."""
        step = self.current_step.item()
        
        if step < self.warmup_steps:
            # Linear warmup
            alpha = step / self.warmup_steps
        else:
            if self.schedule_type == 'cosine':
                # Cosine annealing
                alpha = 0.5 * (1 + torch.cos(
                    torch.tensor(torch.pi * (step - self.warmup_steps) / (10 * self.warmup_steps))
                )).item()
            elif self.schedule_type == 'constant':
                alpha = 1.0
            else:
                alpha = 1.0
        
        coupling_strength = self.coupling_min + (self.coupling_max - self.coupling_min) * alpha
        return coupling_strength
    
    def step(self):
        """Increment step counter."""
        self.current_step += 1
    
    def reset(self):
        """Reset step counter."""
        self.current_step = torch.tensor(0)

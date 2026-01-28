# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.

"""
Kuramoto Phase Dynamics for H-AKORN
===================================

Implements the Kuramoto oscillator model for phase synchronization:

dθ_i/dt = ω_i + (K/N) Σ_j A_ij sin(θ_j - θ_i)

Where:
- θ_i: phase of oscillator i
- ω_i: natural frequency
- K: coupling strength
- A_ij: coupling adjacency matrix
- N: number of oscillators
"""

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class KuramotoPhaseEvolution(nn.Module):
    """
    Kuramoto oscillator dynamics for phase synchronization.
    
    Implements discrete-time evolution:
    θ(t+Δt) = θ(t) + Δt * [ω + (K/N) Σ A_ij sin(θ_j - θ_i)]
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads (oscillators)
        coupling_strength: K parameter (default: 1.0)
        dt: Integration timestep (default: 0.1)
        learnable_frequencies: Whether ω_i are learnable (default: True)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        coupling_strength: float = 1.0,
        dt: float = 0.1,
        learnable_frequencies: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.coupling_strength = coupling_strength
        self.dt = dt
        
        # Natural frequencies ω_i for each head
        if learnable_frequencies:
            self.natural_frequencies = nn.Parameter(
                torch.randn(num_heads) * 0.1
            )
        else:
            self.register_buffer(
                "natural_frequencies",
                torch.randn(num_heads) * 0.1
            )
        
        # Phase state (initialized randomly)
        self.register_buffer(
            "phase_state",
            torch.rand(1, num_heads) * 2 * torch.pi
        )
    
    def forward(
        self,
        coupling_matrix: torch.Tensor,
        batch_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evolve phases according to Kuramoto dynamics.
        
        Args:
            coupling_matrix: [B, H, H] or [H, H] adjacency matrix A_ij
            batch_size: Batch size for phase expansion
        
        Returns:
            phases: [B, H] current phases
            order_parameter: [B] Kuramoto order parameter r
        """
        device = coupling_matrix.device
        
        # Expand phase state to batch size
        if batch_size is not None:
            phases = self.phase_state.expand(batch_size, -1).to(device)
        else:
            if coupling_matrix.dim() == 3:
                batch_size = coupling_matrix.shape[0]
                phases = self.phase_state.expand(batch_size, -1).to(device)
            else:
                phases = self.phase_state.to(device)
        
        # Ensure coupling matrix is 3D [B, H, H]
        if coupling_matrix.dim() == 2:
            coupling_matrix = coupling_matrix.unsqueeze(0)
        
        # Phase differences: θ_j - θ_i
        # [B, H, 1] - [B, 1, H] = [B, H, H]
        phase_diff = phases.unsqueeze(2) - phases.unsqueeze(1)
        
        # Coupling term: Σ A_ij sin(θ_j - θ_i)
        coupling_term = coupling_matrix * torch.sin(phase_diff)
        coupling_sum = coupling_term.sum(dim=2)  # [B, H]
        
        # Kuramoto equation: dθ/dt = ω + (K/N) Σ A_ij sin(θ_j - θ_i)
        dphase_dt = (
            self.natural_frequencies.to(device) +
            (self.coupling_strength / self.num_heads) * coupling_sum
        )
        
        # Discrete time evolution: θ(t+Δt) = θ(t) + Δt * dθ/dt
        new_phases = phases + self.dt * dphase_dt
        
        # Wrap phases to [0, 2π]
        new_phases = torch.fmod(new_phases, 2 * torch.pi)
        
        # Update internal state (detached to avoid backprop through time)
        with torch.no_grad():
            self.phase_state = new_phases[:1].detach().cpu()
        
        # Compute Kuramoto order parameter: r = |⟨e^(iθ)⟩|
        complex_phases = torch.complex(
            torch.cos(new_phases),
            torch.sin(new_phases)
        )
        order_parameter = torch.abs(complex_phases.mean(dim=1))
        
        return new_phases, order_parameter
    
    def reset_phases(self):
        """Reset phase state to random initialization."""
        self.phase_state = torch.rand(1, self.num_heads) * 2 * torch.pi
    
    def get_phase_coherence(self, phases: torch.Tensor) -> torch.Tensor:
        """
        Compute phase coherence (order parameter) r.
        
        r = |⟨e^(iθ)⟩| = |(1/N) Σ e^(iθ_i)|
        
        Args:
            phases: [B, H] phase values
        
        Returns:
            coherence: [B] order parameter values in [0, 1]
        """
        complex_phases = torch.complex(
            torch.cos(phases),
            torch.sin(phases)
        )
        return torch.abs(complex_phases.mean(dim=-1))


class PhaseCouplingRegularizer(nn.Module):
    """
    Regularization term for phase synchronization.
    
    L_phase = λ * (1 - r)²
    
    Encourages high order parameter r (synchronized phases).
    """
    
    def __init__(self, lambda_phase: float = 0.1):
        super().__init__()
        self.lambda_phase = lambda_phase
    
    def forward(self, order_parameter: torch.Tensor) -> torch.Tensor:
        """
        Compute phase coherence regularization loss.
        
        Args:
            order_parameter: [B] Kuramoto order parameter r
        
        Returns:
            loss: Scalar regularization loss
        """
        return self.lambda_phase * ((1.0 - order_parameter) ** 2).mean()

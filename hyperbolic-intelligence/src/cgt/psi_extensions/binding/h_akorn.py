"""
Hyperbolic Artificial Kuramoto Oscillatory Neurons (H-AKOrN)
=============================================================

This module implements the H-AKOrN mechanism for solving the
binding problem in the Ψ-SLM architecture.

The Binding Problem
-------------------
Standard vector embeddings suffer from the "superposition catastrophe":
when feature vectors are summed, the system cannot distinguish between
"red square + blue circle" and "blue square + red circle".

H-AKOrN solves this via phase synchronization. Each state maintains
a phase variable θ alongside its embedding. Semantically related
nodes synchronize (form "synchrony groups"), while distant nodes
remain asynchronous.

The Kuramoto Model
------------------
The phase dynamics follow:

    dθᵢ/dt = ωᵢ + Σⱼ K(i,j) * sin(θⱼ - θᵢ)

where the coupling K(i,j) decays exponentially with hyperbolic distance:

    K(i,j) = K₀ * exp(-d_H(i,j) / τ)

This ensures that only geometrically close nodes (semantically similar)
synchronize, while distant branches remain independent.

LOCAL Operation
---------------
This module implements LOCAL dynamics:
- Each node updates its phase based on local neighbors
- Coupling is local (decays with distance)
- No global topology computation

The Coherence metric Γ = |Σ exp(iθⱼ)| is computed for monitoring
but does NOT feed back into dynamics.

Author: Éric Reis
License: MIT
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import logging
import math

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class HAKORNConfig:
    """
    Configuration for H-AKOrN dynamics.
    
    Attributes
    ----------
    coupling_strength : float
        Base coupling constant K₀.
    decay_scale : float
        Distance decay scale τ.
    natural_frequency : float
        Base natural frequency ω₀.
    frequency_spread : float
        Spread of natural frequencies (diversity).
    dt : float
        Integration time step.
    integration_method : str
        ODE integration: 'euler', 'rk2', 'rk4'.
    """
    
    coupling_strength: float = 1.0
    decay_scale: float = 1.0
    natural_frequency: float = 0.0
    frequency_spread: float = 0.1
    dt: float = 0.1
    integration_method: str = "rk2"


# =============================================================================
# H-AKOrN Implementation
# =============================================================================

if HAS_TORCH:
    
    class HyperbolicAKORN(nn.Module):
        """
        Hyperbolic Artificial Kuramoto Oscillatory Neurons.
        
        This module maintains and updates phase variables for binding.
        
        The phase dynamics are:
            dθᵢ/dt = ωᵢ + Σⱼ K₀ * exp(-d_H(i,j)/τ) * sin(θⱼ - θᵢ)
        
        This creates synchrony groups based on hyperbolic proximity.
        
        Parameters
        ----------
        config : HAKORNConfig
            Configuration object.
        substrate : LorentzSubstrate
            Reference to geometric substrate for distances.
        
        Example
        -------
        >>> akorn = HyperbolicAKORN(config, substrate)
        >>> phases = akorn.initialize(100)
        >>> for _ in range(50):
        ...     phases = akorn.step(states, phases)
        >>> coherence = akorn.compute_coherence(phases)
        """
        
        def __init__(
            self,
            config: Optional[HAKORNConfig] = None,
            substrate: Optional["LorentzSubstrate"] = None,
        ):
            super().__init__()
            
            if config is None:
                config = HAKORNConfig()
            
            self.config = config
            self.substrate = substrate
            
            # Learnable parameters
            self.K0 = nn.Parameter(torch.tensor(config.coupling_strength))
            self.tau = nn.Parameter(torch.tensor(config.decay_scale))
            
        def initialize(
            self,
            num_nodes: int,
            device: Optional[torch.device] = None,
        ) -> torch.Tensor:
            """
            Initialize phase variables.
            
            Phases are initialized uniformly in [0, 2π) to maximize
            initial entropy and allow pattern formation.
            """
            if device is None:
                device = torch.device('cpu')
            
            return torch.rand(num_nodes, device=device, dtype=torch.float64) * 2 * math.pi
        
        def initialize_frequencies(
            self,
            num_nodes: int,
            device: Optional[torch.device] = None,
        ) -> torch.Tensor:
            """
            Initialize natural frequencies.
            
            Small diversity in frequencies prevents trivial global sync.
            """
            if device is None:
                device = torch.device('cpu')
            
            omega_0 = self.config.natural_frequency
            spread = self.config.frequency_spread
            
            return omega_0 + spread * torch.randn(num_nodes, device=device, dtype=torch.float64)
        
        def _compute_coupling(
            self,
            states: torch.Tensor,
        ) -> torch.Tensor:
            """
            Compute coupling matrix from hyperbolic distances.
            
            K(i,j) = K₀ * exp(-d_H(i,j) / τ)
            """
            dist_matrix = self.substrate.distance_matrix(states)
            coupling = self.K0 * torch.exp(-dist_matrix / (self.tau + 1e-8))
            
            # Zero diagonal (no self-coupling)
            n = coupling.shape[0]
            coupling = coupling * (1 - torch.eye(n, device=coupling.device, dtype=coupling.dtype))
            
            return coupling
        
        def _phase_dynamics(
            self,
            phases: torch.Tensor,
            coupling: torch.Tensor,
            omega: torch.Tensor,
        ) -> torch.Tensor:
            """
            Compute dθ/dt for all nodes.
            
            dθᵢ/dt = ωᵢ + Σⱼ K(i,j) * sin(θⱼ - θᵢ)
            """
            # Phase differences: θⱼ - θᵢ
            phase_diff = phases.unsqueeze(0) - phases.unsqueeze(1)  # (n, n)
            
            # Kuramoto interaction
            interaction = (coupling * torch.sin(phase_diff)).sum(dim=1)
            
            return omega + interaction
        
        def _euler_step(
            self,
            phases: torch.Tensor,
            coupling: torch.Tensor,
            omega: torch.Tensor,
            dt: float,
        ) -> torch.Tensor:
            """Euler integration step."""
            dtheta = self._phase_dynamics(phases, coupling, omega)
            return (phases + dt * dtheta) % (2 * math.pi)
        
        def _rk2_step(
            self,
            phases: torch.Tensor,
            coupling: torch.Tensor,
            omega: torch.Tensor,
            dt: float,
        ) -> torch.Tensor:
            """Runge-Kutta 2nd order (midpoint) step."""
            k1 = self._phase_dynamics(phases, coupling, omega)
            k2 = self._phase_dynamics(
                (phases + 0.5 * dt * k1) % (2 * math.pi),
                coupling, omega,
            )
            return (phases + dt * k2) % (2 * math.pi)
        
        def _rk4_step(
            self,
            phases: torch.Tensor,
            coupling: torch.Tensor,
            omega: torch.Tensor,
            dt: float,
        ) -> torch.Tensor:
            """Runge-Kutta 4th order step."""
            k1 = self._phase_dynamics(phases, coupling, omega)
            k2 = self._phase_dynamics(
                (phases + 0.5 * dt * k1) % (2 * math.pi),
                coupling, omega,
            )
            k3 = self._phase_dynamics(
                (phases + 0.5 * dt * k2) % (2 * math.pi),
                coupling, omega,
            )
            k4 = self._phase_dynamics(
                (phases + dt * k3) % (2 * math.pi),
                coupling, omega,
            )
            return (phases + dt * (k1 + 2*k2 + 2*k3 + k4) / 6) % (2 * math.pi)
        
        def step(
            self,
            states: torch.Tensor,
            phases: torch.Tensor,
            omega: Optional[torch.Tensor] = None,
            dt: Optional[float] = None,
        ) -> torch.Tensor:
            """
            Perform one integration step.
            
            Parameters
            ----------
            states : torch.Tensor
                Current embeddings on manifold, shape (N, ambient_dim)
            phases : torch.Tensor
                Current phases, shape (N,)
            omega : torch.Tensor, optional
                Natural frequencies. Initialized if None.
            dt : float, optional
                Time step. Uses config if None.
            
            Returns
            -------
            torch.Tensor
                Updated phases, shape (N,)
            """
            if dt is None:
                dt = self.config.dt
            
            if omega is None:
                omega = self.initialize_frequencies(
                    states.shape[0], device=states.device,
                )
            
            # Compute coupling from current state geometry
            coupling = self._compute_coupling(states)
            
            # Integrate
            method = self.config.integration_method
            if method == "euler":
                return self._euler_step(phases, coupling, omega, dt)
            elif method == "rk2":
                return self._rk2_step(phases, coupling, omega, dt)
            elif method == "rk4":
                return self._rk4_step(phases, coupling, omega, dt)
            else:
                return self._euler_step(phases, coupling, omega, dt)
        
        def compute_coherence(
            self,
            phases: torch.Tensor,
        ) -> torch.Tensor:
            """
            Compute global phase coherence (Kuramoto order parameter).
            
            Γ = |1/N Σⱼ exp(iθⱼ)|
            
            Γ = 0: Fully incoherent (uniform phase distribution)
            Γ = 1: Fully coherent (all phases aligned)
            
            Note: This is a DIAGNOSTIC metric. It does NOT feed back
            into the dynamics (that would violate local/global separation).
            """
            n = phases.shape[0]
            z = torch.exp(1j * phases.to(torch.complex64))
            return (z.sum() / n).abs()
        
        def compute_local_coherence(
            self,
            phases: torch.Tensor,
            states: torch.Tensor,
            radius: float = 1.0,
        ) -> torch.Tensor:
            """
            Compute local coherence within hyperbolic neighborhoods.
            
            Returns per-node coherence with their local neighborhood.
            """
            dist_matrix = self.substrate.distance_matrix(states)
            
            # Local neighborhoods
            mask = (dist_matrix < radius).to(dtype=dist_matrix.dtype)
            mask = mask * (1 - torch.eye(mask.shape[0], device=mask.device, dtype=mask.dtype))
            
            # Local coherence per node
            z = torch.exp(1j * phases.to(torch.complex64))
            local_sum = torch.einsum('ij,j->i', mask, z)
            local_count = mask.sum(dim=1) + 1e-8
            
            return (local_sum / local_count).abs()
        
        def identify_synchrony_groups(
            self,
            phases: torch.Tensor,
            threshold: float = 0.5,
        ) -> torch.Tensor:
            """
            Identify synchrony groups (nodes with similar phases).
            
            Returns cluster labels based on phase similarity.
            """
            # Phase difference matrix
            phase_diff = phases.unsqueeze(0) - phases.unsqueeze(1)
            similarity = torch.cos(phase_diff)
            
            # Simple clustering via thresholding
            adj = (similarity > threshold).to(dtype=similarity.dtype)
            
            # Find connected components (simplified)
            n = phases.shape[0]
            labels = torch.arange(n, device=phases.device)
            
            for _ in range(n):
                # Propagate smallest label in each component
                for i in range(n):
                    neighbors = adj[i].nonzero().squeeze(-1)
                    if neighbors.numel() > 0:
                        labels[i] = min(labels[i], labels[neighbors].min())
            
            # Renumber from 0
            unique_labels = labels.unique()
            mapping = {l.item(): i for i, l in enumerate(unique_labels)}
            labels = torch.tensor([mapping[l.item()] for l in labels], device=phases.device, dtype=phases.dtype)
            
            return labels
        
        def forward(
            self,
            states: torch.Tensor,
            phases: torch.Tensor,
            num_steps: int = 1,
            return_trajectory: bool = False,
        ) -> Tuple[torch.Tensor, Optional[list]]:
            """
            Forward pass: run dynamics for multiple steps.
            
            Returns final phases and optionally trajectory.
            """
            omega = self.initialize_frequencies(states.shape[0], device=states.device)
            
            trajectory = [phases] if return_trajectory else None
            current = phases
            
            for _ in range(num_steps):
                current = self.step(states, current, omega)
                if return_trajectory:
                    trajectory.append(current)
            
            return current, trajectory


# =============================================================================
# Phase Coherence Loss
# =============================================================================

if HAS_TORCH:
    
    class PhaseCoherenceLoss(nn.Module):
        """
        Loss to encourage phase coherence above critical threshold.
        
        In Kuramoto dynamics, the order parameter Γ = |1/N Σⱼ exp(iθⱼ)|
        measures global synchronization:
        - Γ ≈ 0: Incoherent (phases uniformly distributed)
        - Γ ≈ 1: Fully synchronized (all phases aligned)
        
        This loss penalizes when coherence falls below a critical threshold,
        encouraging the system to maintain functional binding.
        
        Parameters
        ----------
        critical_threshold : float
            Minimum desired coherence level (default: 0.3)
        
        Example
        -------
        >>> loss_fn = PhaseCoherenceLoss(critical_threshold=0.3)
        >>> order_param = akorn.compute_coherence(phases)
        >>> loss = loss_fn(order_param)
        """
        
        def __init__(self, critical_threshold: float = 0.3):
            super().__init__()
            self.critical = critical_threshold
        
        def forward(self, order_param: torch.Tensor) -> torch.Tensor:
            """
            Compute coherence loss.
            
            Returns 0 if coherence >= threshold, otherwise (threshold - coherence).
            """
            return torch.relu(self.critical - order_param)


# =============================================================================
# HAKORNLayer - Simplified wrapper for notebook compatibility
# =============================================================================

if HAS_TORCH:
    
    class HAKORNLayer(nn.Module):
        """
        Simplified H-AKORN layer for binding via phase synchronization.
        
        This is a lightweight wrapper that maintains phases as a buffer
        and provides a simple forward interface for use in larger models.
        
        Unlike HyperbolicAKORN which is more flexible, this layer:
        - Keeps phases as internal state (buffer)
        - Uses adjacency matrix for connectivity
        - Returns modulation weights and order parameter
        
        Parameters
        ----------
        num_nodes : int
            Number of nodes in the graph.
        coupling_strength : float
            Base coupling constant K₀.
        temperature : float
            Distance decay scale τ.
        dt : float
            Integration time step.
        
        Example
        -------
        >>> layer = HAKORNLayer(num_nodes=100)
        >>> modulation, order_param = layer(states, substrate, adjacency)
        """
        
        def __init__(
            self,
            num_nodes: int,
            coupling_strength: float = 1.0,
            temperature: float = 0.5,
            dt: float = 0.1,
        ):
            super().__init__()
            
            self.num_nodes = num_nodes
            self.dt = dt
            
            # Learnable parameters
            self.K0 = nn.Parameter(torch.tensor(coupling_strength))
            self.tau = nn.Parameter(torch.tensor(temperature))
            
            # Initialize phases uniformly in [0, 2π)
            self.register_buffer('phases', torch.rand(num_nodes) * 2 * math.pi)
        
        def reset_phases(self):
            """Reset phases to random uniform distribution."""
            self.phases = torch.rand(self.num_nodes, device=self.phases.device) * 2 * math.pi
        
        def forward(
            self,
            states: torch.Tensor,
            substrate: "LorentzSubstrateHardened",
            adjacency: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Compute modulation weights and order parameter.
            
            Parameters
            ----------
            states : torch.Tensor
                Node embeddings on manifold, shape (N, ambient_dim)
            substrate : LorentzSubstrateHardened
                Geometric substrate for distance computation.
            adjacency : torch.Tensor
                Adjacency matrix, shape (N, N)
            
            Returns
            -------
            modulation_weights : torch.Tensor
                Phase-based weights for NCA, shape (N, N)
            order_param : torch.Tensor
                Global coherence scalar.
            """
            # Compute hyperbolic distance matrix
            dist_matrix = substrate.distance_matrix(states)
            
            # Coupling matrix: K(i,j) = K₀ * exp(-d_H(i,j) / τ)
            coupling = self.K0 * torch.exp(-dist_matrix / (self.tau.abs() + 1e-8))
            coupling = coupling * adjacency  # Only connected nodes interact
            
            # Phase dynamics: dθᵢ/dt = Σⱼ K(i,j) * sin(θⱼ - θᵢ)
            phase_diff = self.phases.unsqueeze(0) - self.phases.unsqueeze(1)
            interaction = (coupling * torch.sin(phase_diff)).sum(dim=1)
            
            # RK2 integration (midpoint method)
            k1 = interaction
            phases_mid = (self.phases + 0.5 * self.dt * k1) % (2 * math.pi)
            phase_diff_mid = phases_mid.unsqueeze(0) - phases_mid.unsqueeze(1)
            k2 = (coupling * torch.sin(phase_diff_mid)).sum(dim=1)
            
            # Update phases (detached for gradient stability)
            self.phases = ((self.phases + self.dt * k2) % (2 * math.pi)).detach()
            
            # Modulation weights based on phase coherence
            phase_coherence = torch.cos(phase_diff)
            modulation_weights = (1 + phase_coherence) / 2  # Normalize to [0, 1]
            
            # Order parameter: Γ = |1/N Σⱼ exp(iθⱼ)|
            z = torch.exp(1j * self.phases.to(torch.complex64))
            order_param = (z.sum() / self.num_nodes).abs()
            
            return modulation_weights, order_param


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "HAKORNConfig",
]

if HAS_TORCH:
    __all__.extend([
        "HyperbolicAKORN",
        "HAKORNLayer",
        "PhaseCoherenceLoss",
    ])

"""
Hyperbolic Neural Cellular Automata (H-NCA)
===========================================

This module implements the H-NCA, the computational engine of the Ψ-SLM.

Theoretical Background
----------------------
Unlike standard cellular automata that operate on a fixed Euclidean grid,
the H-NCA operates on the hyperbolic plane. Each cell maintains a state
on the Lorentz manifold and updates via Riemannian flow.

The update follows a four-stage cycle:
1. Lorentzian Perception: Aggregate neighbor states (Einstein midpoint)
2. Tangent Projection: Project to local tangent space (log map)
3. Neural Operator: Apply MLP in flat tangent space
4. Manifold Update: Project back to manifold (exp map)

LOCAL vs GLOBAL
---------------
This module implements LOCAL dynamics only. It:
- Computes local neighborhoods
- Aggregates local information
- Applies local neural transformations

It does NOT:
- Compute global topology
- Evaluate overall manifold structure
- Access global constraint fields directly

The separation ensures clean architecture per UGFT principles.

Mathematical Foundation (Paper Reference)
-----------------------------------------
The update rule is a discretized Riemannian flow:
    h(t+1) = Exp_h(t)(-η * grad_h L_total)

From paper 09_toy_psi_slm.tex, Equation in Section 4.1.

API Compatibility
-----------------
This module uses LorentzSubstrateHardened from cgt_project:
- proj() for hyperboloid projection
- riemannian_grad() for tangent space projection
- exp_map() / log_map() for Riemannian operations

Author: Éric Reis
License: MIT
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# F1 CORRECTION: Import safe_acosh for gradient-preserving arccosh
try:
    from cgt.geometry.lorentz_hardened import safe_acosh
except ImportError:
    # Fallback if not available (should not happen in production)
    def safe_acosh(x, eps=1e-7):
        return torch.acosh(torch.clamp(x, min=1.0 + eps))


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class HNCAConfig:
    """
    Configuration for H-NCA.
    
    Attributes
    ----------
    intrinsic_dim : int
        Dimension of hyperbolic space.
    hidden_dim : int
        Hidden dimension of neural operator.
    num_neighbors : int
        Number of neighbors for aggregation.
    aggregation : str
        Aggregation method: 'mean', 'einstein', 'attention'.
    activation : str
        Activation function: 'gelu', 'relu', 'tanh'.
    dropout : float
        Dropout rate in neural operator.
    phase_coupling : bool
        Whether to use phase from H-AKOrN for weighting.
    """
    
    intrinsic_dim: int = 64
    hidden_dim: int = 128
    num_neighbors: int = 5
    aggregation: str = "einstein"
    activation: str = "gelu"
    dropout: float = 0.1
    phase_coupling: bool = True


# =============================================================================
# H-NCA Implementation
# =============================================================================

if HAS_TORCH:
    
    class HyperbolicNCA(nn.Module):
        """
        Hyperbolic Neural Cellular Automata.
        
        The H-NCA performs local state updates on the Lorentz manifold.
        
        Architecture
        ------------
        1. Neighbor selection (k-nearest in hyperbolic distance)
        2. Lorentzian aggregation (Einstein midpoint or attention)
        3. Tangent space MLP
        4. Exponential map back to manifold
        
        The update rule is a discretized Riemannian flow:
            x(t+1) = exp_x(t)(f(log_x(t)(neighbors)))
        
        Parameters
        ----------
        config : HNCAConfig
            Configuration object.
        substrate : LorentzSubstrateHardened
            Reference to the geometric substrate from cgt_project.
        
        Example
        -------
        >>> from cgt.geometry import LorentzSubstrateHardened
        >>> substrate = LorentzSubstrateHardened(intrinsic_dim=32)
        >>> hnca = HyperbolicNCA(HNCAConfig(intrinsic_dim=32), substrate)
        >>> states = substrate.random_init(100)
        >>> new_states = hnca(states)
        """
        
        def __init__(
            self,
            config: Optional[HNCAConfig] = None,
            substrate: Optional["LorentzSubstrateHardened"] = None,
        ):
            super().__init__()
            
            if config is None:
                config = HNCAConfig()
            
            self.config = config
            self.substrate = substrate
            
            # Dimensions
            self.intrinsic_dim = config.intrinsic_dim
            self.ambient_dim = config.intrinsic_dim + 1
            
            # Neural operator (operates in tangent space)
            activation_fn = {
                'gelu': nn.GELU(),
                'relu': nn.ReLU(),
                'tanh': nn.Tanh(),
            }.get(config.activation, nn.GELU())
            
            self.mlp = nn.Sequential(
                nn.Linear(config.intrinsic_dim, config.hidden_dim),
                activation_fn,
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                activation_fn,
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.intrinsic_dim),
                nn.Tanh(),  # Bound output for stability
            )
            
            # Scale factor for updates (learnable)
            self.update_scale = nn.Parameter(torch.tensor(0.1))
            
            # Attention weights for aggregation (optional)
            if config.aggregation == "attention":
                self.attention = nn.Linear(config.intrinsic_dim, 1)
            else:
                self.attention = None
        
        def _find_neighbors(
            self,
            states: torch.Tensor,
            k: Optional[int] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Find k-nearest neighbors in hyperbolic distance.
            
            Returns neighbor indices and distances.
            """
            if k is None:
                k = self.config.num_neighbors
            
            n = states.shape[0]
            k = min(k, n - 1)  # Can't have more neighbors than points
            
            # Compute distance matrix
            dist_matrix = self.substrate.distance_matrix(states)
            
            # Set diagonal to large value
            dist_matrix = dist_matrix + torch.eye(n, device=states.device, dtype=states.dtype) * 1e6
            
            # Get k-nearest
            distances, indices = torch.topk(dist_matrix, k, dim=1, largest=False)
            
            return indices, distances
        
        def _aggregate_mean(
            self,
            states: torch.Tensor,
            indices: torch.Tensor,
            phases: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            Simple mean aggregation in tangent space.
            """
            n, k = indices.shape
            
            # Gather neighbor states
            neighbor_states = states[indices]  # (n, k, ambient_dim)
            
            # Weight by phase similarity if available
            if phases is not None and self.config.phase_coupling:
                phase_self = phases.unsqueeze(1)  # (n, 1)
                phase_neighbors = phases[indices]  # (n, k)
                weights = torch.cos(phase_self - phase_neighbors)
                weights = F.softmax(weights, dim=1).unsqueeze(-1)
                weighted = (neighbor_states * weights).sum(dim=1)
            else:
                weighted = neighbor_states.mean(dim=1)
            
            return weighted
        
        def _aggregate_einstein(
            self,
            states: torch.Tensor,
            indices: torch.Tensor,
            phases: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            Einstein midpoint aggregation (Lorentzian centroid).
            
            The Einstein midpoint is the point minimizing sum of
            squared hyperbolic distances to all neighbors.
            
            Uses substrate.proj() for hyperboloid projection (CGT API).
            """
            n, k = indices.shape
            
            # Gather neighbor states
            neighbor_states = states[indices]  # (n, k, ambient_dim)
            
            # Phase weighting
            if phases is not None and self.config.phase_coupling:
                phase_self = phases.unsqueeze(1)
                phase_neighbors = phases[indices]
                weights = torch.cos(phase_self - phase_neighbors)
                weights = F.softmax(weights, dim=1).unsqueeze(-1)
            else:
                weights = torch.ones(n, k, 1, device=states.device, dtype=states.dtype) / k
            
            # Weighted sum in ambient space
            weighted_sum = (neighbor_states * weights).sum(dim=1)
            
            # Project back to hyperboloid (Einstein midpoint approximation)
            # Using proj() - CGT API (not project())
            centroid = self.substrate.proj(weighted_sum)
            
            return centroid
        
        def _aggregate_attention(
            self,
            states: torch.Tensor,
            indices: torch.Tensor,
            base_states: torch.Tensor,
        ) -> torch.Tensor:
            """
            Attention-based aggregation.
            """
            n, k = indices.shape
            
            neighbor_states = states[indices]
            
            # Project to tangent space at each base point
            tangent_vecs = []
            for i in range(n):
                logs = []
                for j in range(k):
                    log_v = self.substrate.log_map(
                        base_states[i:i+1],
                        neighbor_states[i, j:j+1],
                    )
                    logs.append(log_v)
                tangent_vecs.append(torch.cat(logs, dim=0))
            
            tangent_vecs = torch.stack(tangent_vecs)  # (n, k, ambient_dim)
            
            # Compute attention scores in tangent space
            # Use intrinsic part only
            tangent_intrinsic = tangent_vecs[..., 1:]  # (n, k, intrinsic_dim)
            scores = self.attention(tangent_intrinsic).squeeze(-1)  # (n, k)
            weights = F.softmax(scores, dim=1).unsqueeze(-1)
            
            # Weighted aggregation
            aggregated = (tangent_vecs * weights).sum(dim=1)
            
            # Return in tangent space
            return aggregated
        
        def forward(
            self,
            states: torch.Tensor,
            phases: Optional[torch.Tensor] = None,
            dt: float = 1.0,
        ) -> torch.Tensor:
            """
            Perform one H-NCA update step.
            
            Parameters
            ----------
            states : torch.Tensor
                Current states on manifold, shape (N, ambient_dim)
            phases : torch.Tensor, optional
                Phase variables from H-AKOrN, shape (N,)
            dt : float
                Time step size (for discrete integration interpretation)
            
            Returns
            -------
            torch.Tensor
                Updated states on manifold, shape (N, ambient_dim)
            """
            n = states.shape[0]
            
            # Step 1: Find neighbors
            indices, distances = self._find_neighbors(states)
            
            # Step 2: Aggregate
            if self.config.aggregation == "mean":
                aggregated = self._aggregate_mean(states, indices, phases)
            elif self.config.aggregation == "einstein":
                aggregated = self._aggregate_einstein(states, indices, phases)
            elif self.config.aggregation == "attention":
                aggregated = self._aggregate_attention(states, indices, states)
                # For attention, we're already in tangent space
                update_direction = aggregated[..., 1:]  # Intrinsic part
                update_direction = self.mlp(update_direction) * self.update_scale * dt
                # Reconstruct tangent vector
                tangent = torch.zeros(n, self.ambient_dim, device=states.device, dtype=states.dtype)
                tangent[..., 1:] = update_direction
                # Using riemannian_grad() - CGT API (not tangent_projection())
                tangent = self.substrate.riemannian_grad(states, tangent)
                return self.substrate.exp_map(states, tangent)
            else:
                aggregated = self._aggregate_mean(states, indices, phases)
            
            # Step 3: Project to tangent space
            log_vecs = self.substrate.log_map(states, aggregated)
            tangent_intrinsic = log_vecs[..., 1:]  # Intrinsic part
            
            # Step 4: Neural operator in tangent space
            update = self.mlp(tangent_intrinsic) * self.update_scale * dt
            
            # Step 5: Project back to manifold
            # Construct full tangent vector
            tangent_full = torch.zeros(n, self.ambient_dim, device=states.device, dtype=states.dtype)
            tangent_full[..., 1:] = update
            # Using riemannian_grad() - CGT API (not tangent_projection())
            tangent_full = self.substrate.riemannian_grad(states, tangent_full)
            
            new_states = self.substrate.exp_map(states, tangent_full)
            
            return new_states
        
        def multi_step(
            self,
            states: torch.Tensor,
            num_steps: int = 10,
            phases: Optional[torch.Tensor] = None,
            dt: float = 1.0,
        ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
            """
            Perform multiple H-NCA steps.
            
            Returns final states and trajectory.
            """
            trajectory = [states]
            current = states
            
            for _ in range(num_steps):
                current = self.forward(current, phases, dt)
                trajectory.append(current)
            
            return current, trajectory


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "HNCAConfig",
    "HyperbolicNCA" if HAS_TORCH else None,
    "HyperbolicNCAWrapper" if HAS_TORCH else None,
]


# =============================================================================
# HyperbolicNCAWrapper - Adjacency-based interface
# =============================================================================

if HAS_TORCH:
    
    class HyperbolicNCAWrapper(nn.Module):
        """
        Wrapper for H-NCA with adjacency matrix interface.
        
        Unlike HyperbolicNCA which uses k-NN for neighbor selection,
        this wrapper accepts an explicit adjacency matrix. This is useful
        for graph-structured data like trees.
        
        Includes manifold projection at each step for numerical stability.
        
        Parameters
        ----------
        substrate : LorentzSubstrateHardened
            Reference to the geometric substrate.
        embed_dim : int
            Intrinsic dimension of hyperbolic space.
        hidden_dim : int
            Hidden dimension of neural operator.
        dropout : float
            Dropout rate.
        max_radius : float
            Maximum hyperbolic radius (prevents explosion).
            
        Example
        -------
        >>> wrapper = HyperbolicNCAWrapper(substrate, embed_dim=16)
        >>> new_states = wrapper(states, adjacency, modulation_weights)
        """
        
        def __init__(
            self,
            substrate: "LorentzSubstrateHardened",
            embed_dim: int = 16,
            hidden_dim: int = 64,
            dropout: float = 0.1,
            max_radius: float = 5.0,
        ):
            super().__init__()
            self.substrate = substrate
            self.embed_dim = embed_dim
            self.ambient_dim = embed_dim + 1
            self.max_radius = max_radius

            # Neural operator in tangent space
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embed_dim),
                nn.Tanh(),
            )
            self.update_scale = nn.Parameter(torch.tensor(0.1))

        def clamp_to_ball(self, h: torch.Tensor) -> torch.Tensor:
            """Clamp states to max radius to prevent explosion."""
            K = self.substrate.K.to(h.device, h.dtype)
            x0 = h[..., 0]
            
            # radius = arccosh(x0 * sqrt(K)) / sqrt(K)
            # F1 CORRECTION: Use safe_acosh to preserve gradient near identity
            arg = (x0 * torch.sqrt(K)).clamp(min=1.0)
            radius = safe_acosh(arg) / torch.sqrt(K)
            
            # Scale down if radius exceeds max
            scale = torch.clamp(self.max_radius / (radius + 1e-8), max=1.0)
            
            # Scale spatial components
            h_scaled = h.clone()
            h_scaled[..., 1:] = h[..., 1:] * scale.unsqueeze(-1)
            
            return self.substrate.proj(h_scaled)

        def forward(
            self,
            states: torch.Tensor,
            adjacency: torch.Tensor,
            modulation_weights: Optional[torch.Tensor] = None,
            dt: float = 1.0,
        ) -> torch.Tensor:
            """
            Forward pass using adjacency matrix for neighbor selection.
            
            Parameters
            ----------
            states : torch.Tensor
                Current states on manifold, shape (N, ambient_dim)
            adjacency : torch.Tensor
                Adjacency matrix, shape (N, N)
            modulation_weights : torch.Tensor, optional
                Phase-based weights from H-AKORN, shape (N, N)
            dt : float
                Time step size
                
            Returns
            -------
            torch.Tensor
                Updated states on manifold
            """
            n = states.shape[0]
            device = states.device

            # Ensure states are on manifold
            states = self.substrate.proj(states)

            # Combine adjacency with modulation
            if modulation_weights is not None:
                weights = adjacency * modulation_weights
            else:
                weights = adjacency

            # Normalize weights
            weights_sum = weights.sum(dim=1, keepdim=True) + 1e-8
            weights_norm = weights / weights_sum

            # Aggregate neighbors (Einstein midpoint approximation)
            aggregated = torch.einsum('ij,jk->ik', weights_norm, states)
            aggregated = self.substrate.proj(aggregated)

            # Log map to tangent space
            log_vecs = self.substrate.log_map(states, aggregated)
            log_vecs = torch.nan_to_num(log_vecs, nan=0.0, posinf=0.0, neginf=0.0)
            
            tangent_intrinsic = log_vecs[..., 1:]  # Spatial components

            # Neural update in tangent space
            update = self.mlp(tangent_intrinsic).to(states.dtype) * self.update_scale * dt
            
            # Clamp update magnitude
            update_norm = torch.norm(update, dim=-1, keepdim=True)
            update = update * torch.clamp(1.0 / (update_norm + 1e-8), max=1.0)

            # Reconstruct tangent vector
            tangent_full = torch.zeros(n, self.ambient_dim, device=device, dtype=states.dtype)
            tangent_full[..., 1:] = update
            tangent_full = self.substrate.riemannian_grad(states, tangent_full)

            # Exp map back to manifold
            new_states = self.substrate.exp_map(states, tangent_full)
            
            # Final projection and radius clamp
            new_states = self.substrate.proj(new_states)
            new_states = self.clamp_to_ball(new_states)

            return new_states

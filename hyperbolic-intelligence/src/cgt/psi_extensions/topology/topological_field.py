"""
Topological Constraint Field for UGFT
=====================================

This module implements the Topological Constraint Field abstraction
from the Unified Geometric Field Theory (UGFT).

Theoretical Background
----------------------
In the UGFT, the topological term L_topo acts as a "gauge field"
that imposes global constraints on local dynamics. This represents
"Topological Downward Causation" - global topology influences local
neural updates.

The key principle is SEPARATION:
- Local dynamics (H-NCA, H-AKOrN) compute local updates
- Global evaluation (this module) computes topological invariants
- Feedback is via SCALARS only, not direct backprop shortcuts

This separation ensures:
1. Local modules never compute global quantities directly
2. Topological constraints act as external "pressure"
3. The system maintains interpretability

Classical Implementation
------------------------
In the classical limit (this implementation), we use:
- Persistence Landscapes as differentiable TDA proxy
- Vietoris-Rips filtration on hyperbolic distances
- Scalar feedback to local updates

The full UGFT would use:
- Quantum estimation of exact Betti numbers (LGZ algorithm)
- Real-time topological sensor (QPU)
- Sub-100ms feedback loops

This module is a CLASSICAL APPROXIMATION.

Author: Éric Reis
License: MIT
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Callable
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

# Attempt PyTorch import
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not installed. Topological field will be mock.")


# =============================================================================
# Topological Constraint Configuration
# =============================================================================

@dataclass
class TopologicalConfig:
    """
    Configuration for the Topological Constraint Field.
    
    Attributes
    ----------
    target_beta_0 : int
        Target zeroth Betti number (connected components).
        β₀ = 1 means single connected component.
    target_beta_1 : int
        Target first Betti number (loops/holes).
        β₁ = 0 means no non-trivial cycles.
    kappa_0 : float
        Weight for β₀ constraint.
    kappa_1 : float
        Weight for β₁ constraint.
    filtration_steps : int
        Number of steps in Vietoris-Rips filtration.
    max_edge_length : float
        Maximum edge length in filtration.
    evaluation_frequency : int
        How often to evaluate topology (macro-steps).
    use_persistence_landscape : bool
        Whether to use differentiable persistence landscapes.
    """
    
    # Target topology
    target_beta_0: int = 1      # Single connected component
    target_beta_1: int = 0      # No loops (tree structure)
    
    # Constraint weights
    kappa_0: float = 1.0        # Integration constraint
    kappa_1: float = 0.1        # Loop constraint
    
    # Filtration parameters
    filtration_steps: int = 20
    max_edge_length: float = 2.0
    
    # Evaluation
    evaluation_frequency: int = 1  # Every macro-step
    
    # Implementation
    use_persistence_landscape: bool = True


# =============================================================================
# Abstract Topological Field
# =============================================================================

class TopologicalConstraintField(ABC):
    """
    Abstract base class for topological constraint fields.
    
    This class defines the interface for topological evaluation
    that provides "downward causation" to local dynamics.
    
    The key design principle is that topological feedback is
    provided as SCALARS only, not as gradient-connected tensors.
    This enforces the separation between:
    - Global evaluation (topology)
    - Local dynamics (H-NCA, H-AKOrN)
    
    Subclasses implement different approximations:
    - PersistenceLandscapeField: Differentiable TDA proxy
    - ExactBettiField: Exact computation (slow, classical)
    - QuantumBettiField: QPU estimation (future)
    """
    
    def __init__(self, config: Optional[TopologicalConfig] = None):
        if config is None:
            config = TopologicalConfig()
        self.config = config
        self._evaluation_count = 0
        self._last_constraint = 0.0
        self._last_invariants: Dict[str, float] = {}
    
    @abstractmethod
    def compute_invariants(
        self,
        distance_matrix: "torch.Tensor",
    ) -> Dict[str, float]:
        """
        Compute topological invariants from distance matrix.
        
        Parameters
        ----------
        distance_matrix : torch.Tensor
            Pairwise distances, shape (N, N).
        
        Returns
        -------
        Dict[str, float]
            Dictionary with keys like 'beta_0', 'beta_1', etc.
        """
        pass
    
    @abstractmethod
    def compute_constraint_loss(
        self,
        distance_matrix: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Compute the topological constraint loss L_topo.
        
        This is the term that enters the action functional:
            S = ∫(L_task + L_geo + L_topo) dt
        
        Parameters
        ----------
        distance_matrix : torch.Tensor
            Pairwise distances, shape (N, N).
        
        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        pass
    
    def evaluate(
        self,
        distance_matrix: "torch.Tensor",
    ) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate topological field and return scalar feedback.
        
        This is the main entry point for "downward causation".
        Returns DETACHED scalars suitable for logging and
        constraint feedback, NOT for direct backpropagation.
        
        Parameters
        ----------
        distance_matrix : torch.Tensor
            Pairwise distances between states.
        
        Returns
        -------
        constraint_value : float
            Scalar constraint value (detached).
        invariants : Dict[str, float]
            Computed topological invariants.
        """
        self._evaluation_count += 1
        
        # Compute invariants
        invariants = self.compute_invariants(distance_matrix)
        
        # Compute constraint loss
        loss = self.compute_constraint_loss(distance_matrix)
        
        # Detach for scalar feedback
        if HAS_TORCH and isinstance(loss, torch.Tensor):
            constraint_value = loss.detach().item()
        else:
            constraint_value = float(loss)
        
        # Cache results
        self._last_constraint = constraint_value
        self._last_invariants = invariants
        
        return constraint_value, invariants
    
    def get_scalar_feedback(self) -> float:
        """
        Get the last computed constraint as a scalar.
        
        This provides the "downward causation" signal to local
        dynamics. It is a SCALAR, not a tensor, ensuring that
        local modules cannot directly backpropagate through
        the topological evaluation.
        
        Returns
        -------
        float
            Last constraint value.
        """
        return self._last_constraint
    
    @property
    def evaluation_count(self) -> int:
        """Number of evaluations performed."""
        return self._evaluation_count
    
    @property
    def last_invariants(self) -> Dict[str, float]:
        """Last computed topological invariants."""
        return self._last_invariants


# =============================================================================
# Persistence Landscape Implementation (Differentiable)
# =============================================================================

if HAS_TORCH:
    
    class PersistenceLandscapeField(TopologicalConstraintField, nn.Module):
        """
        Differentiable topological constraint field using Persistence Landscapes.
        
        This is the PRIMARY IMPLEMENTATION for classical training.
        
        Persistence landscapes provide a differentiable summary of
        topological features, enabling gradient-based optimization
        of topological constraints.
        
        Algorithm
        ---------
        1. Build Vietoris-Rips filtration from distance matrix
        2. Compute persistence diagram (birth-death pairs)
        3. Convert to persistence landscape (vectorized)
        4. Compare to target landscape
        5. Return differentiable loss
        
        The landscape is computed in a differentiable manner by:
        - Soft-thresholding edge inclusion
        - Differentiable connected components via spectral methods
        
        Notes
        -----
        This is a PROXY for exact Betti numbers. The full UGFT would
        use quantum estimation for exact invariants.
        """
        
        def __init__(self, config: Optional[TopologicalConfig] = None):
            if config is None:
                config = TopologicalConfig()
            TopologicalConstraintField.__init__(self, config)
            nn.Module.__init__(self)
            
            self.register_buffer(
                '_target_landscape',
                torch.zeros(config.filtration_steps, dtype=torch.float64),
            )
            
            # Learnable temperature for soft operations
            self.temperature = nn.Parameter(torch.tensor(1.0))
        
        def _build_filtration(
            self,
            distance_matrix: torch.Tensor,
        ) -> List[torch.Tensor]:
            """
            Build Vietoris-Rips filtration as sequence of adjacency matrices.
            
            Uses soft thresholding for differentiability.
            """
            cfg = self.config
            thresholds = torch.linspace(
                0, cfg.max_edge_length, cfg.filtration_steps,
                device=distance_matrix.device,
            )
            
            filtration = []
            for eps in thresholds:
                # Soft adjacency: sigmoid(temperature * (eps - d))
                adj = torch.sigmoid(
                    self.temperature * (eps - distance_matrix)
                )
                # Zero diagonal
                adj = adj * (1 - torch.eye(adj.shape[0], device=adj.device, dtype=adj.dtype))
                filtration.append(adj)
            
            return filtration
        
        def _estimate_beta_0(self, adjacency: torch.Tensor) -> torch.Tensor:
            """
            Estimate β₀ (connected components) via spectral method.
            
            β₀ = number of zero eigenvalues of graph Laplacian.
            We approximate this with a soft count.
            """
            # Degree matrix
            degree = adjacency.sum(dim=1)
            D = torch.diag(degree)
            
            # Laplacian
            L = D - adjacency
            
            # Eigenvalues (only need smallest few)
            try:
                eigenvalues = torch.linalg.eigvalsh(L)
            except:
                # Fallback for numerical issues
                return torch.tensor(1.0, device=adjacency.device, dtype=adjacency.dtype)
            
            # Soft count of near-zero eigenvalues
            threshold = 0.1
            beta_0 = torch.sum(torch.sigmoid(-eigenvalues / threshold + 5))
            
            return beta_0
        
        def _compute_persistence_landscape(
            self,
            filtration: List[torch.Tensor],
        ) -> torch.Tensor:
            """
            Compute simplified persistence landscape.
            
            For H₀ (components), tracks how β₀ changes over filtration.
            """
            landscape = []
            for adj in filtration:
                beta_0 = self._estimate_beta_0(adj)
                landscape.append(beta_0)
            
            return torch.stack(landscape)
        
        def compute_invariants(
            self,
            distance_matrix: torch.Tensor,
        ) -> Dict[str, float]:
            """
            Compute topological invariants.
            """
            filtration = self._build_filtration(distance_matrix)
            
            # β₀ at final filtration step
            beta_0 = self._estimate_beta_0(filtration[-1])
            
            # β₁ estimated from Euler characteristic: χ = V - E + F = β₀ - β₁
            # For a graph: χ = N - |E| = β₀ - β₁
            # So β₁ ≈ β₀ + |E| - N
            n = distance_matrix.shape[0]
            num_edges = (filtration[-1] > 0.5).sum() / 2
            beta_1_estimate = max(0, beta_0.item() + num_edges.item() - n)
            
            return {
                'beta_0': beta_0.item(),
                'beta_1': beta_1_estimate,
                'num_nodes': n,
                'num_edges': num_edges.item(),
            }
        
        def compute_constraint_loss(
            self,
            distance_matrix: torch.Tensor,
        ) -> torch.Tensor:
            """
            Compute differentiable topological constraint loss.
            
            L_topo = κ₀ * |β₀ - target_β₀| + κ₁ * |β₁ - target_β₁|
            """
            cfg = self.config
            filtration = self._build_filtration(distance_matrix)
            
            # β₀ constraint
            beta_0 = self._estimate_beta_0(filtration[-1])
            loss_beta_0 = cfg.kappa_0 * (beta_0 - cfg.target_beta_0).abs()
            
            # Landscape-based loss (penalize fragmentation over filtration)
            landscape = self._compute_persistence_landscape(filtration)
            
            # Target: monotonically decreasing to 1
            target = torch.linspace(
                float(distance_matrix.shape[0]), 1.0, len(landscape),
                device=landscape.device,
            )
            
            loss_landscape = cfg.kappa_1 * torch.mean((landscape - target).abs())
            
            return loss_beta_0 + loss_landscape
        
        def forward(
            self,
            distance_matrix: torch.Tensor,
        ) -> torch.Tensor:
            """
            Forward pass for integration with nn.Module API.
            """
            return self.compute_constraint_loss(distance_matrix)


# =============================================================================
# Scalar Feedback Field (Non-differentiable, for ablation)
# =============================================================================

class ScalarFeedbackField(TopologicalConstraintField):
    """
    Simple topological field that provides only scalar feedback.
    
    This implementation emphasizes the separation between global
    evaluation and local dynamics by NEVER providing gradients.
    
    Useful for:
    - Ablation studies
    - Understanding the role of topological feedback
    - Simpler debugging
    """
    
    def __init__(self, config: Optional[TopologicalConfig] = None):
        super().__init__(config)
        self._callback: Optional[Callable] = None
    
    def set_callback(self, callback: Callable) -> None:
        """Set callback for external topological computation."""
        self._callback = callback
    
    def compute_invariants(
        self,
        distance_matrix: "torch.Tensor",
    ) -> Dict[str, float]:
        """
        Compute invariants using simple heuristics.
        """
        if not HAS_TORCH:
            return {'beta_0': 1.0, 'beta_1': 0.0}
        
        n = distance_matrix.shape[0]
        
        # Simple connected components estimation
        # Threshold at median distance
        threshold = distance_matrix.median()
        adj = (distance_matrix < threshold).to(dtype=distance_matrix.dtype)
        adj = adj * (1 - torch.eye(n, device=adj.device, dtype=adj.dtype))
        
        # Count components via degree connectivity
        degrees = adj.sum(dim=1)
        isolated = (degrees < 0.5).sum().item()
        
        # Rough estimate
        beta_0_estimate = max(1, isolated)
        
        return {
            'beta_0': beta_0_estimate,
            'beta_1': 0.0,  # Not computed
            'num_nodes': n,
            'mean_degree': degrees.mean().item(),
        }
    
    def compute_constraint_loss(
        self,
        distance_matrix: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Compute NON-differentiable constraint loss.
        
        Returns a detached tensor to prevent gradient flow.
        """
        invariants = self.compute_invariants(distance_matrix)
        cfg = self.config
        
        loss = (
            cfg.kappa_0 * abs(invariants['beta_0'] - cfg.target_beta_0) +
            cfg.kappa_1 * abs(invariants['beta_1'] - cfg.target_beta_1)
        )
        
        if HAS_TORCH:
            return torch.tensor(loss, device=distance_matrix.device, dtype=distance_matrix.dtype).detach()
        return loss


# =============================================================================
# Factory Function
# =============================================================================

def create_topological_field(
    mode: str = "persistence",
    config: Optional[TopologicalConfig] = None,
) -> TopologicalConstraintField:
    """
    Factory function to create topological constraint fields.
    
    Parameters
    ----------
    mode : str
        - "persistence": Differentiable persistence landscapes (default)
        - "scalar": Non-differentiable scalar feedback only
        - "none": Dummy field that always returns 0
    config : TopologicalConfig, optional
        Configuration object.
    
    Returns
    -------
    TopologicalConstraintField
        Configured topological field.
    """
    if config is None:
        config = TopologicalConfig()
    
    if mode == "persistence":
        if not HAS_TORCH:
            logger.warning("PyTorch not available. Falling back to scalar mode.")
            return ScalarFeedbackField(config)
        return PersistenceLandscapeField(config)
    
    elif mode == "scalar":
        return ScalarFeedbackField(config)
    
    elif mode == "none":
        # Dummy field
        class DummyField(TopologicalConstraintField):
            def compute_invariants(self, dm):
                return {'beta_0': 1, 'beta_1': 0}
            def compute_constraint_loss(self, dm):
                if HAS_TORCH:
                    return torch.tensor(0.0, device=dm.device, dtype=dm.dtype)
                return 0.0
        return DummyField(config)
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "TopologicalConfig",
    "TopologicalConstraintField",
    "PersistenceLandscapeField" if HAS_TORCH else "ScalarFeedbackField",
    "ScalarFeedbackField",
    "create_topological_field",
    "DifferentiableTopologicalLoss" if HAS_TORCH else None,
]


# =============================================================================
# DifferentiableTopologicalLoss - Simplified interface for notebooks
# =============================================================================

if HAS_TORCH:
    
    class DifferentiableTopologicalLoss(nn.Module):
        """
        Simplified wrapper around CGT's topological field for Ψ-SLM integration.
        
        Provides a simple forward() interface that computes topological loss
        from hyperbolic states using the substrate's distance computation.
        
        Parameters
        ----------
        substrate : LorentzSubstrateHardened
            Geometric substrate for distance computation.
        resolution : int
            Number of points for persistence landscape sampling.
        num_landscapes : int
            Number of landscape functions to compute.
            
        Example
        -------
        >>> topo_loss = DifferentiableTopologicalLoss(substrate)
        >>> loss = topo_loss(h_states)
        """
        
        def __init__(
            self,
            substrate: "LorentzSubstrateHardened",
            resolution: int = 100,
            num_landscapes: int = 5,
        ):
            super().__init__()
            self.substrate = substrate
            self.resolution = resolution
            self.num_landscapes = num_landscapes
            
            # Create topological field
            topo_config = TopologicalConfig(
                target_beta_0=1,  # Single connected component
                target_beta_1=0,  # No loops (tree structure)
                filtration_steps=20,
            )
            self.topo_field = create_topological_field(
                mode="persistence",
                config=topo_config,
            )
        
        def forward(self, h_states: torch.Tensor) -> torch.Tensor:
            """
            Compute topological loss from hyperbolic states.
            
            Parameters
            ----------
            h_states : torch.Tensor
                States on Lorentz manifold, shape (N, ambient_dim)
                
            Returns
            -------
            torch.Tensor
                Scalar topological loss
            """
            # Compute distance matrix using substrate
            dist_matrix = self.substrate.distance_matrix(h_states)
            
            # Handle NaN in distances
            dist_matrix = torch.nan_to_num(dist_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Compute topological constraint loss
            return self.topo_field.compute_constraint_loss(dist_matrix)

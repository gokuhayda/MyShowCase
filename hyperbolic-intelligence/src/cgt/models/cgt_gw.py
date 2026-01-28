"""
CGT-GW: Contrastive Geometric Training with Gromov-Wasserstein
==============================================================

This module implements the CGTGW model, the main architecture for
geometric transfer learning using hyperbolic embeddings.

Architecture
------------
- h_states: Learned initial embeddings (optimized via gradient descent)
- h_nca: Learned dynamics operator (Hyperbolic Neural Cellular Automata)
- h_akorn: Phase synchronization for binding (Hyperbolic Kuramoto)

The optimizer updates h_states based on loss gradients.
NO manual state updates during forward pass.

This represents a Ψ-SLM (Psi Substrate Language Model) where:
- Initial conditions are learned parameters
- Dynamics are learned operators
- Structure is guided by Gromov-Wasserstein + topology

Author: Éric Reis
License: MIT
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

try:
    import geoopt
    HAS_GEOOPT = True
except ImportError:
    HAS_GEOOPT = False

from cgt.geometry.lorentz_hardened import LorentzSubstrateHardened, LorentzConfig
from cgt.psi_extensions.binding import HAKORNLayer, PhaseCoherenceLoss
from cgt.psi_extensions.dynamics import HyperbolicNCAWrapper
from cgt.psi_extensions.topology import DifferentiableTopologicalLoss
from cgt.psi_extensions.transfer import entropic_gromov_wasserstein_loss


@dataclass
class CGTGWConfig:
    """Configuration for CGTGW model."""
    
    num_nodes: int = 63
    embed_dim: int = 16
    hidden_dim: int = 64
    coupling_strength: float = 1.0
    temperature: float = 0.5
    curvature: float = 1.0
    learnable_curvature: bool = False
    r_start: float = 0.5  # Initial hyperbolic radius


def stable_initialization(
    n_nodes: int,
    embed_dim: int,
    substrate: LorentzSubstrateHardened,
    r_start: float = 0.5,
) -> "geoopt.ManifoldParameter":
    """
    Initialize hyperbolic states with geoopt ManifoldParameter.
    
    Uses CGT's LorentzSubstrate for exp_map.
    
    Parameters
    ----------
    n_nodes : int
        Number of nodes.
    embed_dim : int
        Intrinsic dimension.
    substrate : LorentzSubstrateHardened
        Geometric substrate.
    r_start : float
        Initial radius in tangent space.
        
    Returns
    -------
    geoopt.ManifoldParameter
        Initialized states on Lorentz manifold.
    """
    if not HAS_GEOOPT:
        raise ImportError("geoopt is required for ManifoldParameter")
    
    K = substrate.K.item() if isinstance(substrate.K, torch.Tensor) else substrate.K
    manifold = geoopt.Lorentz(k=K)

    # Initialize in tangent space at origin
    # ═══════════════════════════════════════════════════════════════════
    # DTYPE INHERITANCE: Use substrate's dtype (float64 for CGT-GW)
    # ═══════════════════════════════════════════════════════════════════
    dtype = torch.float64  # CGT-GW always uses float64
    device = substrate.K.device if isinstance(substrate.K, torch.Tensor) else torch.device('cpu')
    
    v = torch.randn(n_nodes, embed_dim, dtype=dtype, device=device)
    v = v * r_start / (torch.norm(v, dim=-1, keepdim=True) + 1e-8)

    # Construct tangent vector at origin (time component = 0)
    v_full = torch.zeros(n_nodes, embed_dim + 1, dtype=dtype, device=device)
    v_full[:, 1:] = v

    # Get origin and exp map
    origin = substrate.origin(n_nodes)
    v_full = v_full.to(origin.device)

    # Project to tangent space and then to manifold
    v_tangent = substrate.riemannian_grad(origin, v_full)
    x = substrate.exp_map(origin, v_tangent)

    return geoopt.ManifoldParameter(x, manifold=manifold)


class CGTGW(nn.Module):
    """
    CGT-GW: Contrastive Geometric Training with Gromov-Wasserstein.
    
    A geometric dynamical system for learning hyperbolic embeddings
    that preserve hierarchical structure.
    
    Architecture
    ------------
    - h_states: Learned initial embeddings (ManifoldParameter)
    - h_nca: Hyperbolic NCA for local dynamics
    - h_akorn: Kuramoto oscillators for phase binding
    
    The model learns:
    1. Optimal initial conditions (h_states)
    2. Optimal dynamics (h_nca weights)
    3. Optimal coupling (h_akorn parameters)
    
    Such that running the dynamics produces the target structure.
    
    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    embed_dim : int
        Intrinsic dimension of hyperbolic space.
    hidden_dim : int
        Hidden dimension of NCA MLP.
    coupling_strength : float
        Kuramoto coupling strength K₀.
    temperature : float
        Distance decay scale τ.
    curvature : float
        Hyperbolic curvature K.
        
    Example
    -------
    >>> model = CGTGW(num_nodes=63, embed_dim=16)
    >>> outputs = model(adjacency, num_steps=5)
    >>> losses = model.compute_losses(outputs, target_distances)
    """
    
    def __init__(
        self,
        num_nodes: int,
        embed_dim: int = 16,
        hidden_dim: int = 64,
        coupling_strength: float = 1.0,
        temperature: float = 0.5,
        curvature: float = 1.0,
    ):
        super().__init__()
        
        # Store config
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        
        # CGT's Lorentz substrate (hardened version)
        lorentz_config = LorentzConfig(
            intrinsic_dim=embed_dim,
            initial_curvature=curvature,
            learnable_curvature=False,
        )
        self.substrate = LorentzSubstrateHardened(config=lorentz_config)
        
        # Initialize hyperbolic states (learned initial conditions)
        self.h_states = stable_initialization(
            n_nodes=num_nodes,
            embed_dim=embed_dim,
            substrate=self.substrate,
            r_start=0.5,
        )
        
        # H-NCA layer (learned dynamics)
        self.h_nca = HyperbolicNCAWrapper(
            substrate=self.substrate,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
        )
        
        # H-AKORN layer (phase binding)
        self.h_akorn = HAKORNLayer(
            num_nodes=num_nodes,
            coupling_strength=coupling_strength,
            temperature=temperature,
        )
        
        # Loss modules
        self.topo_loss = DifferentiableTopologicalLoss(substrate=self.substrate)
        self.coherence_loss = PhaseCoherenceLoss(critical_threshold=0.3)
    
    def forward(
        self,
        adjacency: torch.Tensor,
        num_steps: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: evolve initial states through dynamics.
        
        Parameters
        ----------
        adjacency : torch.Tensor
            Adjacency matrix, shape (N, N)
        num_steps : int
            Number of NCA steps to run.
            
        Returns
        -------
        dict
            - 'states': Final hyperbolic states
            - 'modulation_weights': Phase-based weights
            - 'order_parameter': Synchronization measure Δ
        """
        # ═══════════════════════════════════════════════════════════════════
        # DTYPE ASSERTION: Verify input is float64 (CGT-GW requirement)
        # ═══════════════════════════════════════════════════════════════════
        assert adjacency.dtype == torch.float64, \
            f"CGTGW.forward() input dtype corrupted: {adjacency.dtype}, expected float64"
        
        # Start from learned initial states
        # Gradient flows through h_states directly
        h = self.h_states
        
        # Verify h_states dtype
        assert h.dtype == torch.float64, \
            f"CGTGW h_states dtype corrupted: {h.dtype}, expected float64"
        
        # Run dynamics
        # ═══════════════════════════════════════════════════════════════════
        # DTYPE INHERITANCE: All intermediate tensors must inherit float64
        # ═══════════════════════════════════════════════════════════════════
        dtype = h.dtype
        device = h.device
        
        modulation_weights = None
        order_param = torch.tensor(0.0, device=device, dtype=dtype)
        
        for _ in range(num_steps):
            modulation_weights, order_param = self.h_akorn(h, self.substrate, adjacency)
            
            # Verify modulation_weights dtype
            assert modulation_weights.dtype == dtype, \
                f"DTYPE COLLISION in AKORN: {modulation_weights.dtype} vs {dtype}"
            
            h = self.h_nca(h, adjacency, modulation_weights)
            
            # Verify h dtype after NCA
            assert h.dtype == dtype, \
                f"DTYPE COLLISION in NCA: {h.dtype} vs {dtype}"
        
        # NO manual state update - optimizer handles h_states via gradients
        
        return {
            'states': h,
            'modulation_weights': modulation_weights,
            'order_parameter': order_param,
        }
    
    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        target_distances: torch.Tensor,
        source_embeddings: Optional[torch.Tensor] = None,
        lambda_topo: float = 0.1,
        lambda_gw: float = 0.05,
        lambda_coherence: float = 0.01,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all loss components.
        
        Parameters
        ----------
        outputs : dict
            Output from forward pass.
        target_distances : torch.Tensor
            Target pairwise distances to preserve.
        source_embeddings : torch.Tensor, optional
            Teacher embeddings for GW alignment.
        lambda_topo : float
            Weight for topological loss.
        lambda_gw : float
            Weight for GW alignment loss.
        lambda_coherence : float
            Weight for phase coherence loss.
            
        Returns
        -------
        dict
            Loss components with 'total' having requires_grad=True.
        """
        h = outputs['states']
        order_param = outputs['order_parameter']
        
        # ═══════════════════════════════════════════════════════════════════
        # DTYPE ASSERTION: Verify all inputs are float64
        # ═══════════════════════════════════════════════════════════════════
        assert h.dtype == torch.float64, f"h dtype: {h.dtype}"
        assert target_distances.dtype == torch.float64, f"target_distances dtype: {target_distances.dtype}"
        
        dtype = h.dtype
        device = h.device
        
        # Task loss: distance preservation
        pred_distances = self.substrate.distance_matrix(h)
        pred_distances = torch.nan_to_num(pred_distances, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Detach max to avoid scale gradients dominating geometry
        pred_max = pred_distances.max().detach()
        target_max = target_distances.max()
        
        # Safe normalization
        if pred_max > 1e-8:
            pred_norm = pred_distances / (pred_max + 1e-8)
        else:
            pred_norm = pred_distances
            
        if target_max > 1e-8:
            target_norm = target_distances / (target_max + 1e-8)
        else:
            target_norm = target_distances
        
        task_loss = torch.mean((pred_norm - target_norm) ** 2)
        
        # Topological loss
        if lambda_topo > 0:
            topo_loss = self.topo_loss(h)
        else:
            topo_loss = torch.tensor(0.0, device=device, dtype=dtype)
        
        # GW alignment loss
        if source_embeddings is not None and lambda_gw > 0:
            # ═══════════════════════════════════════════════════════════════
            # DTYPE ASSERTION: Verify source_embeddings before GW
            # ═══════════════════════════════════════════════════════════════
            assert source_embeddings.dtype == dtype, \
                f"DTYPE COLLISION in GW: source {source_embeddings.dtype} vs target {dtype}"
            
            gw_loss = entropic_gromov_wasserstein_loss(
                source_embeddings.to(dtype), h, self.substrate
            )
        else:
            gw_loss = torch.tensor(0.0, device=device, dtype=dtype)
        
        # Coherence loss
        if torch.isfinite(order_param):
            coh_loss = self.coherence_loss(order_param)
        else:
            coh_loss = torch.tensor(0.0, device=device, dtype=dtype)
        
        # Total loss
        total = (task_loss
                 + lambda_topo * topo_loss
                 + lambda_gw * gw_loss
                 + lambda_coherence * coh_loss)
        
        # NaN protection
        total = torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)
        
        return {
            'total': total,
            'task': task_loss,
            'topo': topo_loss,
            'gw': gw_loss,
            'coherence': coh_loss,
            'order_param': order_param,
        }


__all__ = [
    "CGTGW",
    "CGTGWConfig",
    "stable_initialization",
]

"""
Geometric Transfer Learning via Entropic Gromov-Wasserstein
===========================================================

Implementation of geometric alignment between Euclidean (teacher) and
hyperbolic (student) representations using the Gromov-Wasserstein distance.

The GW distance compares metric spaces by finding an optimal transport plan
that preserves pairwise distances. This allows knowledge transfer from
Euclidean pre-trained models without direct weight transfer.

Theory
------
Given source metric (X, d_X) and target metric (Y, d_Y), the GW distance is:

    GW(X,Y) = min_π Σ_{i,j,k,l} |d_X(x_i, x_j) - d_Y(y_k, y_l)|² π_{ik} π_{jl}

The entropic regularization adds -ε H(π) to make the optimization tractable
via the Sinkhorn algorithm.

References
----------
- Mémoli (2011): Gromov-Wasserstein distances
- Python Optimal Transport (POT) library
- Geometric Control Manifolds paper (Section 7)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import ot

# F1/GW CORRECTION: Use hardened substrate with safe_acosh
from cgt.geometry.lorentz_hardened import LorentzSubstrateHardened as LorentzSubstrate


def compute_euclidean_cost_matrix(
    embeddings: torch.Tensor,
    p: int = 2,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute intra-domain cost matrix for Euclidean embeddings.
    
    Parameters
    ----------
    embeddings : torch.Tensor
        Euclidean embeddings, shape (N, D).
    p : int
        Minkowski p-norm (default 2 for L2).
    normalize : bool
        Whether to normalize to [0, 1] range.
    
    Returns
    -------
    torch.Tensor
        Cost matrix, shape (N, N).
    """
    C = torch.cdist(embeddings, embeddings, p=float(p))
    
    if normalize:
        # Detach max to avoid gradients through normalization
        C = C / (C.max().detach() + 1e-8)
    
    return C


def compute_hyperbolic_cost_matrix(
    h_states: torch.Tensor,
    substrate: LorentzSubstrate,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute intra-domain cost matrix for hyperbolic states.
    
    Parameters
    ----------
    h_states : torch.Tensor
        Hyperbolic states, shape (N, D+1).
    substrate : LorentzSubstrate
        Geometric substrate for geodesic computation.
    normalize : bool
        Whether to normalize to [0, 1] range.
    
    Returns
    -------
    torch.Tensor
        Cost matrix, shape (N, N).
    """
    # Use substrate's distance_matrix for numerical stability
    C = substrate.distance_matrix(h_states)
    
    # Replace NaN/Inf with zeros (self-distances and numerical issues)
    C = torch.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    
    if normalize:
        max_val = C.max().detach()
        if max_val > 1e-8:
            C = C / (max_val + 1e-8)
        else:
            # If all distances are ~0, return zeros
            C = torch.zeros_like(C)
    
    return C


def entropic_gromov_wasserstein_loss(
    source_embeddings: torch.Tensor,
    target_hyperbolic_states: torch.Tensor,
    substrate: LorentzSubstrate,
    epsilon: float = 0.1,  # Increased for stability
    max_iter: int = 100,
    source_weights: Optional[torch.Tensor] = None,
    target_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute Entropic Gromov-Wasserstein loss for geometric alignment.
    
    Measures structural discrepancy between Euclidean source geometry
    and hyperbolic target geometry. Gradients flow through the hyperbolic
    states to align the learned manifold with the teacher's structure.
    
    Parameters
    ----------
    source_embeddings : torch.Tensor
        Embeddings from teacher model (Euclidean), shape (N, D_e).
    target_hyperbolic_states : torch.Tensor
        States in Lorentz manifold (student), shape (N, D_h).
    substrate : LorentzSubstrate
        Geometric substrate for hyperbolic distance computation.
    epsilon : float
        Entropic regularization coefficient (default 0.1 for stability).
        Higher values = more diffuse transport, easier optimization.
        Lower values = sharper transport, closer to exact GW.
    max_iter : int
        Maximum Sinkhorn iterations.
    source_weights : torch.Tensor, optional
        Mass distribution on source points. Default uniform.
    target_weights : torch.Tensor, optional
        Mass distribution on target points. Default uniform.
    
    Returns
    -------
    torch.Tensor
        GW distance (scalar, differentiable).
    
    Notes
    -----
    The normalization of cost matrices is critical for numerical stability
    of the Sinkhorn algorithm (prevents exp underflow in Gibbs kernel).
    """
    device = source_embeddings.device
    N = source_embeddings.shape[0]
    
    # 1. Compute intra-domain cost matrices (ensure float64 for POT stability)
    C_src = compute_euclidean_cost_matrix(source_embeddings, normalize=True).to(torch.float64)
    C_tgt = compute_hyperbolic_cost_matrix(target_hyperbolic_states, substrate, normalize=True).to(torch.float64)
    
    # Check for NaN/Inf in cost matrices
    if not torch.isfinite(C_src).all() or not torch.isfinite(C_tgt).all():
        # Return zero loss if numerical issues
        return torch.tensor(0.0, device=device, dtype=source_embeddings.dtype, requires_grad=True)
    
    # 2. Define mass distributions (uniform if not specified)
    if source_weights is None:
        p = torch.ones(N, device=device, dtype=torch.float64) / N
    else:
        p = (source_weights / source_weights.sum()).to(torch.float64)
    
    if target_weights is None:
        q = torch.ones(N, device=device, dtype=torch.float64) / N
    else:
        q = (target_weights / target_weights.sum()).to(torch.float64)
    
    # 3. Compute entropic GW via POT
    # entropic_gromov_wasserstein2 returns the distance value
    # The squared loss |C_src - C_tgt|² is the default
    try:
        gw_dist = ot.gromov.entropic_gromov_wasserstein2(
            C_src,
            C_tgt,
            p,
            q,
            loss_fun='square_loss',
            epsilon=epsilon,
            max_iter=max_iter,
            verbose=False,
            log=False,
        )
        
        # Check if result is valid
        if not torch.isfinite(gw_dist):
            return torch.tensor(0.0, device=device, dtype=source_embeddings.dtype, requires_grad=True)
            
        return gw_dist.to(source_embeddings.dtype)
        
    except Exception as e:
        # Return zero if POT fails
        return torch.tensor(0.0, device=device, dtype=source_embeddings.dtype, requires_grad=True)


class GromovWassersteinLoss(nn.Module):
    """
    Module wrapper for GW alignment loss.
    
    Provides convenient interface with configurable parameters.
    
    Parameters
    ----------
    substrate : LorentzSubstrate
        Geometric substrate.
    epsilon : float
        Entropic regularization (default 0.1 for stability).
    max_iter : int
        Sinkhorn iterations.
    """
    
    def __init__(
        self,
        substrate: LorentzSubstrate,
        epsilon: float = 0.1,  # Increased for stability
        max_iter: int = 100,
    ):
        super().__init__()
        self.substrate = substrate
        self.epsilon = epsilon
        self.max_iter = max_iter
    
    def forward(
        self,
        source_embeddings: torch.Tensor,
        target_hyperbolic_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute GW loss.
        
        Parameters
        ----------
        source_embeddings : torch.Tensor
            Euclidean embeddings from teacher.
        target_hyperbolic_states : torch.Tensor
            Hyperbolic states from student.
        
        Returns
        -------
        torch.Tensor
            GW distance.
        """
        return entropic_gromov_wasserstein_loss(
            source_embeddings,
            target_hyperbolic_states,
            self.substrate,
            epsilon=self.epsilon,
            max_iter=self.max_iter,
        )


def compute_gw_divergence(
    states_a: torch.Tensor,
    states_b: torch.Tensor,
    substrate: LorentzSubstrate,
    epsilon: float = 0.01,
    max_iter: int = 100,
) -> torch.Tensor:
    """
    Compute GW divergence between two hyperbolic configurations.
    
    Used for the Divergence Test to measure how much two systems
    differ in their learned geometry.
    
    Parameters
    ----------
    states_a : torch.Tensor
        Hyperbolic states from system A, shape (N, D+1).
    states_b : torch.Tensor
        Hyperbolic states from system B, shape (N, D+1).
    substrate : LorentzSubstrate
        Geometric substrate.
    epsilon : float
        Entropic regularization.
    max_iter : int
        Sinkhorn iterations.
    
    Returns
    -------
    torch.Tensor
        GW divergence (scalar).
    """
    N = states_a.shape[0]
    device = states_a.device
    
    # Compute hyperbolic cost matrices for both systems
    C_a = compute_hyperbolic_cost_matrix(states_a, substrate, normalize=True)
    C_b = compute_hyperbolic_cost_matrix(states_b, substrate, normalize=True)
    
    # Uniform distributions
    p = torch.ones(N, device=device, dtype=torch.float64) / N
    q = torch.ones(N, device=device, dtype=torch.float64) / N
    
    # Compute GW distance between the two configurations
    gw_div = ot.gromov.entropic_gromov_wasserstein2(
        C_a,
        C_b,
        p,
        q,
        loss_fun='square_loss',
        epsilon=epsilon,
        max_iter=max_iter,
        verbose=False,
        log=False,
    )
    
    return gw_div


class GeometricTransferTrainer:
    """
    Training coordinator for geometric transfer learning.
    
    Implements the three-phase training protocol:
    1. Alignment: Match structure of Euclidean teacher
    2. Refinement: Task-specific learning with soft alignment
    3. Geometric: Full topological regularization
    
    Parameters
    ----------
    model : nn.Module
        The hyperbolic model to train.
    substrate : LorentzSubstrate
        Geometric substrate.
    teacher_embeddings : torch.Tensor
        Pre-computed Euclidean embeddings from teacher.
    epsilon : float
        GW regularization.
    """
    
    def __init__(
        self,
        model: nn.Module,
        substrate: LorentzSubstrate,
        teacher_embeddings: torch.Tensor,
        epsilon: float = 0.01,
    ):
        self.model = model
        self.substrate = substrate
        self.teacher_embeddings = teacher_embeddings
        self.gw_loss = GromovWassersteinLoss(substrate, epsilon)
    
    def compute_alignment_loss(
        self,
        hyperbolic_states: torch.Tensor,
    ) -> torch.Tensor:
        """Compute alignment loss for Phase 1."""
        return self.gw_loss(self.teacher_embeddings, hyperbolic_states)
    
    def compute_combined_loss(
        self,
        hyperbolic_states: torch.Tensor,
        task_loss: torch.Tensor,
        lambda_align: float = 0.1,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss for Phase 2.
        
        Returns
        -------
        torch.Tensor
            Total loss.
        dict
            Loss components for logging.
        """
        align_loss = self.compute_alignment_loss(hyperbolic_states)
        total = task_loss + lambda_align * align_loss
        
        return total, {
            'task_loss': task_loss.item(),
            'align_loss': align_loss.item(),
            'total_loss': total.item(),
        }

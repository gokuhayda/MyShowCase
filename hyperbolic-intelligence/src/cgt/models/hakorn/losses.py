# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.

"""
H-AKORN Loss Functions
======================

Specialized loss functions for H-AKORN training:
1. Phase synchronization regularization
2. Coupling strength regularization
3. Hyperbolic constraint regularization
4. Combined H-AKORN training loss
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseSynchronizationLoss(nn.Module):
    """
    Regularization to encourage phase synchronization.
    
    L_sync = λ * mean((1 - r)²)
    
    where r is the Kuramoto order parameter.
    
    Args:
        lambda_sync: Regularization strength
    """
    
    def __init__(self, lambda_sync: float = 0.1):
        super().__init__()
        self.lambda_sync = lambda_sync
    
    def forward(self, order_parameters: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute synchronization loss.
        
        Args:
            order_parameters: List of [B] order parameters per layer
        
        Returns:
            loss: Scalar synchronization loss
        """
        if not order_parameters:
            return torch.tensor(0.0, requires_grad=True)
        
        # Average order parameter across layers
        avg_order_param = torch.stack(order_parameters).mean(dim=0)
        
        # Penalize low synchronization
        loss = self.lambda_sync * ((1.0 - avg_order_param) ** 2).mean()
        
        return loss


class CouplingStrengthRegularizer(nn.Module):
    """
    Regularization on coupling strength to prevent instabilities.
    
    L_coupling = λ * mean(|A_ij|²)
    
    Args:
        lambda_coupling: Regularization strength
    """
    
    def __init__(self, lambda_coupling: float = 0.01):
        super().__init__()
        self.lambda_coupling = lambda_coupling
    
    def forward(self, coupling_matrices: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute coupling strength regularization.
        
        Args:
            coupling_matrices: List of coupling matrices (not currently passed, placeholder)
        
        Returns:
            loss: Scalar coupling regularization loss
        """
        # Placeholder: would need to extract coupling matrices from model
        return torch.tensor(0.0, requires_grad=True)


class PhaseCoherenceVarianceLoss(nn.Module):
    """
    Encourage diversity in phase coherence across layers.
    
    L_variance = -λ * Var(r_l)
    
    Negative variance encourages different synchronization levels per layer.
    
    Args:
        lambda_variance: Regularization strength
    """
    
    def __init__(self, lambda_variance: float = 0.05):
        super().__init__()
        self.lambda_variance = lambda_variance
    
    def forward(self, order_parameters: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute variance loss.
        
        Args:
            order_parameters: List of [B] order parameters per layer
        
        Returns:
            loss: Scalar variance loss
        """
        if len(order_parameters) < 2:
            return torch.tensor(0.0, requires_grad=True)
        
        # Stack order parameters: [L, B]
        order_stack = torch.stack(order_parameters, dim=0)
        
        # Compute variance across layers
        variance = order_stack.var(dim=0).mean()
        
        # Encourage diversity (negative of variance)
        loss = -self.lambda_variance * variance
        
        return loss


class FrequencyEntropyRegularizer(nn.Module):
    """
    Encourage diversity in natural frequencies.
    
    L_entropy = -λ * H(ω)
    
    where H is the entropy of frequency distribution.
    
    Args:
        lambda_entropy: Regularization strength
    """
    
    def __init__(self, lambda_entropy: float = 0.01):
        super().__init__()
        self.lambda_entropy = lambda_entropy
    
    def forward(self, frequencies: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy loss.
        
        Args:
            frequencies: [H] or [L, H] natural frequencies
        
        Returns:
            loss: Scalar entropy loss
        """
        # Normalize frequencies to get pseudo-probabilities
        if frequencies.dim() == 1:
            probs = F.softmax(frequencies, dim=0)
            entropy = -(probs * torch.log(probs + 1e-10)).sum()
        else:
            # Average entropy across layers
            entropies = []
            for freq_layer in frequencies:
                probs = F.softmax(freq_layer, dim=0)
                entropy = -(probs * torch.log(probs + 1e-10)).sum()
                entropies.append(entropy)
            entropy = torch.stack(entropies).mean()
        
        # Encourage high entropy (diversity)
        loss = -self.lambda_entropy * entropy
        
        return loss


class HAKORNLoss(nn.Module):
    """
    Complete training loss for H-AKORN.
    
    L_total = L_LM + λ_sync * L_sync + λ_var * L_var
    
    Where:
    - L_LM: Language modeling loss (cross-entropy)
    - L_sync: Phase synchronization regularization
    - L_var: Phase coherence variance regularization
    
    Args:
        lambda_sync: Phase synchronization regularization weight
        lambda_variance: Phase variance regularization weight
        lambda_coupling: Coupling strength regularization weight
        lambda_entropy: Frequency entropy regularization weight
    """
    
    def __init__(
        self,
        lambda_sync: float = 0.1,
        lambda_variance: float = 0.05,
        lambda_coupling: float = 0.01,
        lambda_entropy: float = 0.01,
    ):
        super().__init__()
        self.lambda_sync = lambda_sync
        self.lambda_variance = lambda_variance
        self.lambda_coupling = lambda_coupling
        self.lambda_entropy = lambda_entropy
        
        self.sync_loss = PhaseSynchronizationLoss(lambda_sync)
        self.variance_loss = PhaseCoherenceVarianceLoss(lambda_variance)
        self.coupling_loss = CouplingStrengthRegularizer(lambda_coupling)
        self.entropy_loss = FrequencyEntropyRegularizer(lambda_entropy)
    
    def forward(
        self,
        lm_loss: torch.Tensor,
        order_parameters: List[torch.Tensor],
        frequencies: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total H-AKORN loss.
        
        Args:
            lm_loss: Language modeling loss
            order_parameters: List of [B] order parameters per layer
            frequencies: Optional [L, H] natural frequencies
        
        Returns:
            losses: Dict with keys:
                - total: Total loss
                - lm: Language modeling loss
                - sync: Synchronization loss
                - variance: Variance loss
                - coupling: Coupling loss (if applicable)
                - entropy: Entropy loss (if applicable)
        """
        # Synchronization loss
        l_sync = self.sync_loss(order_parameters)
        
        # Variance loss
        l_variance = self.variance_loss(order_parameters)
        
        # Coupling loss (placeholder)
        l_coupling = self.coupling_loss([])
        
        # Entropy loss (if frequencies provided)
        l_entropy = torch.tensor(0.0, device=lm_loss.device, requires_grad=True)
        if frequencies is not None:
            l_entropy = self.entropy_loss(frequencies)
        
        # Total loss
        l_total = lm_loss + l_sync + l_variance + l_coupling + l_entropy
        
        return {
            'total': l_total,
            'lm': lm_loss.item() if isinstance(lm_loss, torch.Tensor) else lm_loss,
            'sync': l_sync.item() if isinstance(l_sync, torch.Tensor) else 0.0,
            'variance': l_variance.item() if isinstance(l_variance, torch.Tensor) else 0.0,
            'coupling': l_coupling.item() if isinstance(l_coupling, torch.Tensor) else 0.0,
            'entropy': l_entropy.item() if isinstance(l_entropy, torch.Tensor) else 0.0,
        }


class HAKORNWithHyperbolicLoss(nn.Module):
    """
    H-AKORN loss with additional hyperbolic constraints.
    
    Combines H-AKORN regularization with hyperbolic geometry constraints
    from the base project (if available).
    
    Args:
        lambda_sync: Phase synchronization weight
        lambda_variance: Phase variance weight
        lambda_manifold: Manifold constraint weight
        lambda_radius: Radius regularization weight
    """
    
    def __init__(
        self,
        lambda_sync: float = 0.1,
        lambda_variance: float = 0.05,
        lambda_manifold: float = 0.1,
        lambda_radius: float = 0.001,
    ):
        super().__init__()
        self.lambda_manifold = lambda_manifold
        self.lambda_radius = lambda_radius
        
        self.hakorn_loss = HAKORNLoss(
            lambda_sync=lambda_sync,
            lambda_variance=lambda_variance,
        )
    
    def forward(
        self,
        lm_loss: torch.Tensor,
        order_parameters: List[torch.Tensor],
        hidden_states: Optional[torch.Tensor] = None,
        substrate = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            lm_loss: Language modeling loss
            order_parameters: List of order parameters
            hidden_states: [B, L, D] hidden states (for hyperbolic constraints)
            substrate: Hyperbolic substrate (if available)
        
        Returns:
            losses: Dict with all loss components
        """
        # Base H-AKORN loss
        losses = self.hakorn_loss(lm_loss, order_parameters)
        
        # Add hyperbolic constraints if substrate available
        if substrate is not None and hidden_states is not None:
            # Manifold violation
            l_manifold = substrate.manifold_violation(hidden_states)
            losses['manifold'] = l_manifold.item()
            losses['total'] = losses['total'] + self.lambda_manifold * l_manifold
            
            # Radius regularization
            radii = substrate.lorentz_radius(hidden_states)
            l_radius = F.relu(radii - 10.0).pow(2).mean()
            losses['radius'] = l_radius.item()
            losses['total'] = losses['total'] + self.lambda_radius * l_radius
        
        return losses

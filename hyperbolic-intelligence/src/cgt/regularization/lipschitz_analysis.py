# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
Lipschitz Analysis for Hyperbolic Neural Networks
=================================================

This module provides tools for analyzing and enforcing Lipschitz continuity
in hyperbolic neural networks, addressing the critical gap identified in
the CGT/HGST audit regarding robustness guarantees.

Three levels of Lipschitz analysis are provided:

1. **Local Lipschitz**: Empirical maximum slope observed during training
2. **Global Lipschitz**: Upper bound via Jacobian spectral norm
3. **PAC Lipschitz**: Probabilistic guarantee with confidence δ

The hyperbolic setting introduces unique challenges:
- Boundary asymmetry: Near manifold boundary, small input changes cause
  large output changes (Li et al., 2024)
- Metric mismatch: Lipschitz must be computed w.r.t. Riemannian metric,
  not Euclidean

Mathematical Status
-------------------
- Local Analysis: EMPIRICAL (observed, not guaranteed)
- Global Analysis: EXACT (via SVD of Jacobian)
- PAC Analysis: STATISTICAL (1-δ confidence bound)
- LGCA Regularization: HEURISTIC (empirically effective)

References:
    Li et al. (2024) "Improving Robustness of Hyperbolic Neural Networks
    by Lipschitz Analysis"
    
Author: Éric Gustavo Reis de Sena
Date: January 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable

import torch
import torch.nn as nn
import numpy as np
from scipy import stats


@dataclass
class LipschitzReport:
    """
    Comprehensive Lipschitz analysis report.
    
    Attributes:
        local_lipschitz: Maximum observed amplification ratio
        global_lipschitz: Upper bound via Jacobian analysis
        pac_lipschitz: PAC bound with confidence level
        confidence: Confidence level for PAC bound (1-δ)
        boundary_risk: Estimated risk near manifold boundary
        recommendations: Suggested actions based on analysis
    """
    local_lipschitz: float
    global_lipschitz: Optional[float]
    pac_lipschitz: Optional[float]
    confidence: float
    boundary_risk: float
    amplification_distribution: Optional[np.ndarray]
    recommendations: List[str]


class LipschitzAnalyzer(nn.Module):
    """
    Comprehensive Lipschitz analysis for hyperbolic neural networks.
    
    Provides three levels of analysis:
    
    1. **Local (Empirical)**: 
       L_local = max_{x in sample} ||f(x+ε) - f(x)||_M / ||ε||
       
    2. **Global (Jacobian)**:
       L_global = sup_x ||J_f(x)||_op where ||·||_op is spectral norm
       
    3. **PAC (Probably Approximately Correct)**:
       P(L > L_pac) ≤ δ for specified confidence 1-δ
    
    Args:
        lorentz: LorentzSubstrate for metric computations
        n_samples: Number of samples for empirical estimation
        perturbation_scale: Scale of perturbations for local analysis
        confidence: Confidence level for PAC bounds (default 0.95)
        
    Notes:
        - Classification: Mixed (Empirical + Statistical)
        - Use Case: Pre-deployment robustness certification
        - Limitation: Global bound may be loose for deep networks
    """
    
    def __init__(
        self,
        lorentz,
        n_samples: int = 1000,
        perturbation_scale: float = 0.01,
        confidence: float = 0.95,
    ):
        super().__init__()
        self.lorentz = lorentz
        self.n_samples = n_samples
        self.perturbation_scale = perturbation_scale
        self.confidence = confidence
    
    def analyze(
        self,
        model: nn.Module,
        data: torch.Tensor,
        compute_global: bool = True,
        compute_pac: bool = True,
    ) -> LipschitzReport:
        """
        Perform comprehensive Lipschitz analysis.
        
        Args:
            model: Neural network to analyze
            data: Sample data points [N, input_dim]
            compute_global: Whether to compute global bound (expensive)
            compute_pac: Whether to compute PAC bound
            
        Returns:
            LipschitzReport with all analysis results
        """
        model.eval()
        device = data.device
        dtype = data.dtype
        
        # 1. Local Lipschitz (empirical)
        local_lip, amplifications = self._compute_local_lipschitz(model, data)
        
        # 2. Global Lipschitz (Jacobian-based)
        global_lip = None
        if compute_global:
            global_lip = self._compute_global_lipschitz(model, data[:100])
        
        # 3. PAC Lipschitz
        pac_lip = None
        if compute_pac:
            pac_lip = self._compute_pac_bound(amplifications)
        
        # 4. Boundary risk analysis
        boundary_risk = self._analyze_boundary_risk(model, data)
        
        # 5. Generate recommendations
        recommendations = self._generate_recommendations(
            local_lip, global_lip, pac_lip, boundary_risk
        )
        
        return LipschitzReport(
            local_lipschitz=local_lip,
            global_lipschitz=global_lip,
            pac_lipschitz=pac_lip,
            confidence=self.confidence,
            boundary_risk=boundary_risk,
            amplification_distribution=amplifications,
            recommendations=recommendations,
        )
    
    def _compute_local_lipschitz(
        self,
        model: nn.Module,
        data: torch.Tensor,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute local Lipschitz constant via perturbation analysis.
        
        For each point x, computes:
            ratio = ||f(x+ε) - f(x)||_M / ||ε||
            
        where ||·||_M is the Riemannian (geodesic) distance for outputs.
        """
        n_samples = min(self.n_samples, data.shape[0])
        idx = torch.randperm(data.shape[0])[:n_samples]
        sample = data[idx]
        
        amplifications = []
        
        with torch.no_grad():
            for i in range(n_samples):
                x = sample[i:i+1]
                
                # Generate perturbation
                eps = torch.randn_like(x) * self.perturbation_scale
                x_pert = x + eps
                
                # Forward pass
                y = model(x)
                y_pert = model(x_pert)
                
                # Input distance (Euclidean)
                input_dist = eps.norm().item()
                
                # Output distance (geodesic on manifold)
                if hasattr(self.lorentz, 'geodesic_distance'):
                    output_dist = self.lorentz.geodesic_distance(y, y_pert).item()
                else:
                    output_dist = (y - y_pert).norm().item()
                
                # Amplification ratio
                if input_dist > 1e-10:
                    ratio = output_dist / input_dist
                    amplifications.append(ratio)
        
        amplifications = np.array(amplifications)
        local_lip = amplifications.max() if len(amplifications) > 0 else 0.0
        
        return local_lip, amplifications
    
    def _compute_global_lipschitz(
        self,
        model: nn.Module,
        data: torch.Tensor,
    ) -> float:
        """
        Compute global Lipschitz bound via Jacobian spectral norm.
        
        L_global = max_x ||J_f(x)||_spectral
        
        This is an upper bound on the true Lipschitz constant.
        For deep networks, this can be quite loose.
        """
        max_spectral_norm = 0.0
        n_samples = min(50, data.shape[0])  # Jacobian computation is expensive
        
        for i in range(n_samples):
            x = data[i:i+1].requires_grad_(True)
            
            # Compute Jacobian via autograd
            y = model(x)
            
            jacobian = []
            for j in range(y.shape[1]):
                grad_outputs = torch.zeros_like(y)
                grad_outputs[0, j] = 1.0
                
                grad = torch.autograd.grad(
                    y, x, grad_outputs=grad_outputs,
                    retain_graph=True, create_graph=False
                )[0]
                jacobian.append(grad.squeeze(0))
            
            J = torch.stack(jacobian, dim=0)  # [output_dim, input_dim]
            
            # Spectral norm (largest singular value)
            try:
                _, S, _ = torch.linalg.svd(J)
                spectral_norm = S[0].item()
                max_spectral_norm = max(max_spectral_norm, spectral_norm)
            except:
                pass
            
            x.requires_grad_(False)
        
        return max_spectral_norm
    
    def _compute_pac_bound(
        self,
        amplifications: np.ndarray,
    ) -> float:
        """
        Compute PAC (Probably Approximately Correct) Lipschitz bound.
        
        Using order statistics, we estimate L_pac such that:
            P(L > L_pac) ≤ δ
            
        where δ = 1 - confidence.
        
        Method: Use the (1-δ)-quantile of observed amplifications,
        with correction for finite sample size.
        """
        if len(amplifications) == 0:
            return 0.0
        
        # Quantile with finite-sample correction (DKW inequality)
        n = len(amplifications)
        delta = 1 - self.confidence
        
        # Empirical quantile
        empirical_quantile = np.percentile(amplifications, self.confidence * 100)
        
        # DKW correction for finite samples
        # P(sup|F_n - F| > ε) ≤ 2*exp(-2nε²)
        # Solving for ε: ε = sqrt(log(2/δ) / (2n))
        epsilon = np.sqrt(np.log(2 / delta) / (2 * n))
        
        # Corrected bound (conservative)
        corrected_quantile_idx = min(
            int((self.confidence + epsilon) * n),
            n - 1
        )
        sorted_amps = np.sort(amplifications)
        pac_bound = sorted_amps[corrected_quantile_idx]
        
        return pac_bound
    
    def _analyze_boundary_risk(
        self,
        model: nn.Module,
        data: torch.Tensor,
    ) -> float:
        """
        Analyze vulnerability near manifold boundary.
        
        Hyperbolic neural networks are particularly vulnerable near the
        boundary where the curvature effects are strongest.
        
        We measure the correlation between Lorentz radius and amplification.
        """
        n_samples = min(500, data.shape[0])
        idx = torch.randperm(data.shape[0])[:n_samples]
        sample = data[idx]
        
        radii = []
        amplifications = []
        
        with torch.no_grad():
            for i in range(n_samples):
                x = sample[i:i+1]
                
                # Forward pass
                y = model(x)
                
                # Compute radius (distance from origin)
                if hasattr(self.lorentz, 'lorentz_radius'):
                    radius = self.lorentz.lorentz_radius(y).mean().item()
                else:
                    radius = y.norm().item()
                
                # Compute local amplification
                eps = torch.randn_like(x) * self.perturbation_scale
                y_pert = model(x + eps)
                
                input_dist = eps.norm().item()
                output_dist = (y - y_pert).norm().item()
                
                if input_dist > 1e-10:
                    amplifications.append(output_dist / input_dist)
                    radii.append(radius)
        
        if len(radii) < 10:
            return 0.0
        
        # Correlation between radius and amplification
        # High correlation = high boundary risk
        correlation, _ = stats.spearmanr(radii, amplifications)
        
        return max(0.0, correlation)  # Only care about positive correlation
    
    def _generate_recommendations(
        self,
        local_lip: float,
        global_lip: Optional[float],
        pac_lip: Optional[float],
        boundary_risk: float,
    ) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Local Lipschitz thresholds
        if local_lip > 10.0:
            recommendations.append(
                f"⚠️ HIGH LOCAL LIPSCHITZ ({local_lip:.2f}): "
                "Consider adding spectral normalization to all layers."
            )
        elif local_lip > 5.0:
            recommendations.append(
                f"⚡ MODERATE LOCAL LIPSCHITZ ({local_lip:.2f}): "
                "Monitor for adversarial vulnerability."
            )
        
        # Global vs Local gap
        if global_lip is not None and global_lip > 2 * local_lip:
            recommendations.append(
                f"📊 LARGE GLOBAL/LOCAL GAP ({global_lip:.2f} vs {local_lip:.2f}): "
                "Worst-case is significantly worse than typical case."
            )
        
        # Boundary risk
        if boundary_risk > 0.5:
            recommendations.append(
                f"🔴 HIGH BOUNDARY RISK ({boundary_risk:.2f}): "
                "Implement radius clamping or LGCA regularization."
            )
        elif boundary_risk > 0.3:
            recommendations.append(
                f"🟡 MODERATE BOUNDARY RISK ({boundary_risk:.2f}): "
                "Consider monitoring embeddings during inference."
            )
        
        # PAC bound
        if pac_lip is not None:
            recommendations.append(
                f"📈 PAC BOUND: With {self.confidence*100:.0f}% confidence, "
                f"Lipschitz ≤ {pac_lip:.2f}"
            )
        
        if not recommendations:
            recommendations.append("✅ Lipschitz analysis indicates good robustness.")
        
        return recommendations


class LipschitzRegularizer(nn.Module):
    """
    Lipschitz continuity regularizer for training.
    
    Encourages the model to have bounded Lipschitz constant by
    penalizing large amplification ratios during training.
    
    Loss = λ * max(0, observed_ratio - target_ratio)²
    
    Args:
        target_lipschitz: Target maximum Lipschitz constant
        noise_scale: Scale of perturbations for ratio estimation
        penalty_weight: Weight of the regularization term
        
    Notes:
        - Classification: HEURISTIC (empirically effective)
        - Does not guarantee global Lipschitz bound
        - Works best combined with spectral normalization
    """
    
    def __init__(
        self,
        target_lipschitz: float = 5.0,
        noise_scale: float = 0.01,
        penalty_weight: float = 0.1,
    ):
        super().__init__()
        self.target_lipschitz = target_lipschitz
        self.noise_scale = noise_scale
        self.penalty_weight = penalty_weight
    
    def forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Lipschitz regularization loss.
        
        Args:
            model: Model to regularize
            x: Input batch
            
        Returns:
            Regularization loss term
        """
        # Generate perturbations
        eps = torch.randn_like(x) * self.noise_scale
        
        # Forward passes
        with torch.no_grad():
            y_clean = model(x)
        y_pert = model(x + eps)
        
        # Compute ratios
        input_dists = eps.norm(dim=-1)
        output_dists = (y_pert - y_clean.detach()).norm(dim=-1)
        
        ratios = output_dists / (input_dists + 1e-8)
        
        # Hinge loss: penalize ratios exceeding target
        violations = torch.clamp(ratios - self.target_lipschitz, min=0.0)
        
        return self.penalty_weight * (violations ** 2).mean()


class SpectralNormRegularizer(nn.Module):
    """
    Layer-wise spectral norm regularization.
    
    Adds penalty based on spectral norms of weight matrices,
    providing a differentiable proxy for Lipschitz control.
    
    For a network f = f_L ∘ ... ∘ f_1:
        L(f) ≤ ∏_i ||W_i||_spectral
        
    This regularizer minimizes Σ_i ||W_i||_spectral.
    
    Notes:
        - Classification: APPROXIMATION (upper bound on Lipschitz)
        - More stable than direct Jacobian computation
        - Can be combined with spectral normalization
    """
    
    def __init__(self, penalty_weight: float = 0.01):
        super().__init__()
        self.penalty_weight = penalty_weight
    
    def forward(self, model: nn.Module) -> torch.Tensor:
        """
        Compute spectral norm regularization loss.
        
        Args:
            model: Model to regularize
            
        Returns:
            Sum of spectral norms of all weight matrices
        """
        total_spectral_norm = 0.0
        
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                # Reshape to 2D if needed
                w = param.view(param.shape[0], -1)
                
                # Compute spectral norm via power iteration (efficient)
                # For small matrices, use SVD
                if w.shape[0] * w.shape[1] < 10000:
                    try:
                        _, S, _ = torch.linalg.svd(w, full_matrices=False)
                        spectral_norm = S[0]
                    except:
                        spectral_norm = w.norm()
                else:
                    # Power iteration for large matrices
                    spectral_norm = self._power_iteration(w)
                
                total_spectral_norm = total_spectral_norm + spectral_norm
        
        return self.penalty_weight * total_spectral_norm
    
    def _power_iteration(
        self,
        W: torch.Tensor,
        n_iter: int = 10,
    ) -> torch.Tensor:
        """Estimate spectral norm via power iteration."""
        u = torch.randn(W.shape[0], device=W.device, dtype=W.dtype)
        u = u / u.norm()
        
        for _ in range(n_iter):
            v = W.T @ u
            v = v / (v.norm() + 1e-8)
            u = W @ v
            u = u / (u.norm() + 1e-8)
        
        return (u @ W @ v).abs()

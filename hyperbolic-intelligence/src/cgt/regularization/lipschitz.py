# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
CGT Regularization Module
=========================

Regularization techniques for geometric stability in CGT.

This module implements Lipschitz continuity regularization to ensure
the encoder mapping does not excessively amplify perturbations.

Mathematical Status
-------------------
- Lipschitz estimation: First-order approximation via finite differences
- Does NOT provide global Lipschitz guarantees
- Serves as FALSIFICATION protocol, not formal proof

Notes
-----
- Complements spectral normalization in projector layers
- Empirical local evidence, not theoretical bound
- Used for F2 stability testing

Author: Éric Gustavo Reis de Sena
Date: January 2026
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LipschitzRegularizer(nn.Module):
    """
    Lipschitz continuity regularizer for encoder stability.

    Penalizes cases where the encoder amplifies input perturbations,
    encouraging smooth mappings from Euclidean to hyperbolic space.

    Measures: max(0, d_H(f(x), f(x+ε)) / d_E(ε) - 1)

    Attributes:
        noise_scale: Standard deviation of perturbation noise.

    Notes:
        - Space: Measures mapping Lipschitz constant locally
        - Status: First-order approximation (NOT global guarantee)
        - Mathematical role: Falsification protocol (F2)
        - Complements but does not replace spectral normalization

    Limitations:
        - Only samples perturbations at noise_scale magnitude
        - Cannot guarantee Lipschitz bound holds globally
        - Batch-dependent estimate may vary across batches
    """

    def __init__(self, noise_scale: float = 0.05):
        """
        Initialize Lipschitz regularizer.

        Args:
            noise_scale: Magnitude of perturbations for testing.

        Notes:
            - noise_scale should match expected input noise level
            - Too large may cause numerical issues
            - Too small may not capture amplification behavior
        """
        super().__init__()
        self.noise_scale = noise_scale

    def forward(
        self,
        model: nn.Module,
        teacher_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Lipschitz violation penalty.

        Args:
            model: CGT student model with substrate attribute.
            teacher_emb: Teacher embeddings [B, D].

        Returns:
            Mean Lipschitz violation (scalar).

        Notes:
            - Space: Ratio of hyperbolic to Euclidean distances
            - Status: Empirical local estimate
            - Penalizes expansion ratio > 1 (amplification)
        """
        # Generate random perturbation
        noise = torch.randn_like(teacher_emb) * self.noise_scale

        # Encode original and perturbed
        with torch.no_grad():
            emb_orig = model(teacher_emb, use_homeostatic=False)
        emb_pert = model(teacher_emb + noise, use_homeostatic=False)

        # Input distance (Euclidean)
        d_input = noise.norm(dim=-1)

        # Output distance (geodesic)
        d_output = model.substrate.dist(emb_orig, emb_pert)

        # Lipschitz violation: ratio > 1 means amplification
        ratio = d_output / (d_input + 1e-8)
        violation = F.relu(ratio - 1.0)

        return violation.mean()


class SpectralNormRegularizer(nn.Module):
    """
    Spectral norm monitoring (not regularization).

    Tracks the spectral norm of weight matrices for diagnostics.
    Actual spectral normalization is applied via nn.utils.spectral_norm
    in the projector layers.

    Notes:
        - Space: Weight matrix spectral analysis
        - Status: Diagnostic tool (not a loss term)
        - Actual constraint via spectral_norm wrapper in projector
    """

    def __init__(self):
        """Initialize spectral norm monitor."""
        super().__init__()

    @torch.no_grad()
    def compute_norms(self, model: nn.Module) -> dict:
        """
        Compute spectral norms of all linear layers.

        Args:
            model: Model to analyze.

        Returns:
            Dictionary of layer names to spectral norms.

        Notes:
            - Space: Per-layer spectral analysis
            - Status: Exact (via SVD)
        """
        norms = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                try:
                    s = torch.linalg.svdvals(module.weight.float())
                    norms[name] = s[0].item()
                except RuntimeError:
                    norms[name] = float("nan")
        return norms

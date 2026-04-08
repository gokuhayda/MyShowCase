"""
cgt/dynamics/trajectory_v2.py
==================================
Isolated trajectory container and synchronization metrics for v2.
Zero imports from legacy cgt.*.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch


def compute_synchronization_index_v2(phases: torch.Tensor) -> torch.Tensor:
    """
    Kuramoto order parameter r = |<e^{iθ}>|.

    Args:
        phases: [..., N]  phase values in radians
    Returns:
        r: [...] order parameter in [0, 1]
    """
    complex_phases = torch.complex(torch.cos(phases), torch.sin(phases))
    return torch.abs(complex_phases.mean(dim=-1))


class TrajectoryV2:
    """
    Container for Kuramoto phase trajectories.

    Attributes
    ----------
    phases      : List[[B, N]] — one tensor per recorded step
    order_params: List[float]  — synchronization index per step
    """

    def __init__(self) -> None:
        self.phases: List[torch.Tensor] = []
        self.order_params: List[float] = []

    @classmethod
    def from_phase_list(cls, phase_list: List[torch.Tensor]) -> "TrajectoryV2":
        t = cls()
        for ph in phase_list:
            t.phases.append(ph)
            r = compute_synchronization_index_v2(ph).mean().item()
            t.order_params.append(r)
        return t

    def final_order_parameter(self) -> float:
        if not self.order_params:
            return 0.0
        return self.order_params[-1]

    def phase_entropy(self, n_bins: int = 16) -> float:
        """
        Entropy of phase distribution at final step.
        High entropy = diverse phases = healthy oscillator state.
        Zero entropy = collapsed (all phases equal) = gibberish mode.
        """
        if not self.phases:
            return 0.0
        ph = self.phases[-1].flatten()
        # normalize to [0, 2π)
        ph = torch.fmod(ph, 2.0 * math.pi)
        hist = torch.histc(ph, bins=n_bins, min=0.0, max=2.0 * math.pi)
        hist = hist / (hist.sum() + 1e-12)
        entropy = -(hist * (hist + 1e-12).log()).sum().item()
        return float(entropy)

    def __len__(self) -> int:
        return len(self.phases)

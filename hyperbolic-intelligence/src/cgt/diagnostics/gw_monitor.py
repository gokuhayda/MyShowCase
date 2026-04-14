# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.

"""
GW Divergence Monitor (GCM Section 10)
========================================

Measures DGW(t) = GW(M_student(t), M_teacher(t)) — the Gromov-Wasserstein
distance between student and teacher representation manifolds over time.

Prediction (GCM Section 10.3):
    Phase 1 (t < t_c): DGW ≈ 0  (both learn local features similarly)
    Phase 2 (t > t_c): DGW diverges → student converges to globally coherent
                        configuration while vanilla model stays disordered.

Usage in training hooks:
    monitor = GWDivergenceMonitor(update_every=1000)
    monitor.step(student_emb, teacher_emb, step=trainer.step)
    print(monitor.report())
"""

from __future__ import annotations
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import numpy as np


class GWDivergenceMonitor:
    """Tracks GW distance between student and teacher manifolds.

    Uses the POT library (already installed) for entropic GW computation.
    Falls back to RSA (representational similarity) if POT unavailable.

    Args:
        update_every:  Compute DGW every N steps (expensive).
        max_points:    Subsample for tractability.
        epsilon:       Entropic regularization for GW (Cuturi 2013).
        n_iter:        Sinkhorn iterations.
    """

    def __init__(
        self,
        update_every: int   = 1000,
        max_points:   int   = 128,
        epsilon:      float = 0.01,
        n_iter:       int   = 50,
    ) -> None:
        self.update_every = update_every
        self.max_points   = max_points
        self.epsilon      = epsilon
        self.n_iter       = n_iter
        self.history:     List[Dict] = []
        self._pot_ok      = self._check_pot()

    def _check_pot(self) -> bool:
        try:
            import ot  # Python Optimal Transport
            return True
        except ImportError:
            return False

    def step(
        self,
        student_emb: torch.Tensor,  # [N, D_s] student spatial embeddings
        teacher_emb: torch.Tensor,  # [N, D_t] teacher spatial embeddings
        training_step: int = 0,
    ) -> Optional[float]:
        """Compute DGW if on schedule. Returns DGW or None."""
        if training_step % self.update_every != 0:
            return None

        N = min(student_emb.shape[0], teacher_emb.shape[0], self.max_points)
        se = student_emb[:N].detach().float().cpu()
        te = teacher_emb[:N].detach().float().cpu()

        if self._pot_ok:
            dgw = self._compute_gw_pot(se, te)
        else:
            dgw = self._compute_rsa_proxy(se, te)

        record = {
            'step':    training_step,
            'dgw':     dgw,
            'backend': 'pot' if self._pot_ok else 'rsa',
        }
        self.history.append(record)
        return dgw

    def _compute_gw_pot(self, se: torch.Tensor, te: torch.Tensor) -> float:
        """Entropic GW via POT library."""
        try:
            import ot
            N = se.shape[0]
            Cs = torch.cdist(se, se, p=2).numpy()
            Ct = torch.cdist(te, te, p=2).numpy()
            # Normalize distance matrices
            Cs /= Cs.max() + 1e-8
            Ct /= Ct.max() + 1e-8
            # Uniform marginals
            p = np.ones(N) / N
            q = np.ones(N) / N
            gw, log = ot.gromov.entropic_gromov_wasserstein(
                Cs, Ct, p, q,
                loss_fun='square_loss',
                epsilon=self.epsilon,
                max_iter=self.n_iter,
                log=True,
            )
            return float(log.get('gw_dist', np.sum(gw * (Cs @ gw - gw @ Ct)**2)))
        except Exception:
            return self._compute_rsa_proxy(se, te)

    def _compute_rsa_proxy(self, se: torch.Tensor, te: torch.Tensor) -> float:
        """RSA proxy: correlation between distance matrices (fast fallback)."""
        Cs = torch.cdist(se, se, p=2).flatten()
        Ct = torch.cdist(te, te, p=2).flatten()
        # Normalise
        Cs = (Cs - Cs.mean()) / (Cs.std() + 1e-8)
        Ct = (Ct - Ct.mean()) / (Ct.std() + 1e-8)
        # 1 - correlation as distance proxy
        corr = (Cs * Ct).mean()
        return float(1.0 - corr.item())

    def report(self) -> str:
        if not self.history:
            return "DGW: no data yet"
        last = self.history[-1]
        trend = ""
        if len(self.history) >= 3:
            recent = [h['dgw'] for h in self.history[-3:]]
            if recent[-1] > recent[0] * 1.5:
                trend = " ↑ phase transition?"
            elif recent[-1] < recent[0] * 0.7:
                trend = " ↓ converging"
        return (f"DGW[{last['step']:,}]={last['dgw']:.4f} "
                f"backend={last['backend']}{trend}")

    def phase_transition_detected(self, threshold: float = 1.5) -> bool:
        """True if DGW has grown by threshold× since start."""
        if len(self.history) < 3:
            return False
        first = self.history[0]['dgw']
        last  = self.history[-1]['dgw']
        return last > first * threshold

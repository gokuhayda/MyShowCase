# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.

"""
PCGrad — Projecting Conflicting Gradients for Multi-Task Learning
==================================================================

Implementation of Yu et al. (2020). "Gradient Surgery for Multi-Task
Learning." NeurIPS 2020.

Problem:
    In HyDRA/CGT distillation, the composite loss has 5–8 terms:
        L = λ_CE * L_CE + λ_KL * L_KL + λ_hidden * L_hidden
          + λ_radius * L_radius + λ_contrast * L_contrast + ...

    These gradients can CONFLICT: the gradient from L_radius may point
    in the opposite direction of the gradient from L_hidden, creating
    destructive interference that slows convergence or drives DegEq.

    The existing GradNormMeasure (distillation_v2.py) DETECTS gradient
    imbalance but does not RESOLVE conflicts. PCGrad resolves them.

Algorithm:
    For each pair of task gradients (g_i, g_j):
        If cos(g_i, g_j) < 0 (conflicting):
            g_i ← g_i - (g_i · g_j / ||g_j||²) * g_j
            (project g_i onto the normal plane of g_j)

    This removes the conflicting component while preserving the
    cooperative component. The result is a gradient that improves
    all tasks simultaneously, or at worst is neutral.

Integration:
    PCGrad operates BETWEEN loss.backward() and optimizer.step().
    It modifies .grad in-place. Compatible with RiemannianAdamW.

    # Option A: wrap individual loss backward calls
    pcgrad = PCGrad(loss_names=['kl', 'hidden', 'radius', 'contrast'])
    pcgrad.backward_and_project(loss_dict, model.parameters())
    optimizer.step()

    # Option B: apply to accumulated gradients (lighter)
    total_loss.backward()
    pcgrad.project_gradients(task_grads, shared_params)
    optimizer.step()

Design: ADDITIVE ONLY. Does not modify optimizer, model, or loss modules.

References:
    Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., & Finn, C.
    (2020). Gradient Surgery for Multi-Task Learning. NeurIPS 2020.
    arXiv:2001.06782
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn


class PCGrad:
    """
    Projecting Conflicting Gradients for multi-task learning.

    Resolves gradient conflicts between task losses by projecting
    conflicting gradients onto each other's normal planes.

    Args:
        reduction: How to combine projected gradients.
            'sum': sum all projected task gradients (default).
            'mean': average projected task gradients.
        conflict_threshold: Cosine threshold below which gradients
            are considered conflicting (default 0.0 = any negative cosine).
        shuffle: Whether to randomize task order each call (recommended).
    """

    def __init__(
        self,
        reduction: str = 'sum',
        conflict_threshold: float = 0.0,
        shuffle: bool = True,
    ):
        self.reduction = reduction
        self.conflict_threshold = conflict_threshold
        self.shuffle = shuffle

        # Diagnostics
        self._n_conflicts: int = 0
        self._n_pairs: int = 0
        self._last_cosines: Dict[str, float] = {}

    def backward_and_project(
        self,
        task_losses: Dict[str, torch.Tensor],
        shared_params: Sequence[nn.Parameter],
        retain_graph: bool = False,
    ) -> Dict[str, float]:
        """
        Compute per-task gradients, project conflicts, and store result.

        This is the full PCGrad pipeline: computes gradients for each task
        independently, resolves conflicts via projection, and writes the
        result into .grad of shared_params.

        Args:
            task_losses: Dict of {task_name: loss_tensor}.
                Each loss must have a valid computation graph.
            shared_params: Parameters to compute gradients for.
                Typically list(model.parameters()).
            retain_graph: Whether to retain the computation graph
                after each backward (needed if losses share graph).

        Returns:
            Dict with conflict statistics.
        """
        params = [p for p in shared_params if p.requires_grad]
        if not params or not task_losses:
            return {"n_conflicts": 0, "n_pairs": 0}

        task_names = list(task_losses.keys())
        n_tasks = len(task_names)

        # ── 1. Compute per-task gradients ──
        task_grads: Dict[str, List[torch.Tensor]] = {}
        for name in task_names:
            loss = task_losses[name]
            if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
                continue
            grads = torch.autograd.grad(
                loss, params,
                retain_graph=True,
                allow_unused=True,
            )
            task_grads[name] = [
                g.clone() if g is not None else torch.zeros_like(p)
                for g, p in zip(grads, params)
            ]

        if len(task_grads) < 2:
            # Only one task with valid gradients — no conflicts possible
            if task_grads:
                name = next(iter(task_grads))
                for p, g in zip(params, task_grads[name]):
                    if p.grad is None:
                        p.grad = g
                    else:
                        p.grad.add_(g)
            return {"n_conflicts": 0, "n_pairs": 0}

        # ── 2. Project conflicting gradients ──
        active_names = list(task_grads.keys())
        if self.shuffle:
            random.shuffle(active_names)

        projected = {name: [g.clone() for g in task_grads[name]]
                     for name in active_names}

        self._n_conflicts = 0
        self._n_pairs = 0
        self._last_cosines = {}

        for i, name_i in enumerate(active_names):
            for j, name_j in enumerate(active_names):
                if i == j:
                    continue
                self._n_pairs += 1

                # Flatten both gradients for cosine computation
                g_i = torch.cat([g.flatten() for g in projected[name_i]])
                g_j = torch.cat([g.flatten() for g in task_grads[name_j]])

                dot = (g_i * g_j).sum()
                norm_j_sq = (g_j * g_j).sum().clamp(min=1e-12)
                cos = dot / (g_i.norm() * g_j.norm()).clamp(min=1e-12)

                pair_key = f"{name_i}-{name_j}"
                self._last_cosines[pair_key] = cos.item()

                if cos.item() < self.conflict_threshold:
                    # Conflict: project g_i onto normal plane of g_j
                    self._n_conflicts += 1
                    coeff = dot / norm_j_sq

                    # Apply projection per-parameter
                    offset = 0
                    for k, g_jk in enumerate(task_grads[name_j]):
                        numel = g_jk.numel()
                        projected[name_i][k] -= coeff * g_jk
                        offset += numel

        # ── 3. Combine projected gradients ──
        for p_idx, p in enumerate(params):
            combined = torch.zeros_like(p)
            for name in active_names:
                combined.add_(projected[name][p_idx])

            if self.reduction == 'mean':
                combined.div_(len(active_names))

            if p.grad is None:
                p.grad = combined
            else:
                p.grad.copy_(combined)

        return {
            "n_conflicts": self._n_conflicts,
            "n_pairs": self._n_pairs,
            "conflict_ratio": self._n_conflicts / max(self._n_pairs, 1),
        }

    def project_flat_gradients(
        self,
        task_grads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Project pre-computed flat gradient vectors.

        Lighter alternative when per-task gradients are already available
        as flat tensors (e.g., from GradNormMeasure).

        Args:
            task_grads: Dict of {task_name: flat_gradient_tensor [D]}.

        Returns:
            Combined projected gradient [D].
        """
        names = list(task_grads.keys())
        if len(names) < 2:
            return sum(task_grads.values())

        if self.shuffle:
            random.shuffle(names)

        projected = {name: task_grads[name].clone() for name in names}

        for i, name_i in enumerate(names):
            for j, name_j in enumerate(names):
                if i == j:
                    continue
                g_i = projected[name_i]
                g_j = task_grads[name_j]
                dot = (g_i * g_j).sum()
                if dot < 0:
                    projected[name_i] = g_i - (dot / g_j.norm().pow(2).clamp(min=1e-12)) * g_j

        result = sum(projected.values())
        if self.reduction == 'mean':
            result = result / len(names)
        return result

    def get_diagnostics(self) -> Dict[str, float]:
        """Return conflict diagnostics for logging."""
        result = {
            "pcgrad_conflicts": self._n_conflicts,
            "pcgrad_pairs": self._n_pairs,
            "pcgrad_conflict_ratio": self._n_conflicts / max(self._n_pairs, 1),
        }
        for pair, cos in self._last_cosines.items():
            result[f"pcgrad_cos_{pair}"] = round(cos, 4)
        return result


class PCGradWrapper:
    """
    Convenience wrapper that integrates PCGrad with the existing
    distillation training loop.

    Replaces the standard:
        total_loss.backward()
        optimizer.step()

    With:
        pcgrad_wrapper.step(task_losses, model, optimizer)

    This handles: per-task backward, projection, gradient accumulation,
    and optimizer step in one call.

    Args:
        pcgrad: PCGrad instance (or creates default).
        grad_clip: Max gradient norm for clipping (default 1.0).
    """

    def __init__(
        self,
        pcgrad: Optional[PCGrad] = None,
        grad_clip: float = 1.0,
    ):
        self.pcgrad = pcgrad or PCGrad()
        self.grad_clip = grad_clip

    def step(
        self,
        task_losses: Dict[str, torch.Tensor],
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """
        Full PCGrad training step.

        Args:
            task_losses: Dict of {task_name: loss_tensor}.
            model: The model being trained.
            optimizer: The optimizer.

        Returns:
            Dict with PCGrad diagnostics.
        """
        optimizer.zero_grad()

        params = list(model.parameters())
        stats = self.pcgrad.backward_and_project(
            task_losses, params, retain_graph=True
        )

        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, self.grad_clip)

        optimizer.step()
        return stats

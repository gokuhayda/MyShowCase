# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
CGT Trainer
===========

Training orchestration for Contrastive Geometric Transfer.

This module implements the training loop with:
- Riemannian optimization for manifold parameters
- Staggered loss annealing
- Early stopping with validation
- Comprehensive logging

Author: Éric Gustavo Reis de Sena
Date: January 2026
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm

from cgt.evaluation import evaluate_stsb
from cgt.models import RiemannianOptimizerWrapper
from cgt.regularization import LipschitzRegularizer


class CGTTrainer:
    """
    Training orchestrator for CGT models.

    Handles the complete training loop including:
    - Hybrid Euclidean/Riemannian optimization
    - Multi-objective loss computation
    - Validation and early stopping
    - Metric logging

    Attributes:
        student: CGT student model.
        criterion: Multi-objective loss function.
        config: Training configuration dictionary.
        optimizer: Riemannian-wrapped optimizer.
        scheduler: Learning rate scheduler.
        history: Training metrics history.

    Notes:
        - Space: Trains model mapping R^D → H^n
        - Status: Standard training loop with manifold support
    """

    def __init__(
        self,
        student: nn.Module,
        criterion: nn.Module,
        config: Dict[str, Any],
        lipschitz_reg: Optional[nn.Module] = None,
    ):
        """
        Initialize trainer.

        Args:
            student: CGT student model.
            criterion: Multi-objective loss.
            config: Training configuration with keys:
                - lr: Learning rate
                - batch_size: Batch size
                - epochs: Number of epochs
                - patience: Early stopping patience
                - weight_decay: Weight decay
                - device: Computation device
            lipschitz_reg: Optional Lipschitz regularizer.
        """
        self.student = student
        self.criterion = criterion
        self.config = config
        self.lipschitz_reg = lipschitz_reg or LipschitzRegularizer()
        self.device = config.get("device", torch.device("cuda"))
        self.dtype = torch.float64

        # Move to device
        self.student = self.student.to(device=self.device, dtype=self.dtype)
        self.criterion = self.criterion.to(device=self.device, dtype=self.dtype)

        # Initialize optimizer
        base_opt = optim.AdamW(
            self.student.parameters(),
            lr=config["lr"],
            weight_decay=config.get("weight_decay", 0.01),
        )
        self.optimizer = RiemannianOptimizerWrapper(base_opt, self.student.substrate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            base_opt, T_max=config["epochs"]
        )

        # History tracking
        self.history: Dict[str, List[float]] = {
            "total": [],
            "contrastive": [],
            "distill": [],
            "topo": [],
            "lipschitz": [],
            "beta_0": [],
            "manifold_err": [],
            "val_rho": [],
        }
        self.best_rho = 0.0
        self.best_state = None
        self.patience_counter = 0

    def train_epoch(
        self,
        epoch: int,
        teacher_embeddings: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number.
            teacher_embeddings: Pre-computed teacher embeddings [N, D].

        Returns:
            Dictionary of epoch statistics.
        """
        self.student.train()
        batch_size = self.config["batch_size"]
        max_epochs = self.config["epochs"]

        # Ensure correct dtype
        teacher_embeddings = teacher_embeddings.to(
            device=self.device, dtype=self.dtype
        )

        # Initialize stats
        epoch_stats = {
            "total": 0.0,
            "contrastive": 0.0,
            "distill": 0.0,
            "topo": 0.0,
            "lipschitz": 0.0,
            "beta_0": 0.0,
            "manifold_err": 0.0,
        }
        n_batches = 0

        # Shuffle indices
        indices = torch.randperm(len(teacher_embeddings), device=self.device)

        for i in range(0, len(teacher_embeddings), batch_size):
            batch_idx = indices[i : i + batch_size]
            batch_t = teacher_embeddings[batch_idx]

            if len(batch_t) < 4:
                continue

            self.optimizer.zero_grad()

            # Forward pass
            hyp_emb = self.student(batch_t, use_homeostatic=True)

            # Compute loss
            loss, loss_dict = self.criterion(
                hyp_emb,
                batch_t,
                self.student.substrate,
                self.student,
                epoch,
                max_epochs,
                self.lipschitz_reg,
            )

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Accumulate statistics
            for key in epoch_stats:
                if key in loss_dict:
                    epoch_stats[key] += loss_dict[key]
            n_batches += 1

        # Average statistics
        for key in epoch_stats:
            epoch_stats[key] /= max(n_batches, 1)

        return epoch_stats

    def validate(
        self,
        val_data: Tuple,
    ) -> float:
        """
        Run validation.

        Args:
            val_data: Tuple of (sent1, sent2, scores, sent_to_idx, teacher_emb).

        Returns:
            Validation Spearman correlation.
        """
        sent1, sent2, scores, sent_to_idx, teacher_emb = val_data[:5]

        rho = evaluate_stsb(
            self.student,
            sent1,
            sent2,
            scores,
            sent_to_idx,
            teacher_emb,
            use_geodesic=True,
            device=self.device,
        )

        return rho

    def fit(
        self,
        teacher_embeddings: torch.Tensor,
        val_data: Tuple,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Full training loop.

        Args:
            teacher_embeddings: Training embeddings [N, D].
            val_data: Validation data tuple.
            verbose: Show progress bar.

        Returns:
            Dictionary with final results and history.
        """
        epochs = self.config["epochs"]
        patience = self.config.get("patience", 20)

        iterator = tqdm(range(epochs), desc="Training") if verbose else range(epochs)

        for epoch in iterator:
            # Train
            train_stats = self.train_epoch(epoch, teacher_embeddings)

            # Validate
            val_rho = self.validate(val_data)
            train_stats["val_rho"] = val_rho

            # Update history
            for key, value in train_stats.items():
                if key in self.history:
                    self.history[key].append(value)

            # Early stopping
            if val_rho > self.best_rho:
                self.best_rho = val_rho
                self.best_state = {
                    k: v.cpu().clone() for k, v in self.student.state_dict().items()
                }
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.patience_counter >= patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                break

            # Update progress bar
            if verbose and hasattr(iterator, "set_postfix"):
                iterator.set_postfix(
                    loss=f"{train_stats['total']:.4f}",
                    val_rho=f"{val_rho:.4f}",
                    best=f"{self.best_rho:.4f}",
                )

            # Learning rate step
            self.scheduler.step()

        # Restore best model
        if self.best_state is not None:
            self.student.load_state_dict(
                {k: v.to(self.device) for k, v in self.best_state.items()}
            )

        return {
            "best_rho": self.best_rho,
            "final_epoch": epoch + 1,
            "history": self.history,
        }

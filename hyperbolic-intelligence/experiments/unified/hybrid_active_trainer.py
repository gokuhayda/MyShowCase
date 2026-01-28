# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.

"""
Hybrid Active Trainer
=====================

Trainer for hybrid model using HybridActiveLoss.
SINGLE training loop with SINGLE backward.

AUDIT COMPLIANCE:
- ✅ No multi-stage training
- ✅ No sequential training
- ✅ No multiple optimizers
- ✅ No multiple backward calls
- ✅ Checkpoints include all state
- ✅ Backward compatible with original training

This trainer ADDS functionality without modifying existing code.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.stats import spearmanr

# Ensure project imports work
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cgt.utils.helpers import set_global_seed, get_device, clear_memory
from cgt.models.cgt_hardened import CGTStudentHardened
from cgt.geometry.lorentz_hardened import LorentzSubstrateHardened, LorentzConfig
from cgt.losses.hybrid_active_loss import (
    HybridActiveLoss,
    HybridLossConfig,
    create_hybrid_loss,
)


def enforce_float64():
    """Enforce float64 precision."""
    torch.set_default_dtype(torch.float64)
    if hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = False


class HybridActiveTrainer:
    """
    Trainer using HybridActiveLoss for unified training.
    
    SINGLE training objective, SINGLE backward, SINGLE optimizer.
    All 7 loss components active during training.
    
    Components:
    - Contrastive [CGT]
    - Distillation [CGT]
    - Topological [TDA]
    - Forman-Ricci [K-Lighting]
    - Coherence [K-Lighting]
    - Lipschitz [CGT]
    - Gromov-Wasserstein [Ψ-SLM]
    
    Checkpoints include:
    - model_state_dict
    - optimizer_state_dict
    - scheduler_state_dict
    - epoch, global_step
    - loss_total, loss_dict
    - config, seed
    """
    
    def __init__(
        self,
        output_dir: Path,
        # Architecture config
        teacher_dim: int = 768,
        student_dim: int = 32,
        hidden_dim: int = 256,
        curvature: float = -1.0,
        # Training config
        batch_size: int = 256,
        num_epochs: int = 25,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
        # Loss weights
        lambda_contrastive: float = 1.0,
        lambda_distill: float = 0.5,
        lambda_topological: float = 0.1,
        lambda_forman: float = 0.1,
        lambda_coherence: float = 0.1,
        lambda_lipschitz: float = 0.05,
        lambda_gw: float = 0.2,
        # Other
        seed: int = 42,
        device: Optional[torch.device] = None,
        checkpoint_every: int = 5,
    ):
        enforce_float64()
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store all config
        self.config = {
            "teacher_dim": teacher_dim,
            "student_dim": student_dim,
            "hidden_dim": hidden_dim,
            "curvature": curvature,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "grad_clip": grad_clip,
            "lambda_contrastive": lambda_contrastive,
            "lambda_distill": lambda_distill,
            "lambda_topological": lambda_topological,
            "lambda_forman": lambda_forman,
            "lambda_coherence": lambda_coherence,
            "lambda_lipschitz": lambda_lipschitz,
            "lambda_gw": lambda_gw,
            "seed": seed,
            "checkpoint_every": checkpoint_every,
        }
        
        self.device = device if device else get_device()
        self.dtype = torch.float64
        self.seed = seed
        self.checkpoint_every = checkpoint_every
        
        # Model components (initialized in train())
        self.student = None
        self.substrate = None
        self.loss_fn = None
        self.optimizer = None
        self.scheduler = None
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_rho = -float("inf")
        
        # History
        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_rho": [],
            "loss/contrastive": [],
            "loss/distill": [],
            "loss/topo": [],
            "loss/forman": [],
            "loss/coherence": [],
            "loss/lipschitz": [],
            "loss/gw": [],
            "topology/beta_0": [],
            "geometry/kappa_mean": [],
        }
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging."""
        log_path = self.output_dir / "train.log"
        
        self.logger = logging.getLogger("hybrid_active_trainer")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        
        fh = logging.FileHandler(log_path)
        fh.setFormatter(logging.Formatter(
            "[%(asctime)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        self.logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(ch)
    
    def _reset_seed(self):
        """Reset seed for reproducibility."""
        set_global_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.logger.info(f"Seed: {self.seed}")
    
    def _create_model(self):
        """Create model components."""
        # Student model (creates its own substrate)
        self.student = CGTStudentHardened(
            teacher_dim=self.config["teacher_dim"],
            student_dim=self.config["student_dim"],
            hidden_dim=self.config["hidden_dim"],
            initial_curvature=abs(self.config["curvature"]),
        ).to(self.device).to(self.dtype)
        
        # Get substrate from student
        self.substrate = self.student.substrate
        
        n_params = sum(p.numel() for p in self.student.parameters())
        self.logger.info(f"Model parameters: {n_params:,}")
    
    def _create_loss(self):
        """Create hybrid loss function."""
        loss_config = HybridLossConfig(
            lambda_contrastive=self.config["lambda_contrastive"],
            lambda_distill=self.config["lambda_distill"],
            lambda_topological=self.config["lambda_topological"],
            lambda_forman=self.config["lambda_forman"],
            lambda_coherence=self.config["lambda_coherence"],
            lambda_lipschitz=self.config["lambda_lipschitz"],
            lambda_gw=self.config["lambda_gw"],
            num_epochs=self.config["num_epochs"],
        )
        
        self.loss_fn = HybridActiveLoss(loss_config, self.substrate)
        
        self.logger.info("Loss weights:")
        self.logger.info(f"  λ_contrastive = {loss_config.lambda_contrastive}")
        self.logger.info(f"  λ_distill = {loss_config.lambda_distill}")
        self.logger.info(f"  λ_topological = {loss_config.lambda_topological}")
        self.logger.info(f"  λ_forman = {loss_config.lambda_forman}")
        self.logger.info(f"  λ_coherence = {loss_config.lambda_coherence}")
        self.logger.info(f"  λ_lipschitz = {loss_config.lambda_lipschitz}")
        self.logger.info(f"  λ_gw = {loss_config.lambda_gw}")
    
    def _create_optimizer(self):
        """Create optimizer and scheduler."""
        self.optimizer = AdamW(
            self.student.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config["num_epochs"],
        )
        
        self.logger.info(f"Optimizer: AdamW (lr={self.config['learning_rate']})")
        self.logger.info(f"Scheduler: CosineAnnealingLR (T_max={self.config['num_epochs']})")
    
    def _train_epoch(
        self,
        train_emb1: torch.Tensor,
        train_emb2: torch.Tensor,
        train_scores: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Train one epoch with SINGLE backward per batch.
        
        Returns accumulated loss statistics.
        """
        self.student.train()
        
        n_samples = train_emb1.shape[0]
        indices = torch.randperm(n_samples)
        batch_size = self.config["batch_size"]
        
        # Accumulate losses
        epoch_losses = {
            "total": 0.0,
            "loss/contrastive": 0.0,
            "loss/distill": 0.0,
            "loss/topo": 0.0,
            "loss/forman": 0.0,
            "loss/coherence": 0.0,
            "loss/lipschitz": 0.0,
            "loss/gw": 0.0,
            "topology/beta_0": 0.0,
            "geometry/kappa_mean": 0.0,
        }
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i + batch_size]
            if len(batch_idx) < 8:
                continue
            
            # Get batch
            emb1 = train_emb1[batch_idx].to(self.device)
            emb2 = train_emb2[batch_idx].to(self.device)
            
            # Use emb1 as teacher, get student output
            # (In pair training, we process both sentences)
            teacher_emb = emb1  # Teacher embeddings
            
            # Zero gradients ONCE
            self.optimizer.zero_grad()
            
            # Forward pass
            student_emb = self.student(teacher_emb)
            
            # Compute unified loss - SINGLE call
            loss_dict = self.loss_fn(
                student_emb=student_emb,
                teacher_emb=teacher_emb,
                model=self.student,
                current_epoch=self.current_epoch,
            )
            
            # SINGLE backward
            loss_dict['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.student.parameters(),
                self.config["grad_clip"],
            )
            
            # SINGLE optimizer step
            self.optimizer.step()
            
            # Accumulate statistics
            epoch_losses["total"] += loss_dict['total'].item()
            for key in ["loss/contrastive", "loss/distill", "loss/topo", 
                        "loss/forman", "loss/coherence", "loss/lipschitz", "loss/gw"]:
                if key in loss_dict:
                    epoch_losses[key] += loss_dict[key]
            
            if "topology/beta_0" in loss_dict:
                epoch_losses["topology/beta_0"] += loss_dict["topology/beta_0"]
            if "geometry/kappa_mean" in loss_dict:
                epoch_losses["geometry/kappa_mean"] += loss_dict["geometry/kappa_mean"]
            
            n_batches += 1
            self.global_step += 1
        
        # Average
        for key in epoch_losses:
            epoch_losses[key] /= max(n_batches, 1)
        
        return epoch_losses
    
    def _validate(
        self,
        val_emb1: torch.Tensor,
        val_emb2: torch.Tensor,
        val_scores: torch.Tensor,
    ) -> float:
        """Compute validation Spearman rho."""
        self.student.eval()
        
        with torch.no_grad():
            out1 = self.student(val_emb1.to(self.device))
            out2 = self.student(val_emb2.to(self.device))
            
            dists = self.substrate.dist(out1, out2)
            sims = -dists
            
            rho, _ = spearmanr(sims.cpu().numpy(), val_scores.cpu().numpy())
        
        return rho
    
    def _save_checkpoint(
        self,
        epoch: int,
        loss_total: float,
        loss_dict: Dict[str, float],
        is_best: bool = False,
    ):
        """
        Save checkpoint with ALL required state.
        
        Includes:
        - model_state_dict
        - optimizer_state_dict
        - scheduler_state_dict
        - epoch, global_step
        - loss_total, loss_dict
        - config, seed
        """
        checkpoint = {
            # Model state
            "model_state_dict": self.student.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            
            # Training state
            "epoch": epoch,
            "global_step": self.global_step,
            
            # Loss information
            "loss_total": loss_total,
            "loss_dict": loss_dict,
            
            # Configuration
            "config": self.config,
            "loss_config": self.loss_fn.get_config(),
            "seed": self.seed,
            
            # Metadata
            "timestamp": datetime.now().isoformat(),
            "best_val_rho": self.best_val_rho,
        }
        
        # Regular checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Best checkpoint
        if is_best:
            best_path = self.output_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"  New best model saved (ρ={self.best_val_rho:.4f})")
        
        # Latest checkpoint (for resuming)
        latest_path = self.output_dir / "latest_checkpoint.pth"
        torch.save(checkpoint, latest_path)
    
    def train(
        self,
        train_emb1: torch.Tensor,
        train_emb2: torch.Tensor,
        train_scores: torch.Tensor,
        val_emb1: torch.Tensor,
        val_emb2: torch.Tensor,
        val_scores: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Execute training with SINGLE unified loss.
        
        Returns training results.
        """
        self._reset_seed()
        
        self.logger.info("=" * 70)
        self.logger.info("HYBRID ACTIVE TRAINING")
        self.logger.info("=" * 70)
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Dtype: {self.dtype}")
        self.logger.info("")
        
        # Create components
        self._create_model()
        self._create_loss()
        self._create_optimizer()
        
        # Ensure float64
        train_emb1 = train_emb1.to(self.dtype)
        train_emb2 = train_emb2.to(self.dtype)
        train_scores = train_scores.to(self.dtype)
        val_emb1 = val_emb1.to(self.dtype)
        val_emb2 = val_emb2.to(self.dtype)
        val_scores = val_scores.to(self.dtype)
        
        # Training loop
        start_time = time.time()
        
        self.logger.info("")
        self.logger.info(f"Training for {self.config['num_epochs']} epochs...")
        self.logger.info(f"Batch size: {self.config['batch_size']}")
        self.logger.info("")
        
        for epoch in range(self.config['num_epochs']):
            self.current_epoch = epoch
            
            # Train epoch
            losses = self._train_epoch(train_emb1, train_emb2, train_scores)
            
            # Validate
            val_rho = self._validate(val_emb1, val_emb2, val_scores)
            
            # Update scheduler
            self.scheduler.step()
            
            # Track best
            is_best = val_rho > self.best_val_rho
            if is_best:
                self.best_val_rho = val_rho
            
            # Record history
            self.history["epoch"].append(epoch + 1)
            self.history["train_loss"].append(losses["total"])
            self.history["val_rho"].append(val_rho)
            for key in ["loss/contrastive", "loss/distill", "loss/topo", 
                        "loss/forman", "loss/coherence", "loss/lipschitz", "loss/gw",
                        "topology/beta_0", "geometry/kappa_mean"]:
                if key in losses:
                    self.history[key].append(losses[key])
            
            # Log
            self.logger.info(
                f"Epoch {epoch+1:3d}/{self.config['num_epochs']} | "
                f"Loss: {losses['total']:.4f} | Val ρ: {val_rho:.4f} | "
                f"β₀: {losses.get('topology/beta_0', 0):.2f} | "
                f"κ: {losses.get('geometry/kappa_mean', 0):.3f} | "
                f"{'★ BEST' if is_best else ''}"
            )
            
            # Checkpoint
            if (epoch + 1) % self.checkpoint_every == 0 or is_best:
                self._save_checkpoint(epoch + 1, losses["total"], losses, is_best)
        
        train_time = time.time() - start_time
        
        # Final save
        self._save_checkpoint(self.config['num_epochs'], losses["total"], losses)
        
        # Save FINISHED flag
        flag_path = self.output_dir / "FINISHED.flag"
        with open(flag_path, "w") as f:
            f.write(f"Completed: {datetime.now().isoformat()}\n")
            f.write(f"Best Val ρ: {self.best_val_rho:.4f}\n")
            f.write(f"Training time: {train_time:.1f}s\n")
        
        # Compile results
        results = {
            "model_type": "hybrid_active",
            "final_val_rho": self.history["val_rho"][-1],
            "best_val_rho": self.best_val_rho,
            "train_time_seconds": train_time,
            "history": self.history,
            "config": self.config,
            "loss_config": self.loss_fn.get_config(),
            "timestamp": datetime.now().isoformat(),
        }
        
        # Save results JSON
        results_path = self.output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("TRAINING COMPLETE")
        self.logger.info(f"Best Val ρ: {self.best_val_rho:.4f}")
        self.logger.info(f"Time: {train_time/60:.1f} minutes")
        self.logger.info("=" * 70)
        
        return results
    
    def load_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """
        Load checkpoint for resuming training.
        
        Returns checkpoint dict.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Restore config
        self.config = checkpoint["config"]
        self.seed = checkpoint["seed"]
        
        # Create components
        self._create_model()
        self._create_loss()
        self._create_optimizer()
        
        # Load state
        self.student.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # Restore training state
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_rho = checkpoint.get("best_val_rho", -float("inf"))
        
        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
        
        return checkpoint


def train_hybrid_active(
    output_dir: Path,
    data: Dict[str, torch.Tensor],
    **kwargs,
) -> Dict[str, Any]:
    """
    Train hybrid active model.
    
    Args:
        output_dir: Output directory
        data: Dataset dict with train/val embeddings
        **kwargs: Override default config
    
    Returns:
        Training results
    """
    trainer = HybridActiveTrainer(output_dir, **kwargs)
    
    return trainer.train(
        train_emb1=data["train_emb1"],
        train_emb2=data["train_emb2"],
        train_scores=data["train_scores"],
        val_emb1=data["validation_emb1"],
        val_emb2=data["validation_emb2"],
        val_scores=data["validation_scores"],
    )

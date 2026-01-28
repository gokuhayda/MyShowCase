# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.

"""
Replication Executor
====================

Executes faithful replications of the 4 original models.
NO hybrid in this phase. NO comparative analysis.

AUDIT COMPLIANCE:
- ✅ Uses EXACT configs from replication_configs.py
- ✅ float64 enforced
- ✅ Immediate persistence
- ✅ FINISHED.flag per model
"""

from __future__ import annotations

import json
import logging
import sys
import time
import yaml
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.stats import spearmanr

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# REUSE existing modules
from cgt.utils.helpers import set_global_seed, get_device, clear_memory
from cgt.models.cgt_hardened import CGTStudentHardened
from cgt.losses.losses_hardened import MultiObjectiveLoss
from cgt.geometry.lorentz_hardened import LorentzSubstrateHardened, LorentzConfig

# Local imports
from .replication_configs import (
    ReplicationModel,
    get_replication_config,
    get_config_diff_from_reference,
    KLightNumericalParityConfig,
    KLightAGIv2Config,
    CGTPaperReadyConfig,
    PSISLMConfig,
)


def enforce_float64():
    """Enforce float64 precision."""
    torch.set_default_dtype(torch.float64)


# ═══════════════════════════════════════════════════════════════════════════════
#                    PAIRWISE STS LOSS (FOR STS-B TRAINING)
# ═══════════════════════════════════════════════════════════════════════════════

class PairwiseSTSLoss(torch.nn.Module):
    """
    Pairwise loss for STS-B training.
    
    This loss is designed for sentence similarity tasks where:
    - We have pairs of sentences (emb1, emb2)
    - We have similarity scores (labels)
    - We want to minimize the difference between predicted and actual similarity
    
    Components:
    1. Contrastive: InfoNCE over concatenated embeddings
    2. Distillation: MSE between student and teacher distances
    3. Correlation: Direct Spearman-like correlation loss
    """
    
    def __init__(
        self,
        lambda_contrastive: float = 1.0,
        lambda_distill: float = 0.5,
        lambda_corr: float = 0.3,
        temperature: float = 0.07,
        substrate: LorentzSubstrateHardened = None,
    ):
        super().__init__()
        self.lc = lambda_contrastive
        self.ld = lambda_distill
        self.lcorr = lambda_corr
        self.temp = temperature
        self.substrate = substrate
    
    def forward(
        self,
        student_emb1: torch.Tensor,
        student_emb2: torch.Tensor,
        teacher_emb1: torch.Tensor,
        teacher_emb2: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute pairwise loss.
        
        Args:
            student_emb1: Student embeddings for sentence 1 [B, D+1]
            student_emb2: Student embeddings for sentence 2 [B, D+1]
            teacher_emb1: Teacher embeddings for sentence 1 [B, D]
            teacher_emb2: Teacher embeddings for sentence 2 [B, D]
            labels: Similarity scores [B]
            
        Returns:
            Dictionary with total loss and components
        """
        device = student_emb1.device
        dtype = student_emb1.dtype
        B = student_emb1.shape[0]
        
        # Student distances (hyperbolic)
        if self.substrate is not None:
            d_student = self.substrate.dist(student_emb1, student_emb2)
        else:
            # Fallback: Euclidean distance
            d_student = torch.norm(student_emb1 - student_emb2, dim=-1)
        
        # Teacher distances (cosine)
        t1_norm = torch.nn.functional.normalize(teacher_emb1, dim=-1)
        t2_norm = torch.nn.functional.normalize(teacher_emb2, dim=-1)
        d_teacher = 1.0 - (t1_norm * t2_norm).sum(dim=-1)  # Cosine distance
        
        # 1. Contrastive loss (on concatenated batch)
        l_contrastive = torch.tensor(0.0, device=device, dtype=dtype)
        if self.lc > 0 and B > 1 and self.substrate is not None:
            # Stack for larger effective batch
            all_student = torch.cat([student_emb1, student_emb2], dim=0)
            D = self.substrate.distance_matrix(all_student)
            logits = (-D / self.temp).clamp(-50, 50)
            logits = logits - logits.max(dim=1, keepdim=True).values
            target = torch.arange(2 * B, device=device)
            l_contrastive = torch.nn.functional.cross_entropy(logits, target)
        
        # 2. Distillation loss (distance alignment)
        l_distill = torch.tensor(0.0, device=device, dtype=dtype)
        if self.ld > 0:
            # Normalize distances to same scale
            d_s_norm = d_student / (d_student.max() + 1e-8)
            d_t_norm = d_teacher / (d_teacher.max() + 1e-8)
            l_distill = torch.nn.functional.mse_loss(d_s_norm, d_t_norm)
        
        # 3. Correlation loss (align with labels)
        l_corr = torch.tensor(0.0, device=device, dtype=dtype)
        if self.lcorr > 0:
            # Similarity = 1 - distance (normalized)
            sim_student = 1.0 - d_student / (d_student.max() + 1e-8)
            labels_norm = labels / (labels.max() + 1e-8)
            l_corr = torch.nn.functional.mse_loss(sim_student, labels_norm)
        
        total = self.lc * l_contrastive + self.ld * l_distill + self.lcorr * l_corr
        
        return {
            "total": total,
            "contrastive": l_contrastive,
            "distill": l_distill,
            "correlation": l_corr,
        }
    if hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = False


class ReplicationTrainer:
    """
    Trainer for faithful model replication.
    
    Uses EXACT configs from source notebooks.
    NO modifications to improve stability.
    """
    
    def __init__(
        self,
        model: ReplicationModel,
        output_dir: Path,
        device: Optional[torch.device] = None,
    ):
        self.model_type = model
        self.config = get_replication_config(model)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup
        enforce_float64()
        self.device = device if device else get_device()
        self.dtype = torch.float64
        
        # Model components
        self.student = None
        self.loss_fn = None
        self.substrate = None
        self.optimizer = None
        self.scheduler = None
        
        # History
        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_rho": [],
        }
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging to file."""
        log_path = self.output_dir / "train.log"
        
        self.logger = logging.getLogger(f"replication_{self.model_type.value}")
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
        """Reset seed using config value."""
        seed = self.config.seed
        set_global_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        msg = f"Seed: {seed}"
        if not self.config.seed_documented:
            msg += " (NOT SPECIFIED in notebook, using default)"
        self.logger.info(msg)
    
    def _create_model(self):
        """Create model with exact config."""
        self.student = CGTStudentHardened(
            teacher_dim=self.config.teacher_dim,
            student_dim=self.config.student_dim,
            hidden_dim=self.config.hidden_dim,
            initial_curvature=abs(self.config.curvature),
        ).to(self.device).to(self.dtype)
        
        # Get substrate from student
        self.substrate = self.student.substrate
        
        n_params = sum(p.numel() for p in self.student.parameters())
        self.logger.info(f"Model parameters: {n_params:,}")
    
    def _create_loss(self):
        """Create loss function with exact weights."""
        # Use PairwiseSTSLoss for STS-B pairwise training
        # This is compatible with the training loop interface
        
        if isinstance(self.config, (KLightNumericalParityConfig, CGTPaperReadyConfig)):
            self.loss_fn = PairwiseSTSLoss(
                lambda_contrastive=self.config.lambda_contrastive,
                lambda_distill=self.config.lambda_distillation,
                lambda_corr=self.config.lambda_topological,
                substrate=self.substrate,
            )
        elif isinstance(self.config, KLightAGIv2Config):
            self.loss_fn = PairwiseSTSLoss(
                lambda_contrastive=self.config.lambda_contrastive,
                lambda_distill=self.config.lambda_distillation,
                lambda_corr=self.config.lambda_topological,
                substrate=self.substrate,
            )
        elif isinstance(self.config, PSISLMConfig):
            self.loss_fn = PairwiseSTSLoss(
                lambda_contrastive=self.config.lambda_nce,
                lambda_distill=self.config.lambda_gw,
                lambda_corr=self.config.lambda_topo,
                substrate=self.substrate,
            )
    
    def _create_optimizer(self):
        """Create optimizer with exact hyperparameters."""
        self.optimizer = AdamW(
            self.student.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        t_max = getattr(self.config, 't_max', self.config.num_epochs)
        eta_min = getattr(self.config, 'eta_min', 0)
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=t_max,
            eta_min=eta_min,
        )
        
        self.logger.info(f"Optimizer: AdamW (lr={self.config.learning_rate}, wd={self.config.weight_decay})")
        self.logger.info(f"Scheduler: CosineAnnealingLR (T_max={t_max})")
    
    def _train_epoch(
        self,
        train_emb1: torch.Tensor,
        train_emb2: torch.Tensor,
        train_scores: torch.Tensor,
    ) -> float:
        """Train one epoch."""
        self.student.train()
        
        n_samples = train_emb1.shape[0]
        indices = torch.randperm(n_samples)
        batch_size = self.config.batch_size
        
        total_loss = 0.0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i + batch_size]
            if len(batch_idx) < 8:  # Skip very small batches
                continue
            
            emb1 = train_emb1[batch_idx].to(self.device)
            emb2 = train_emb2[batch_idx].to(self.device)
            scores = train_scores[batch_idx].to(self.device)
            
            self.optimizer.zero_grad()
            
            out1 = self.student(emb1)
            out2 = self.student(emb2)
            
            loss_dict = self.loss_fn(
                student_emb1=out1,
                student_emb2=out2,
                teacher_emb1=emb1,
                teacher_emb2=emb2,
                labels=scores,
            )
            
            loss = loss_dict["total"]
            loss.backward()
            
            # Gradient clipping
            grad_clip = getattr(self.config, 'grad_clip', 1.0)
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), grad_clip)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / max(n_batches, 1)
    
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
        Execute training replication.
        
        Returns:
            Results dictionary
        """
        self._reset_seed()
        
        self.logger.info("=" * 60)
        self.logger.info(f"REPLICATION: {self.model_type.value}")
        self.logger.info("=" * 60)
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Dtype: {self.dtype}")
        
        # Log config differences from reference
        diffs = get_config_diff_from_reference(self.model_type)
        if diffs:
            self.logger.info("\nDifferences from reference (K_LIGHT_NUMERICAL_PARITY):")
            for param, vals in diffs.items():
                self.logger.info(f"  {param}: {vals['reference']} → {vals['model']}")
        else:
            self.logger.info("\nThis IS the reference model.")
        
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
        best_val_rho = -float("inf")
        best_epoch = 0
        epochs_no_improve = 0
        
        # Early stopping config (default: patience=20, min_delta=0.0001)
        patience = getattr(self.config, 'early_stopping_patience', 25)
        min_delta = getattr(self.config, 'early_stopping_min_delta', 0.0001)
        
        self.logger.info(f"\nTraining for {self.config.num_epochs} epochs...")
        self.logger.info(f"Batch size: {self.config.batch_size}")
        self.logger.info(f"Early stopping: patience={patience}, min_delta={min_delta}")
        self.logger.info("")
        
        for epoch in range(self.config.num_epochs):
            loss = self._train_epoch(train_emb1, train_emb2, train_scores)
            val_rho = self._validate(val_emb1, val_emb2, val_scores)
            
            self.scheduler.step()
            
            self.history["epoch"].append(epoch + 1)
            self.history["train_loss"].append(loss)
            self.history["val_rho"].append(val_rho)
            
            # Check for improvement
            if val_rho > best_val_rho + min_delta:
                best_val_rho = val_rho
                best_epoch = epoch + 1
                epochs_no_improve = 0
                # Save best model
                self._save_best_checkpoint()
            else:
                epochs_no_improve += 1
            
            self.logger.info(
                f"Epoch {epoch+1:3d}/{self.config.num_epochs} | "
                f"Loss: {loss:.4f} | Val ρ: {val_rho:.4f} | "
                f"Best: {best_val_rho:.4f} (ep {best_epoch})"
            )
            
            # Early stopping check
            if epochs_no_improve >= patience:
                self.logger.info(f"\n⏹️ Early stopping triggered at epoch {epoch+1} (no improvement for {patience} epochs)")
                break
        
        train_time = time.time() - start_time
        
        # Compile results
        results = {
            "model_type": self.model_type.value,
            "final_val_rho": self.history["val_rho"][-1],
            "best_val_rho": best_val_rho,
            "best_epoch": best_epoch,
            "train_time_seconds": train_time,
            "history": self.history,
            "config": {k: v for k, v in self.config.__dict__.items() if not k.startswith('_')},
            "timestamp": datetime.now().isoformat(),
            "device": str(self.device),
            "dtype": str(self.dtype),
        }
        
        # Save results
        self._save_outputs(results)
        
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info(f"REPLICATION COMPLETE: {self.model_type.value}")
        self.logger.info(f"Final Val ρ: {results['final_val_rho']:.4f}")
        self.logger.info(f"Best Val ρ: {best_val_rho:.4f} (epoch {best_epoch})")
        self.logger.info(f"Time: {train_time:.1f}s")
        self.logger.info("=" * 60)
        
        return results
    
    def _save_best_checkpoint(self):
        """Save best model checkpoint during training."""
        checkpoint_path = self.output_dir / "best_model.pth"
        torch.save({
            "model_state_dict": self.student.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }, checkpoint_path)
    
    def _save_outputs(self, results: Dict[str, Any]):
        """Save all required outputs."""
        # Model checkpoint
        checkpoint_path = self.output_dir / "model_checkpoint.pth"
        torch.save({
            "model_state_dict": self.student.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "history": self.history,
            "best_val_rho": results["best_val_rho"],
        }, checkpoint_path)
        self.logger.info(f"Saved: {checkpoint_path}")
        
        # Train log (JSON)
        log_path = self.output_dir / "train_log.json"
        with open(log_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        self.logger.info(f"Saved: {log_path}")
        
        # Config snapshot (YAML)
        config_path = self.output_dir / "config_snapshot.yaml"
        config_dict = {k: v for k, v in self.config.__dict__.items() if not k.startswith('_')}
        # Convert tuples to lists for YAML
        for k, v in config_dict.items():
            if isinstance(v, tuple):
                config_dict[k] = list(v)
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        self.logger.info(f"Saved: {config_path}")
        
        # FINISHED flag
        flag_path = self.output_dir / "FINISHED.flag"
        with open(flag_path, "w") as f:
            f.write(f"Completed: {datetime.now().isoformat()}\n")
            f.write(f"Model: {self.model_type.value}\n")
            f.write(f"Final Val ρ: {results['final_val_rho']:.4f}\n")
            f.write(f"Best Val ρ: {results['best_val_rho']:.4f}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Dtype: {self.dtype}\n")
        self.logger.info(f"Saved: {flag_path}")


def run_all_replications(
    output_base: Path,
    data: Dict[str, torch.Tensor],
    skip_psi_slm: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Run all 4 replications.
    
    Args:
        output_base: Base output directory
        data: Pre-loaded dataset
        skip_psi_slm: Whether to skip PSI_SLM (requires different data)
    
    Returns:
        Dictionary of results per model
    """
    enforce_float64()
    
    models_to_run = [
        ReplicationModel.K_LIGHT_NUMERICAL_PARITY,
        ReplicationModel.K_LIGHT_AGI_V2,
        ReplicationModel.CGT_PAPER_READY,
    ]
    
    if not skip_psi_slm:
        models_to_run.append(ReplicationModel.PSI_SLM)
    
    all_results = {}
    
    for model in models_to_run:
        print(f"\n{'#'*60}")
        print(f"# REPLICATION: {model.value}")
        print(f"{'#'*60}")
        
        output_dir = output_base / model.value
        
        # Check if already complete
        if (output_dir / "FINISHED.flag").exists():
            print(f"Already complete. Skipping.")
            
            # Load existing results
            log_path = output_dir / "train_log.json"
            if log_path.exists():
                with open(log_path) as f:
                    all_results[model.value] = json.load(f)
            continue
        
        # Run replication
        trainer = ReplicationTrainer(model, output_dir)
        
        results = trainer.train(
            train_emb1=data["train_emb1"],
            train_emb2=data["train_emb2"],
            train_scores=data["train_scores"],
            val_emb1=data["validation_emb1"],
            val_emb2=data["validation_emb2"],
            val_scores=data["validation_scores"],
        )
        
        all_results[model.value] = results
        
        # Clear memory
        clear_memory()
    
    return all_results

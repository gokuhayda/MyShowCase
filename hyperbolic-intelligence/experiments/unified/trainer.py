# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
Unified Trainer
===============

Training interface that wraps existing hardened modules.
NO formula derivation. NO reimplementation. ENCAPSULATION ONLY.

AUDIT COMPLIANCE:
- ✅ Uses CGTStudentHardened from src/cgt/models/cgt_hardened
- ✅ Uses MultiObjectiveLoss from src/cgt/losses/losses_hardened
- ✅ Uses set_global_seed from src/cgt/utils/helpers
- ✅ float64 enforced end-to-end
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.stats import spearmanr

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# REUSE existing modules (NO DUPLICATION)
from cgt.utils.helpers import set_global_seed, get_device, clear_memory
from cgt.models.cgt_hardened import CGTStudentHardened
from cgt.losses.losses_hardened import MultiObjectiveLoss
from cgt.geometry.lorentz_hardened import LorentzSubstrateHardened, LorentzConfig

# Local config
from .config import (
    ModelType,
    ModelConfig,
    TrainingHyperparameters,
    LossWeights,
    ModelArchitecture,
    get_model_configs,
)


class UnifiedTrainer:
    """
    Unified trainer for all 5 model types.
    
    ENCAPSULATION ONLY - wraps existing hardened modules.
    
    Key guarantees:
    - float64 precision throughout
    - Deterministic seed reset before each model
    - Isolated output directories per model
    """
    
    def __init__(
        self,
        model_type: ModelType,
        output_dir: Path,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ):
        if dtype != torch.float64:
            raise ValueError(f"dtype must be torch.float64, got {dtype}")
        
        self.model_type = model_type
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = device if device else get_device()
        self.dtype = dtype
        
        torch.set_default_dtype(self.dtype)
        
        self.config = get_model_configs()[model_type]
        
        self.model: Optional[CGTStudentHardened] = None
        self.loss_fn: Optional[MultiObjectiveLoss] = None
        self.substrate: Optional[LorentzSubstrateHardened] = None
        self.optimizer: Optional[AdamW] = None
        self.scheduler: Optional[CosineAnnealingLR] = None
        
        self.history: Dict[str, list] = {
            "epoch": [], "train_loss": [], "val_rho": [],
            "distill_loss": [], "topo_loss": [], "lipschitz_loss": [],
            "homeostatic_loss": [], "beta_0": [], "lr": [],
        }
        
        self.train_start_time: Optional[float] = None
        self.train_end_time: Optional[float] = None
    
    def _reset_seed(self, seed: int) -> None:
        set_global_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _create_model(self) -> CGTStudentHardened:
        arch = self.config.architecture
        
        model = CGTStudentHardened(
            teacher_dim=arch.teacher_dim,
            student_dim=arch.student_dim,
            hidden_dim=arch.hidden_dim,
            initial_curvature=abs(arch.curvature),
        )
        
        # Get substrate from model
        self.substrate = model.substrate
        
        return model.to(self.device).to(self.dtype)
    
    def _create_loss(self) -> MultiObjectiveLoss:
        weights = self.config.loss_weights
        
        return MultiObjectiveLoss(
            lambda_contrastive=weights.contrastive,
            lambda_distillation=weights.distillation,
            lambda_topological=weights.topological,
            lambda_lipschitz=weights.lipschitz,
            lambda_homeostatic=weights.homeostatic,
            substrate=self.substrate,
        )
    
    def _create_optimizer(self) -> Tuple[AdamW, CosineAnnealingLR]:
        hp = self.config.training
        
        optimizer = AdamW(
            self.model.parameters(),
            lr=hp.learning_rate,
            weight_decay=hp.weight_decay,
        )
        
        scheduler = CosineAnnealingLR(optimizer, T_max=hp.t_max)
        
        return optimizer, scheduler
    
    def _train_epoch(
        self,
        train_emb1: torch.Tensor,
        train_emb2: torch.Tensor,
        train_scores: torch.Tensor,
    ) -> Dict[str, float]:
        self.model.train()
        hp = self.config.training
        
        n_samples = train_emb1.shape[0]
        indices = torch.randperm(n_samples)
        
        epoch_losses = {
            "total": 0.0, "distill": 0.0, "topo": 0.0,
            "lipschitz": 0.0, "homeostatic": 0.0, "beta_0": 0.0,
        }
        n_batches = 0
        
        for i in range(0, n_samples, hp.batch_size):
            batch_idx = indices[i:i + hp.batch_size]
            
            emb1 = train_emb1[batch_idx].to(self.device)
            emb2 = train_emb2[batch_idx].to(self.device)
            scores = train_scores[batch_idx].to(self.device)
            
            self.optimizer.zero_grad()
            
            out1 = self.model(emb1)
            out2 = self.model(emb2)
            
            loss_dict = self.loss_fn(
                student_emb1=out1, student_emb2=out2,
                teacher_emb1=emb1, teacher_emb2=emb2,
                labels=scores,
            )
            
            total_loss = loss_dict["total"]
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), hp.grad_clip)
            self.optimizer.step()
            
            for key in epoch_losses:
                if key in loss_dict:
                    epoch_losses[key] += loss_dict[key].item()
            
            n_batches += 1
        
        for key in epoch_losses:
            epoch_losses[key] /= max(n_batches, 1)
        
        return epoch_losses
    
    def _validate(
        self,
        val_emb1: torch.Tensor,
        val_emb2: torch.Tensor,
        val_scores: torch.Tensor,
    ) -> float:
        self.model.eval()
        
        with torch.no_grad():
            out1 = self.model(val_emb1.to(self.device))
            out2 = self.model(val_emb2.to(self.device))
            
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
        hp = self.config.training
        
        self._reset_seed(hp.seed)
        
        self.model = self._create_model()
        self.loss_fn = self._create_loss()
        self.optimizer, self.scheduler = self._create_optimizer()
        
        train_emb1 = train_emb1.to(self.dtype)
        train_emb2 = train_emb2.to(self.dtype)
        train_scores = train_scores.to(self.dtype)
        val_emb1 = val_emb1.to(self.dtype)
        val_emb2 = val_emb2.to(self.dtype)
        val_scores = val_scores.to(self.dtype)
        
        self.train_start_time = time.time()
        best_val_rho = -float("inf")
        best_epoch = 0
        
        print(f"\n{'='*60}")
        print(f"Training: {self.model_type.value}")
        print(f"{'='*60}")
        print(f"Device: {self.device} | Dtype: {self.dtype}")
        print(f"Epochs: {hp.num_epochs} | Batch: {hp.batch_size}")
        print(f"LR: {hp.learning_rate} | WD: {hp.weight_decay}")
        print(f"{'='*60}\n")
        
        for epoch in range(hp.num_epochs):
            losses = self._train_epoch(train_emb1, train_emb2, train_scores)
            val_rho = self._validate(val_emb1, val_emb2, val_scores)
            
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            self.history["epoch"].append(epoch + 1)
            self.history["train_loss"].append(losses["total"])
            self.history["val_rho"].append(val_rho)
            self.history["distill_loss"].append(losses["distill"])
            self.history["topo_loss"].append(losses["topo"])
            self.history["lipschitz_loss"].append(losses["lipschitz"])
            self.history["homeostatic_loss"].append(losses["homeostatic"])
            self.history["beta_0"].append(losses["beta_0"])
            self.history["lr"].append(current_lr)
            
            if val_rho > best_val_rho:
                best_val_rho = val_rho
                best_epoch = epoch + 1
            
            print(f"Epoch {epoch+1:3d}/{hp.num_epochs} | "
                  f"Loss: {losses['total']:.4f} | "
                  f"Val ρ: {val_rho:.4f} | "
                  f"Best: {best_val_rho:.4f} (ep {best_epoch})")
        
        self.train_end_time = time.time()
        train_time = self.train_end_time - self.train_start_time
        
        results = {
            "model_type": self.model_type.value,
            "final_val_rho": self.history["val_rho"][-1],
            "best_val_rho": best_val_rho,
            "best_epoch": best_epoch,
            "train_time_seconds": train_time,
            "history": self.history,
            "config": {
                "training": asdict(hp),
                "loss_weights": asdict(self.config.loss_weights),
                "architecture": asdict(self.config.architecture),
            },
            "timestamp": datetime.now().isoformat(),
        }
        
        results_path = self.output_dir / f"{self.model_type.value}_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n{'='*60}")
        print(f"Complete: {self.model_type.value}")
        print(f"Final ρ: {results['final_val_rho']:.4f} | Best: {best_val_rho:.4f}")
        print(f"Time: {train_time:.1f}s | Saved: {results_path}")
        print(f"{'='*60}\n")
        
        return results
    
    def evaluate(
        self,
        test_emb1: torch.Tensor,
        test_emb2: torch.Tensor,
        test_scores: torch.Tensor,
    ) -> Dict[str, float]:
        test_rho = self._validate(test_emb1, test_emb2, test_scores)
        return {"test_rho": test_rho, "model_type": self.model_type.value}
    
    def get_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model(embeddings.to(self.device).to(self.dtype))
    
    def save_model(self, path: Optional[Path] = None) -> Path:
        if path is None:
            path = self.output_dir / f"{self.model_type.value}_model.pt"
        
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": asdict(self.config),
            "history": self.history,
        }, path)
        
        return path
    
    def load_model(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        if self.model is None:
            self.model = self._create_model()
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.history = checkpoint.get("history", self.history)

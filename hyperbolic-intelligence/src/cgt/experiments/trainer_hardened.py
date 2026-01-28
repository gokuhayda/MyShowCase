# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
CGT Trainer Orchestrator [HARDENED VERSION]
===========================================

EXACT implementation matching CGT_Paper_Ready_v6_1_HARDENED notebook Cell 34.
Integrates Riemannian optimization, staggered annealing, and manifold diagnostics.

V9.9.3: Float64 + Device consistency + Fixed loss extraction

Author: Éric Gustavo Reis de Sena
Date: January 2026
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import spearmanr


# Default device and dtype
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float64
np_dtype = np.float64


class RiemannianOptimizerWrapper:
    """
    Wraps a standard optimizer to add Riemannian gradient corrections.
    
    This is the SIMPLIFIED version from Cell 34 - just wraps the base optimizer.
    For full Riemannian updates, use the version from Cell 18.
    """

    def __init__(self, base_optimizer: optim.Optimizer, substrate):
        self.base_optimizer = base_optimizer
        self.substrate = substrate

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def step(self):
        self.base_optimizer.step()

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups


class CGTTrainer:
    """
    Orchestrator for Contrastive Geometric Transfer training.
    
    EXACT match to CGT_Paper_Ready_v6_1_HARDENED notebook Cell 34.
    Integrates Riemannian optimization, staggered annealing, and manifold diagnostics.

    V9.9.3: Float64 + Device consistency + Fixed loss extraction
    """

    def __init__(
        self, 
        student: nn.Module, 
        criterion: nn.Module, 
        config: Dict,
        evaluate_fn: Optional[Callable] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            student: CGT student model
            criterion: MultiObjectiveLoss instance
            config: Training configuration dict with keys:
                - lr: Learning rate
                - epochs: Number of epochs
                - batch_size: Batch size
                - weight_decay: Weight decay (default 0.01)
                - patience: Early stopping patience (default 10)
                - enable_homeostatic: Whether to use homeostatic field (default True)
                - device: Device to use
            evaluate_fn: Optional evaluation function (default uses built-in)
        """
        self.student = student
        self.criterion = criterion
        self.config = config
        self.device = config.get('device', device)
        self.dtype = torch.float64  # FLOAT64
        self.evaluate_fn = evaluate_fn

        # Move student to device and dtype
        self.student = self.student.to(device=self.device, dtype=self.dtype)

        # Initialize Hybrid Optimizer
        base_opt = optim.AdamW(
            self.student.parameters(),
            lr=config['lr'],
            weight_decay=config.get('weight_decay', 0.01)
        )
        self.optimizer = RiemannianOptimizerWrapper(base_opt, self.student.substrate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(base_opt, T_max=config['epochs'])

        # Performance Tracking (Dicionário Expandido)
        self.history = {k: [] for k in [
            'total', 'contrastive', 'distill', 'topo',
            'lipschitz', 'homeostatic', 'beta_0', 'manifold_err', 'val_rho'
        ]}
        self.best_rho = 0.0
        self.best_state = None
        self.patience_counter = 0

        # Debug flag para mostrar chaves da loss na primeira época
        self._debug_loss_keys = True

    def train_epoch(self, epoch: int, teacher_embeddings: torch.Tensor) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            teacher_embeddings: Teacher embeddings tensor
            
        Returns:
            Dictionary of averaged epoch statistics
        """
        self.student.train()
        
        # Inicializa stats com todas as chaves do log desejado
        epoch_stats = {
            'total': 0.0, 'contrastive': 0.0, 'distill': 0.0, 'topo': 0.0,
            'lipschitz': 0.0, 'homeostatic': 0.0, 'beta_0': 0.0, 'manifold_err': 0.0
        }
        n_batches = 0
        batch_size = self.config['batch_size']

        # Garantir dtype correto nos dados de treino
        teacher_embeddings = teacher_embeddings.to(device=self.device, dtype=self.dtype)

        for i in range(0, len(teacher_embeddings), batch_size):
            batch_t = teacher_embeddings[i:i+batch_size]
            if batch_t.shape[0] < 8:
                continue

            self.optimizer.zero_grad()
            batch_s = self.student(
                batch_t, 
                use_homeostatic=self.config.get('enable_homeostatic', True)
            )

            # Forward e Backward
            losses = self.criterion(batch_s, batch_t, self.student, current_epoch=epoch)
            losses['total'].backward()

            # Debug: mostrar chaves da loss na primeira iteração
            if self._debug_loss_keys and n_batches == 0:
                print(f"  📋 Loss keys disponíveis: {list(losses.keys())}")
                self._debug_loss_keys = False

            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Diagnóstico Científico
            with torch.no_grad():
                m_err = self.student.substrate.manifold_violation(batch_s)

            # Acumula Estatísticas - busca múltiplas variações de chaves
            epoch_stats['total'] += losses['total'].item()
            epoch_stats['contrastive'] += self._extract_loss(losses,
                'loss/contrastive', 'contrastive', 'contrast', 'loss_contrastive')
            epoch_stats['distill'] += self._extract_loss(losses,
                'loss/distill', 'distill', 'distillation', 'loss_distill', 'kl', 'loss/kl')
            epoch_stats['topo'] += self._extract_loss(losses,
                'loss/topo', 'topo', 'topology', 'loss_topo', 'loss/topology')
            epoch_stats['lipschitz'] += self._extract_loss(losses,
                'loss/lipschitz', 'lipschitz', 'lip', 'loss_lipschitz', 'loss/lip')
            epoch_stats['homeostatic'] += self._extract_loss(losses,
                'loss/homeostatic', 'homeostatic', 'homeo', 'loss_homeostatic', 'loss/homeo')
            epoch_stats['beta_0'] += self._extract_loss(losses,
                'metric/beta_0', 'topology/beta_0', 'beta_0', 'beta0', 'connected_components', default=1.0)
            epoch_stats['manifold_err'] += m_err.item()
            n_batches += 1

        self.scheduler.step()
        return {k: v / max(n_batches, 1) for k, v in epoch_stats.items()}

    def _extract_loss(self, losses: Dict, *keys, default: float = 0.0) -> float:
        """Helper para extrair valor de loss do dicionário com múltiplas chaves possíveis."""
        for key in keys:
            if key in losses:
                val = losses[key]
                if isinstance(val, torch.Tensor):
                    return val.item()
                elif isinstance(val, (int, float)):
                    return float(val)
        return default

    def fit(
        self,
        train_teacher_emb: torch.Tensor,
        validation_data: Tuple,
    ) -> Dict:
        """
        Full training loop with validation.
        
        Args:
            train_teacher_emb: Training teacher embeddings
            validation_data: Tuple of (s1_list, s2_list, scores, sent_to_idx, teacher_emb, rho_sbert_val)
            
        Returns:
            Dictionary with training history and best model state
        """
        val_s1, val_s2, val_scores, val_sent_to_idx, val_teacher_emb, rho_sbert_val = validation_data

        # Garantir dtype correto nos dados
        train_teacher_emb = train_teacher_emb.to(device=self.device, dtype=self.dtype)
        val_teacher_emb = val_teacher_emb.to(device=self.device, dtype=self.dtype)

        print(f"\n{'='*60}")
        print(f"STARTING TRAINING [V9.9.3 - FLOAT64]")
        print(f"{'='*60}")
        print(f"  ├─ Device: {self.device}")
        print(f"  ├─ Dtype: {self.dtype}")
        print(f"  └─ Patience: {self.config.get('patience', 10)}")

        for epoch in range(self.config['epochs']):
            avg_stats = self.train_epoch(epoch, train_teacher_emb)

            # Validação STSb
            self.student.eval()
            with torch.no_grad():
                if self.evaluate_fn is not None:
                    val_rho = self.evaluate_fn(
                        self.student, val_s1, val_s2, val_scores,
                        val_sent_to_idx, val_teacher_emb
                    )
                else:
                    val_rho = self._evaluate_stsb(
                        val_s1, val_s2, val_scores,
                        val_sent_to_idx, val_teacher_emb
                    )

            # Persistência no Histórico
            for k, v in avg_stats.items():
                if k in self.history:
                    self.history[k].append(v)
            self.history['val_rho'].append(val_rho)

            # Lógica de Melhor Modelo
            retention = (val_rho / rho_sbert_val) * 100
            marker = ""
            if val_rho > self.best_rho:
                self.best_rho = val_rho
                self.best_state = {
                    k: v.cpu().clone().to(dtype=self.dtype)
                    for k, v in self.student.state_dict().items()
                }
                self.patience_counter = 0
                marker = " ✓"
            else:
                self.patience_counter += 1

            # PRINT FORMATADO
            beta_str = f"{avg_stats['beta_0']:.2f}" if avg_stats['beta_0'] != 1.0 else "N/A"

            print(f"Epoch {epoch+1:3d} | "
                  f"Loss: {avg_stats['total']:.4f} | "
                  f"C: {avg_stats['contrastive']:.4f} D: {avg_stats['distill']:.4f} T: {avg_stats['topo']:.4f} | "
                  f"β₀: {beta_str:>4} | Lip: {avg_stats['lipschitz']:.4f} | "
                  f"Val ρ: {val_rho:.4f} ({retention:.1f}%){marker}")

            if self.patience_counter >= self.config.get('patience', 10):
                print(f"\n⚠️ Early stopping: No improvement for {self.patience_counter} epochs.")
                break

        # Restaurar melhor modelo
        if self.best_state:
            self.student.load_state_dict({
                k: v.to(device=self.device, dtype=self.dtype)
                for k, v in self.best_state.items()
            })

        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE │ Best ρ: {self.best_rho:.4f} ({self.best_rho/rho_sbert_val*100:.1f}%)")
        print(f"{'='*60}")

        return {
            'history': self.history,
            'best_rho': self.best_rho,
            'best_state': self.best_state
        }

    @torch.no_grad()
    def _evaluate_stsb(
        self,
        s1_list: List[str],
        s2_list: List[str],
        scores,
        sent_to_idx: Dict[str, int],
        teacher_emb: torch.Tensor,
        use_homeostatic: bool = False
    ) -> float:
        """
        Evaluate model on STS-B using Spearman correlation.
        
        Built-in evaluation function matching notebook implementation.
        """
        self.student.eval()

        # Ensure correct device and dtype
        teacher_emb = teacher_emb.to(device=self.device, dtype=self.dtype)

        # Get indices
        idx1 = [sent_to_idx[s] for s in s1_list]
        idx2 = [sent_to_idx[s] for s in s2_list]

        # Get embeddings
        emb1 = self.student(teacher_emb[idx1], use_homeostatic=use_homeostatic)
        emb2 = self.student(teacher_emb[idx2], use_homeostatic=use_homeostatic)

        # Compute Lorentz similarity: ⟨u,v⟩_L = -u₀v₀ + Σuᵢvᵢ
        lorentz_inner = -emb1[:, 0] * emb2[:, 0] + (emb1[:, 1:] * emb2[:, 1:]).sum(dim=1)

        # Transform to similarity: 1 / (1 - inner)
        sims = (1.0 / (1.0 - lorentz_inner)).cpu().numpy().astype(np_dtype)

        # Convert scores
        if isinstance(scores, torch.Tensor):
            scores_np = scores.cpu().numpy().astype(np_dtype)
        elif isinstance(scores, list):
            scores_np = np.array(scores, dtype=np_dtype)
        else:
            scores_np = np.array(scores, dtype=np_dtype)

        # Spearman correlation
        rho, _ = spearmanr(sims, scores_np)

        return float(rho) if not np.isnan(rho) else 0.0

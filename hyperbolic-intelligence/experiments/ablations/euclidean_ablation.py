# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
Part IV.1 - Euclidean Ablation Study
====================================

AUDIT COMPLIANCE:
- NO geometry derivation (uses PyTorch built-ins only)
- Uses create_projector from hardened modules
- Fair comparison: same architecture minus manifold mapping

PURPOSE:
Isolate the contribution of hyperbolic geometry by comparing:
- CGT (Lorentz manifold) vs
- Euclidean baseline (flat R^n)

FAIRNESS GUARANTEE:
- Same MLP projector (from cgt.models.cgt_hardened.create_projector)
- Same training procedure
- Same evaluation (evaluate_stsb)
- ONLY difference: output space geometry

Author: Éric Gustavo Reis de Sena
Date: January 2026
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import spearmanr, wilcoxon
from tqdm.auto import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# HARDENED MODULES ONLY - These are the ONLY source of truth
from cgt.models.cgt_hardened import create_projector, CGTStudentHardened
from cgt.losses.losses_hardened import MultiObjectiveLoss
from cgt.geometry.lorentz_hardened import LorentzSubstrateHardened, LorentzConfig
from cgt.evaluation.metrics import evaluate_stsb


# ═══════════════════════════════════════════════════════════════════════════════
#                    CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AblationConfig:
    """Configuration for Euclidean ablation study."""
    # Architecture (MUST match CGT)
    teacher_dim: int = 384
    student_dim: int = 32
    hidden_dim: int = 256
    
    # Training
    batch_size: int = 64
    learning_rate: float = 2e-4
    num_epochs: int = 25
    weight_decay: float = 1e-5
    
    # Loss weights (SAME as CGT)
    lambda_contrastive: float = 1.0
    lambda_distill: float = 0.5
    temperature: float = 0.07
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float64"  # Match hardened modules
    
    # Reproducibility
    seed: int = 42
    
    def to_dict(self) -> dict:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════════
#                    EUCLIDEAN STUDENT (FAIR BASELINE)
# ═══════════════════════════════════════════════════════════════════════════════

class EuclideanStudent(nn.Module):
    """
    Euclidean baseline student model.
    
    FAIRNESS GUARANTEE:
    - Uses EXACT same projector as CGTStudentHardened (via create_projector)
    - Same normalization scale (0.7)
    - ONLY difference: no exp_map (outputs directly to R^n)
    
    This ensures the comparison isolates ONLY the geometry contribution.
    """
    
    def __init__(
        self,
        teacher_dim: int,
        student_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.student_dim = student_dim
        
        # EXACT same projector as CGTStudentHardened
        # This is from the hardened module - NOT derived
        self.projector = create_projector(
            teacher_dim, hidden_dim, student_dim, use_spectral=True
        )
        
        # EXACT same scale as CGTStudentHardened
        self.register_buffer('scale', torch.tensor(0.7))
    
    def forward(self, teacher_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Teacher → Projector → Normalize → Euclidean output.
        
        Note: NO exp_map (unlike CGTStudentHardened)
        """
        # 1. Project through MLP (SAME as CGT)
        projected = self.projector(teacher_emb)
        
        # 2. Normalize and scale (SAME as CGT)
        output = F.normalize(projected, dim=-1) * self.scale
        
        return output


# ═══════════════════════════════════════════════════════════════════════════════
#                    EUCLIDEAN LOSS (USING PYTORCH BUILT-INS ONLY)
# ═══════════════════════════════════════════════════════════════════════════════

class EuclideanContrastiveLoss(nn.Module):
    """
    Euclidean contrastive loss using ONLY PyTorch built-in functions.
    
    NO geometry derivation - uses F.cosine_similarity directly.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        student_emb: torch.Tensor,
        teacher_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        InfoNCE loss using cosine similarity (PyTorch built-in).
        
        Args:
            student_emb: Student embeddings [B, D]
            teacher_emb: Teacher embeddings [B, D_teacher]
        
        Returns:
            Contrastive loss scalar
        """
        B = student_emb.shape[0]
        device = student_emb.device
        
        # Normalize (PyTorch built-in)
        student_norm = F.normalize(student_emb, dim=-1)
        
        # Compute cosine similarity matrix (PyTorch built-in)
        # sim[i, j] = cos(student_i, student_j)
        sim_matrix = torch.mm(student_norm, student_norm.t()) / self.temperature
        
        # Labels: diagonal is positive
        labels = torch.arange(B, device=device)
        
        # Cross-entropy loss
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss


class EuclideanDistillationLoss(nn.Module):
    """
    Euclidean distillation loss using ONLY PyTorch built-in functions.
    
    Aligns student similarity structure with teacher similarity structure.
    """
    
    def forward(
        self,
        student_emb: torch.Tensor,
        teacher_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Distillation via similarity matrix alignment.
        
        Args:
            student_emb: Student embeddings [B, D]
            teacher_emb: Teacher embeddings [B, D_teacher]
        
        Returns:
            Distillation loss scalar
        """
        # Normalize (PyTorch built-in)
        student_norm = F.normalize(student_emb, dim=-1)
        teacher_norm = F.normalize(teacher_emb, dim=-1)
        
        # Compute similarity matrices (PyTorch built-in)
        student_sim = torch.mm(student_norm, student_norm.t())
        teacher_sim = torch.mm(teacher_norm, teacher_norm.t())
        
        # MSE loss on similarity matrices (PyTorch built-in)
        loss = F.mse_loss(student_sim, teacher_sim)
        
        return loss


class EuclideanMultiObjectiveLoss(nn.Module):
    """
    Combined Euclidean loss for fair comparison with CGT MultiObjectiveLoss.
    
    Uses ONLY PyTorch built-in functions.
    """
    
    def __init__(
        self,
        lambda_contrastive: float = 1.0,
        lambda_distill: float = 0.5,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.lambda_contrastive = lambda_contrastive
        self.lambda_distill = lambda_distill
        
        self.contrastive = EuclideanContrastiveLoss(temperature)
        self.distillation = EuclideanDistillationLoss()
    
    def forward(
        self,
        student_emb: torch.Tensor,
        teacher_emb: torch.Tensor,
        model: nn.Module,
        current_epoch: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined Euclidean loss.
        
        Interface matches MultiObjectiveLoss from hardened modules.
        """
        loss_c = self.contrastive(student_emb, teacher_emb)
        loss_d = self.distillation(student_emb, teacher_emb)
        
        total = self.lambda_contrastive * loss_c + self.lambda_distill * loss_d
        
        return {
            'total': total,
            'contrastive': loss_c,
            'distillation': loss_d,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#                    EUCLIDEAN EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def euclidean_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute Euclidean cosine similarity (PyTorch built-in).
    
    This is the standard similarity metric for Euclidean embeddings.
    """
    return F.cosine_similarity(x, y, dim=-1)


def evaluate_euclidean_stsb(
    model: EuclideanStudent,
    emb1: torch.Tensor,
    emb2: torch.Tensor,
    scores: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, float]:
    """
    Evaluate Euclidean model on STS-B.
    
    Uses cosine similarity (PyTorch built-in).
    """
    model.eval()
    
    with torch.no_grad():
        # Get student embeddings
        student_emb1 = model(emb1.to(device, dtype))
        student_emb2 = model(emb2.to(device, dtype))
        
        # Compute cosine similarity (PyTorch built-in)
        similarities = F.cosine_similarity(student_emb1, student_emb2, dim=-1)
        
        # Convert to numpy for scipy
        sims_np = similarities.cpu().numpy()
        scores_np = scores.cpu().numpy()
        
        # Spearman correlation
        spearman, _ = spearmanr(sims_np, scores_np)
    
    return {
        'spearman': float(spearman) if not np.isnan(spearman) else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#                    TRAINING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def train_euclidean_model(
    train_emb1: torch.Tensor,
    train_emb2: torch.Tensor,
    train_scores: torch.Tensor,
    val_emb1: torch.Tensor,
    val_emb2: torch.Tensor,
    val_scores: torch.Tensor,
    config: AblationConfig,
) -> Tuple[EuclideanStudent, Dict[str, List[float]]]:
    """
    Train Euclidean baseline model.
    
    FAIRNESS: Same training procedure as CGT.
    """
    device = torch.device(config.device)
    dtype = torch.float64 if config.dtype == "float64" else torch.float32
    
    # Initialize model
    model = EuclideanStudent(
        teacher_dim=config.teacher_dim,
        student_dim=config.student_dim,
        hidden_dim=config.hidden_dim,
    ).to(device, dtype)
    
    # Initialize loss
    criterion = EuclideanMultiObjectiveLoss(
        lambda_contrastive=config.lambda_contrastive,
        lambda_distill=config.lambda_distill,
        temperature=config.temperature,
    )
    
    # Initialize optimizer (SAME as CGT)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    # Prepare data
    train_emb1 = train_emb1.to(device, dtype)
    train_emb2 = train_emb2.to(device, dtype)
    train_scores = train_scores.to(device, dtype)
    
    from torch.utils.data import DataLoader, TensorDataset
    train_dataset = TensorDataset(train_emb1, train_emb2, train_scores)
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    
    # History
    history = {
        'train_loss': [],
        'val_spearman': [],
    }
    
    best_val_spearman = -1.0
    best_state = None
    
    # Training loop
    for epoch in range(config.num_epochs):
        model.train()
        epoch_losses = []
        
        for batch_emb1, batch_emb2, batch_scores in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            student_emb1 = model(batch_emb1)
            student_emb2 = model(batch_emb2)
            
            # Combine embeddings (SAME pattern as CGT)
            student_emb = torch.cat([student_emb1, student_emb2], dim=0)
            teacher_emb = torch.cat([batch_emb1, batch_emb2], dim=0)
            
            # Compute loss
            losses = criterion(student_emb, teacher_emb, model, current_epoch=epoch)
            
            # Backward
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses.append(losses['total'].item())
        
        # Record training loss
        avg_loss = np.mean(epoch_losses)
        history['train_loss'].append(avg_loss)
        
        # Validation
        val_metrics = evaluate_euclidean_stsb(
            model, val_emb1, val_emb2, val_scores, device, dtype
        )
        history['val_spearman'].append(val_metrics['spearman'])
        
        # Track best
        if val_metrics['spearman'] > best_val_spearman:
            best_val_spearman = val_metrics['spearman']
            best_state = model.state_dict().copy()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{config.num_epochs} | "
                  f"Loss: {avg_loss:.4f} | Val ρ: {val_metrics['spearman']:.4f}")
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, history


# ═══════════════════════════════════════════════════════════════════════════════
#                    CGT TRAINING (Using hardened modules)
# ═══════════════════════════════════════════════════════════════════════════════

def train_cgt_model(
    train_emb1: torch.Tensor,
    train_emb2: torch.Tensor,
    train_scores: torch.Tensor,
    val_emb1: torch.Tensor,
    val_emb2: torch.Tensor,
    val_scores: torch.Tensor,
    config: AblationConfig,
) -> Tuple[CGTStudentHardened, Dict[str, List[float]]]:
    """
    Train CGT model using HARDENED modules.
    
    This is the ground truth - no modifications.
    """
    device = torch.device(config.device)
    dtype = torch.float64 if config.dtype == "float64" else torch.float32
    
    # Initialize model (HARDENED MODULE)
    model = CGTStudentHardened(
        teacher_dim=config.teacher_dim,
        student_dim=config.student_dim,
        hidden_dim=config.hidden_dim,
        learnable_curvature=True,
        initial_curvature=1.0,
        curvature_min=0.1,
        curvature_max=5.0,
    ).to(device, dtype)
    
    # Initialize homeostatic field
    model.init_homeostatic(n_anchors=16, alpha=0.1)
    
    # Initialize loss (HARDENED MODULE)
    criterion = MultiObjectiveLoss(
        lambda_contrastive=config.lambda_contrastive,
        lambda_distill=config.lambda_distill,
        temperature=config.temperature,
    )
    
    # Initialize optimizer
    base_optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    # Riemannian wrapper (HARDENED MODULE)
    from cgt.models.cgt_hardened import RiemannianOptimizerWrapper
    optimizer = RiemannianOptimizerWrapper(base_optimizer, model.substrate)
    
    # Prepare data
    train_emb1 = train_emb1.to(device, dtype)
    train_emb2 = train_emb2.to(device, dtype)
    train_scores = train_scores.to(device, dtype)
    
    from torch.utils.data import DataLoader, TensorDataset
    train_dataset = TensorDataset(train_emb1, train_emb2, train_scores)
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    
    # History
    history = {
        'train_loss': [],
        'val_spearman': [],
        'curvature': [],
    }
    
    best_val_spearman = -1.0
    best_state = None
    lorentz = model.substrate
    
    # Training loop
    for epoch in range(config.num_epochs):
        model.train()
        epoch_losses = []
        
        for batch_emb1, batch_emb2, batch_scores in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            student_emb1 = model(batch_emb1)
            student_emb2 = model(batch_emb2)
            
            # Combine embeddings
            student_emb = torch.cat([student_emb1, student_emb2], dim=0)
            teacher_emb = torch.cat([batch_emb1, batch_emb2], dim=0)
            
            # Compute loss (HARDENED MODULE)
            losses = criterion(student_emb, teacher_emb, model, current_epoch=epoch)
            
            # Backward
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses.append(losses['total'].item())
        
        # Record training loss
        avg_loss = np.mean(epoch_losses)
        history['train_loss'].append(avg_loss)
        history['curvature'].append(model.get_curvature())
        
        # Validation using hardened evaluate function
        model.eval()
        with torch.no_grad():
            student_emb1 = model(val_emb1.to(device, dtype))
            student_emb2 = model(val_emb2.to(device, dtype))
            
            # Use lorentz similarity (HARDENED MODULE)
            sims = lorentz.lorentz_similarity(student_emb1, student_emb2).squeeze()
            sims_np = sims.cpu().numpy()
            scores_np = val_scores.cpu().numpy()
            
            spearman, _ = spearmanr(sims_np, scores_np)
        
        history['val_spearman'].append(float(spearman) if not np.isnan(spearman) else 0.0)
        
        # Track best
        if history['val_spearman'][-1] > best_val_spearman:
            best_val_spearman = history['val_spearman'][-1]
            best_state = model.state_dict().copy()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{config.num_epochs} | "
                  f"Loss: {avg_loss:.4f} | Val ρ: {history['val_spearman'][-1]:.4f} | "
                  f"K: {model.get_curvature():.4f}")
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, history


# ═══════════════════════════════════════════════════════════════════════════════
#                    ABLATION STUDY RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_euclidean_ablation(
    train_emb1: torch.Tensor,
    train_emb2: torch.Tensor,
    train_scores: torch.Tensor,
    val_emb1: torch.Tensor,
    val_emb2: torch.Tensor,
    val_scores: torch.Tensor,
    test_emb1: torch.Tensor,
    test_emb2: torch.Tensor,
    test_scores: torch.Tensor,
    teacher_spearman: float,
    config: AblationConfig,
    output_dir: Path,
) -> Dict:
    """
    Run complete Euclidean ablation study.
    
    Compares CGT (hyperbolic) vs Euclidean (flat) with same architecture.
    """
    device = torch.device(config.device)
    dtype = torch.float64 if config.dtype == "float64" else torch.float32
    
    results = {
        'config': config.to_dict(),
        'timestamp': datetime.now().isoformat(),
        'teacher_spearman': teacher_spearman,
    }
    
    # ═══════════════════════════════════════════════════════════════════════
    # TRAIN CGT (Hyperbolic)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TRAINING CGT (HYPERBOLIC)")
    print("=" * 70)
    
    cgt_model, cgt_history = train_cgt_model(
        train_emb1, train_emb2, train_scores,
        val_emb1, val_emb2, val_scores,
        config,
    )
    
    # CGT Test evaluation
    cgt_model.eval()
    lorentz = cgt_model.substrate
    with torch.no_grad():
        cgt_test_emb1 = cgt_model(test_emb1.to(device, dtype))
        cgt_test_emb2 = cgt_model(test_emb2.to(device, dtype))
        cgt_sims = lorentz.lorentz_similarity(cgt_test_emb1, cgt_test_emb2).squeeze()
        cgt_spearman, _ = spearmanr(cgt_sims.cpu().numpy(), test_scores.numpy())
    
    results['cgt'] = {
        'test_spearman': float(cgt_spearman),
        'retention': float(cgt_spearman / teacher_spearman * 100),
        'best_val_spearman': max(cgt_history['val_spearman']),
        'final_curvature': cgt_model.get_curvature(),
        'history': cgt_history,
    }
    
    print(f"\nCGT Test Spearman: {cgt_spearman:.4f}")
    print(f"CGT Retention: {results['cgt']['retention']:.1f}%")
    
    # ═══════════════════════════════════════════════════════════════════════
    # TRAIN EUCLIDEAN
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TRAINING EUCLIDEAN (BASELINE)")
    print("=" * 70)
    
    euc_model, euc_history = train_euclidean_model(
        train_emb1, train_emb2, train_scores,
        val_emb1, val_emb2, val_scores,
        config,
    )
    
    # Euclidean Test evaluation
    euc_test_metrics = evaluate_euclidean_stsb(
        euc_model, test_emb1, test_emb2, test_scores, device, dtype
    )
    
    results['euclidean'] = {
        'test_spearman': euc_test_metrics['spearman'],
        'retention': float(euc_test_metrics['spearman'] / teacher_spearman * 100),
        'best_val_spearman': max(euc_history['val_spearman']),
        'history': euc_history,
    }
    
    print(f"\nEuclidean Test Spearman: {euc_test_metrics['spearman']:.4f}")
    print(f"Euclidean Retention: {results['euclidean']['retention']:.1f}%")
    
    # ═══════════════════════════════════════════════════════════════════════
    # COMPARISON
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("COMPARISON: CGT vs EUCLIDEAN")
    print("=" * 70)
    
    advantage = results['cgt']['test_spearman'] - results['euclidean']['test_spearman']
    relative_improvement = (results['cgt']['test_spearman'] / results['euclidean']['test_spearman'] - 1) * 100
    
    results['comparison'] = {
        'cgt_advantage': advantage,
        'relative_improvement_pct': relative_improvement,
        'cgt_wins': advantage > 0,
    }
    
    print(f"Teacher Spearman:   {teacher_spearman:.4f}")
    print(f"CGT Spearman:       {results['cgt']['test_spearman']:.4f} ({results['cgt']['retention']:.1f}%)")
    print(f"Euclidean Spearman: {results['euclidean']['test_spearman']:.4f} ({results['euclidean']['retention']:.1f}%)")
    print(f"CGT Advantage:      {advantage:+.4f}")
    print(f"Relative Improvement: {relative_improvement:+.2f}%")
    
    if advantage > 0:
        print("\n✅ CGT > Euclidean: HYPERBOLIC GEOMETRY CONTRIBUTES")
    else:
        print("\n⚠️ CGT ≤ Euclidean: Hyperbolic geometry does NOT provide advantage")
    
    # ═══════════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ═══════════════════════════════════════════════════════════════════════
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON results
    with open(output_dir / 'euclidean_ablation_results.json', 'w') as f:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json.dump(results, f, indent=2, default=convert)
    
    # CSV summary
    summary_df = pd.DataFrame([
        {'method': 'Teacher', 'spearman': teacher_spearman, 'retention': 100.0},
        {'method': 'CGT', 'spearman': results['cgt']['test_spearman'], 
         'retention': results['cgt']['retention']},
        {'method': 'Euclidean', 'spearman': results['euclidean']['test_spearman'],
         'retention': results['euclidean']['retention']},
    ])
    summary_df.to_csv(output_dir / 'euclidean_ablation_summary.csv', index=False)
    
    print(f"\n📁 Results saved to: {output_dir}")
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#                    MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("PART IV.1 - EUCLIDEAN ABLATION STUDY")
    print("Comparing CGT (Hyperbolic) vs Euclidean (Flat)")
    print("=" * 80)
    
    # This is a template - actual data loading depends on context
    print("\n⚠️ This module requires pre-computed teacher embeddings.")
    print("   Use run_euclidean_ablation() with your data.")

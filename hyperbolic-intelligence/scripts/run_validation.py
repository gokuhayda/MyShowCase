#!/usr/bin/env python3
"""
CGT Validation Protocol - Complete Self-Contained Script
=========================================================

Implements the full validation protocol for CGT experiments:
1. 5-fold cross-validation on train+val
2. Final model training
3. External held-out evaluation
4. Generalization gap analysis

Usage:
    python run_validation.py

Author: CGT Research Team
License: MIT
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import spearmanr, pearsonr
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Optional
from pathlib import Path
import copy
import sys

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class ValidationConfig:
    """Configuration for validation protocol."""
    n_folds: int = 5
    random_seed: int = 42
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 1e-4
    patience: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    cgt_gw_batch_size: int = 64


# ==============================================================================
# REPRODUCIBILITY
# ==============================================================================

def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==============================================================================
# K-FOLD CROSS-VALIDATION
# ==============================================================================

def create_k_folds(
    n_samples: int,
    n_folds: int,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create k-fold train/val index splits."""
    np.random.seed(seed)
    indices = np.random.permutation(n_samples)
    fold_size = n_samples // n_folds
    
    folds = []
    for i in range(n_folds):
        val_start = i * fold_size
        val_end = val_start + fold_size if i < n_folds - 1 else n_samples
        
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
        
        folds.append((train_idx, val_idx))
    
    return folds


# ==============================================================================
# CGT-GW PROJECTION
# ==============================================================================

def project_through_cgt_gw(
    cgt_gw: nn.Module,
    emb1: torch.Tensor,
    emb2: torch.Tensor,
    device: str,
    batch_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project embeddings through CGT-GW projector.
    Handles batching and dtype consistency (float64 for hyperbolic geometry).
    """
    cgt_gw.eval()
    device_obj = torch.device(device)
    cgt_gw = cgt_gw.to(device_obj)
    
    projected1_list = []
    projected2_list = []
    
    n_samples = len(emb1)
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_emb1 = emb1[i:i+batch_size].to(device_obj).double()
            batch_emb2 = emb2[i:i+batch_size].to(device_obj).double()
            
            proj1 = cgt_gw(batch_emb1).cpu()
            proj2 = cgt_gw(batch_emb2).cpu()
            
            projected1_list.append(proj1)
            projected2_list.append(proj2)
    
    return torch.cat(projected1_list, dim=0), torch.cat(projected2_list, dim=0)


# ==============================================================================
# TRAINING
# ==============================================================================

def train_single_fold(
    student_factory: Callable[[], nn.Module],
    train_emb1: torch.Tensor,
    train_emb2: torch.Tensor,
    train_scores: torch.Tensor,
    val_emb1: torch.Tensor,
    val_emb2: torch.Tensor,
    val_scores: torch.Tensor,
    config: ValidationConfig,
) -> Tuple[nn.Module, Dict[str, float]]:
    """Train student model on single fold."""
    device = torch.device(config.device)
    student = student_factory().to(device).double()
    
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=config.learning_rate,
        weight_decay=1e-5,
    )
    
    train_dataset = TensorDataset(
        train_emb1.double(),
        train_emb2.double(),
        train_scores.double(),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )
    
    best_val_rho = -1.0
    best_state = None
    patience_counter = 0
    
    for epoch in range(config.epochs):
        student.train()
        for emb1, emb2, scores in train_loader:
            emb1 = emb1.to(device)
            emb2 = emb2.to(device)
            scores = scores.to(device)
            
            optimizer.zero_grad()
            
            z1 = student(emb1)
            z2 = student(emb2)
            sims = F.cosine_similarity(z1, z2)
            scores_norm = scores / 5.0  # STS normalization
            loss = F.mse_loss(sims, scores_norm)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
        
        # Validation
        student.eval()
        with torch.no_grad():
            v_emb1 = val_emb1.to(device).double()
            v_emb2 = val_emb2.to(device).double()
            
            z1 = student(v_emb1)
            z2 = student(v_emb2)
            sims = F.cosine_similarity(z1, z2)
            
            val_rho, _ = spearmanr(sims.cpu().numpy(), val_scores.numpy())
        
        if val_rho > best_val_rho:
            best_val_rho = val_rho
            best_state = copy.deepcopy(student.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break
    
    if best_state is not None:
        student.load_state_dict(best_state)
    
    # Final metrics
    student.eval()
    with torch.no_grad():
        v_emb1 = val_emb1.to(device).double()
        v_emb2 = val_emb2.to(device).double()
        
        z1 = student(v_emb1)
        z2 = student(v_emb2)
        sims = F.cosine_similarity(z1, z2)
        
        final_rho, _ = spearmanr(sims.cpu().numpy(), val_scores.numpy())
        final_pearson, _ = pearsonr(sims.cpu().numpy(), val_scores.numpy())
    
    return student, {"spearman": final_rho, "pearson": final_pearson}


# ==============================================================================
# CROSS-VALIDATION
# ==============================================================================

def run_cross_validation(
    cgt_gw: nn.Module,
    student_factory: Callable[[], nn.Module],
    emb1: torch.Tensor,
    emb2: torch.Tensor,
    scores: torch.Tensor,
    config: ValidationConfig,
) -> Dict:
    """Run k-fold CV with CGT-GW projection."""
    set_seed(config.random_seed)
    
    print(f"  Projecting through CGT-GW...")
    projected_emb1, projected_emb2 = project_through_cgt_gw(
        cgt_gw=cgt_gw,
        emb1=emb1,
        emb2=emb2,
        device=config.device,
        batch_size=config.cgt_gw_batch_size,
    )
    
    n_samples = len(scores)
    folds = create_k_folds(n_samples, config.n_folds, config.random_seed)
    
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"  Fold {fold_idx + 1}/{config.n_folds}...", end=" ", flush=True)
        set_seed(config.random_seed + fold_idx)
        
        fold_train_emb1 = projected_emb1[train_idx]
        fold_train_emb2 = projected_emb2[train_idx]
        fold_train_scores = scores[train_idx]
        
        fold_val_emb1 = projected_emb1[val_idx]
        fold_val_emb2 = projected_emb2[val_idx]
        fold_val_scores = scores[val_idx]
        
        _, metrics = train_single_fold(
            student_factory=student_factory,
            train_emb1=fold_train_emb1,
            train_emb2=fold_train_emb2,
            train_scores=fold_train_scores,
            val_emb1=fold_val_emb1,
            val_emb2=fold_val_emb2,
            val_scores=fold_val_scores,
            config=config,
        )
        
        fold_results.append(metrics)
        print(f"ρ = {metrics['spearman']:.4f}")
    
    spearman_scores = [r["spearman"] for r in fold_results]
    pearson_scores = [r["pearson"] for r in fold_results]
    
    return {
        "fold_results": fold_results,
        "cv_spearman_mean": np.mean(spearman_scores),
        "cv_spearman_std": np.std(spearman_scores),
        "cv_pearson_mean": np.mean(pearson_scores),
        "cv_pearson_std": np.std(pearson_scores),
        "n_folds": config.n_folds,
    }


# ==============================================================================
# EXTERNAL EVALUATION
# ==============================================================================

def evaluate_external(
    cgt_gw: nn.Module,
    student: nn.Module,
    test_emb1: torch.Tensor,
    test_emb2: torch.Tensor,
    test_scores: np.ndarray,
    config: ValidationConfig,
) -> Dict:
    """Evaluate on external held-out dataset."""
    
    projected_emb1, projected_emb2 = project_through_cgt_gw(
        cgt_gw=cgt_gw,
        emb1=test_emb1,
        emb2=test_emb2,
        device=config.device,
        batch_size=config.cgt_gw_batch_size,
    )
    
    device = torch.device(config.device)
    student = student.to(device).double()
    student.eval()
    
    with torch.no_grad():
        t_emb1 = projected_emb1.to(device).double()
        t_emb2 = projected_emb2.to(device).double()
        
        z1 = student(t_emb1)
        z2 = student(t_emb2)
        sims = F.cosine_similarity(z1, z2).cpu().numpy()
    
    spearman_rho, spearman_p = spearmanr(sims, test_scores)
    pearson_r, pearson_p = pearsonr(sims, test_scores)
    
    return {
        "spearman": spearman_rho,
        "spearman_pvalue": spearman_p,
        "pearson": pearson_r,
        "pearson_pvalue": pearson_p,
        "n_samples": len(test_scores),
    }


# ==============================================================================
# SUMMARY
# ==============================================================================

def print_validation_summary(results: Dict) -> None:
    """Print concise validation summary."""
    cv = results["cross_validation"]
    ext = results["external_evaluation"]
    
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    print(f"\n{'Metric':<20} {'CV (5-fold)':<20} {'External Test':<15}")
    print("-" * 55)
    
    cv_spearman = f"{cv['cv_spearman_mean']:.4f} ± {cv['cv_spearman_std']:.4f}"
    ext_spearman = f"{ext['spearman']:.4f}"
    print(f"{'Spearman ρ':<20} {cv_spearman:<20} {ext_spearman:<15}")
    
    cv_pearson = f"{cv['cv_pearson_mean']:.4f} ± {cv['cv_pearson_std']:.4f}"
    ext_pearson = f"{ext['pearson']:.4f}"
    print(f"{'Pearson r':<20} {cv_pearson:<20} {ext_pearson:<15}")
    
    gap = ext["spearman"] - cv["cv_spearman_mean"]
    gap_sign = "+" if gap >= 0 else ""
    print(f"\n{'Generalization Gap':<20} {gap_sign}{gap:.4f}")
    
    print("\n" + "-" * 55)
    if abs(gap) < cv["cv_spearman_std"]:
        print("✓ External performance within CV confidence interval")
        print("  → Model generalizes well to unseen data")
    elif gap < 0:
        print("⚠ External performance below CV mean")
        print(f"  → Potential overfitting ({abs(gap)/cv['cv_spearman_std']:.1f}σ below)")
    else:
        print("✓ External performance above CV mean")
        print("  → Model generalizes favorably")
    
    print("=" * 60)


# ==============================================================================
# MAIN PROTOCOL
# ==============================================================================

def run_validation_protocol(
    cgt_gw: nn.Module,
    student_factory: Callable[[], nn.Module],
    train_emb1: torch.Tensor,
    train_emb2: torch.Tensor,
    train_scores: torch.Tensor,
    val_emb1: torch.Tensor,
    val_emb2: torch.Tensor,
    val_scores: torch.Tensor,
    external_test_emb1: torch.Tensor,
    external_test_emb2: torch.Tensor,
    external_test_scores: np.ndarray,
    config: Optional[ValidationConfig] = None,
    dataset_name: str = "STS-B",
    external_dataset_name: str = "STS12-16",
) -> Dict:
    """
    Complete CGT validation protocol.
    
    1. 5-fold CV on train+val (projected through CGT-GW)
    2. Train final student
    3. Evaluate on external held-out dataset
    """
    if config is None:
        config = ValidationConfig()
    
    print("=" * 60)
    print("CGT VALIDATION PROTOCOL")
    print(f"Dataset: {dataset_name}")
    print(f"External: {external_dataset_name}")
    print(f"Device: {config.device}")
    print("=" * 60)
    
    # Combine train + val for CV
    combined_emb1 = torch.cat([train_emb1, val_emb1], dim=0)
    combined_emb2 = torch.cat([train_emb2, val_emb2], dim=0)
    combined_scores = torch.cat([train_scores, val_scores], dim=0)
    
    # Step 1: Cross-validation
    print(f"\n[1/3] {config.n_folds}-Fold Cross-Validation...")
    print(f"      Samples: {len(combined_scores)}")
    
    cv_results = run_cross_validation(
        cgt_gw=cgt_gw,
        student_factory=student_factory,
        emb1=combined_emb1,
        emb2=combined_emb2,
        scores=combined_scores,
        config=config,
    )
    
    # Step 2: Train final model
    print(f"\n[2/3] Training Final Model...")
    print(f"      Train: {len(train_scores)}, Val: {len(val_scores)}")
    
    set_seed(config.random_seed)
    
    proj_train_emb1, proj_train_emb2 = project_through_cgt_gw(
        cgt_gw=cgt_gw,
        emb1=train_emb1,
        emb2=train_emb2,
        device=config.device,
        batch_size=config.cgt_gw_batch_size,
    )
    proj_val_emb1, proj_val_emb2 = project_through_cgt_gw(
        cgt_gw=cgt_gw,
        emb1=val_emb1,
        emb2=val_emb2,
        device=config.device,
        batch_size=config.cgt_gw_batch_size,
    )
    
    final_student, _ = train_single_fold(
        student_factory=student_factory,
        train_emb1=proj_train_emb1,
        train_emb2=proj_train_emb2,
        train_scores=train_scores,
        val_emb1=proj_val_emb1,
        val_emb2=proj_val_emb2,
        val_scores=val_scores,
        config=config,
    )
    print("      Done.")
    
    # Step 3: External evaluation
    print(f"\n[3/3] External Dataset Evaluation...")
    print(f"      Samples: {len(external_test_scores)}")
    
    external_results = evaluate_external(
        cgt_gw=cgt_gw,
        student=final_student,
        test_emb1=external_test_emb1,
        test_emb2=external_test_emb2,
        test_scores=external_test_scores,
        config=config,
    )
    
    # Compile results
    results = {
        "config": {
            "n_folds": config.n_folds,
            "random_seed": config.random_seed,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
        },
        "cross_validation": cv_results,
        "external_evaluation": external_results,
        "dataset_name": dataset_name,
        "external_dataset_name": external_dataset_name,
    }
    
    print_validation_summary(results)
    
    return results


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CGT VALIDATION PROTOCOL")
    print("=" * 60)
    print()
    print("Usage example:")
    print()
    print("""
from run_validation import run_validation_protocol, ValidationConfig
from cgt.models.cgt_gw_projector import CGTGWProjector, CGTGWProjectorConfig
from cgt.models.cgt_hardened import CGTStudentHardened

# Load your data
data = torch.load("path/to/embeddings.pt")

# Load or create CGT-GW
cgt_gw = CGTGWProjector(CGTGWProjectorConfig(input_dim=384, output_dim=256))
cgt_gw.load_state_dict(torch.load("path/to/cgt_gw.pth"))

# Define student factory
def student_factory():
    return CGTStudentHardened(teacher_dim=256, student_dim=32, hidden_dim=256)

# Run validation
results = run_validation_protocol(
    cgt_gw=cgt_gw,
    student_factory=student_factory,
    train_emb1=data["train_emb1"],
    train_emb2=data["train_emb2"],
    train_scores=data["train_scores"],
    val_emb1=data["val_emb1"],
    val_emb2=data["val_emb2"],
    val_scores=data["val_scores"],
    external_test_emb1=data["test_emb1"],
    external_test_emb2=data["test_emb2"],
    external_test_scores=data["test_scores"],
    config=ValidationConfig(n_folds=5, epochs=50, device="cuda"),
    dataset_name="STS-B",
    external_dataset_name="STS12-16",
)
    """)

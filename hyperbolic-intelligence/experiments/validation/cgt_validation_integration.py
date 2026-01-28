"""
CGT Validation Protocol - Integration Script
=============================================

Integrates with the CGT pipeline for real experimental runs.

Usage:
    from cgt_validation_integration import run_cgt_validation
    
    results = run_cgt_validation(
        cgt_gw_projector=projector,
        student_factory=lambda: CGTStudentHardened(...),
        train_data=data["train_*"],
        val_data=data["val_*"],
        external_data=external_dataset,
    )

Author: CGT Research Team
License: MIT
"""

import sys
sys.path.insert(0, "src")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import spearmanr, pearsonr
from typing import Dict, Callable, Optional, Tuple
from dataclasses import dataclass
import copy

# Import validation protocol functions
from cgt_validation_protocol import (
    ValidationConfig,
    set_seed,
    create_k_folds,
    print_validation_summary,
)


@dataclass
class CGTValidationConfig(ValidationConfig):
    """Extended config for CGT pipeline."""
    use_cgt_gw: bool = True
    cgt_gw_batch_size: int = 64


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
    device = torch.device(device)
    cgt_gw = cgt_gw.to(device)
    
    projected1_list = []
    projected2_list = []
    
    n_samples = len(emb1)
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_emb1 = emb1[i:i+batch_size].to(device).double()
            batch_emb2 = emb2[i:i+batch_size].to(device).double()
            
            proj1 = cgt_gw(batch_emb1).cpu()
            proj2 = cgt_gw(batch_emb2).cpu()
            
            projected1_list.append(proj1)
            projected2_list.append(proj2)
    
    return torch.cat(projected1_list, dim=0), torch.cat(projected2_list, dim=0)


def train_student_fold(
    student_factory: Callable[[], nn.Module],
    train_emb1: torch.Tensor,
    train_emb2: torch.Tensor,
    train_scores: torch.Tensor,
    val_emb1: torch.Tensor,
    val_emb2: torch.Tensor,
    val_scores: torch.Tensor,
    config: CGTValidationConfig,
) -> Tuple[nn.Module, Dict[str, float]]:
    """Train student model on single fold."""
    device = torch.device(config.device)
    student = student_factory().to(device)
    
    # Ensure student is float64 compatible
    student = student.double()
    
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


def run_cgt_cross_validation(
    cgt_gw: nn.Module,
    student_factory: Callable[[], nn.Module],
    train_emb1: torch.Tensor,
    train_emb2: torch.Tensor,
    train_scores: torch.Tensor,
    val_emb1: torch.Tensor,
    val_emb2: torch.Tensor,
    val_scores: torch.Tensor,
    config: CGTValidationConfig,
) -> Dict:
    """Run k-fold CV with CGT-GW projection."""
    
    # Combine train + val
    combined_emb1 = torch.cat([train_emb1, val_emb1], dim=0)
    combined_emb2 = torch.cat([train_emb2, val_emb2], dim=0)
    combined_scores = torch.cat([train_scores, val_scores], dim=0)
    
    print(f"  Projecting through CGT-GW...")
    projected_emb1, projected_emb2 = project_through_cgt_gw(
        cgt_gw=cgt_gw,
        emb1=combined_emb1,
        emb2=combined_emb2,
        device=config.device,
        batch_size=config.cgt_gw_batch_size,
    )
    
    n_samples = len(combined_scores)
    folds = create_k_folds(n_samples, config.n_folds, config.random_seed)
    
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"  Fold {fold_idx + 1}/{config.n_folds}...", end=" ", flush=True)
        set_seed(config.random_seed + fold_idx)
        
        fold_train_emb1 = projected_emb1[train_idx]
        fold_train_emb2 = projected_emb2[train_idx]
        fold_train_scores = combined_scores[train_idx]
        
        fold_val_emb1 = projected_emb1[val_idx]
        fold_val_emb2 = projected_emb2[val_idx]
        fold_val_scores = combined_scores[val_idx]
        
        _, metrics = train_student_fold(
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


def evaluate_cgt_external(
    cgt_gw: nn.Module,
    student: nn.Module,
    test_emb1: torch.Tensor,
    test_emb2: torch.Tensor,
    test_scores: np.ndarray,
    config: CGTValidationConfig,
) -> Dict:
    """Evaluate on external held-out dataset."""
    
    # Project through CGT-GW
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


def run_cgt_validation(
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
    config: Optional[CGTValidationConfig] = None,
    dataset_name: str = "STS-B",
    external_dataset_name: str = "STS12-16",
) -> Dict:
    """
    Complete CGT validation protocol.
    
    1. Project all data through CGT-GW
    2. Run 5-fold CV on projected train+val
    3. Train final student on projected data
    4. Evaluate on projected external test set
    """
    if config is None:
        config = CGTValidationConfig()
    
    print("=" * 60)
    print("CGT VALIDATION PROTOCOL")
    print(f"Dataset: {dataset_name}")
    print(f"External: {external_dataset_name}")
    print("=" * 60)
    
    # Step 1: Cross-validation
    print(f"\n[1/3] {config.n_folds}-Fold Cross-Validation...")
    cv_results = run_cgt_cross_validation(
        cgt_gw=cgt_gw,
        student_factory=student_factory,
        train_emb1=train_emb1,
        train_emb2=train_emb2,
        train_scores=train_scores,
        val_emb1=val_emb1,
        val_emb2=val_emb2,
        val_scores=val_scores,
        config=config,
    )
    
    # Step 2: Train final model
    print(f"\n[2/3] Training Final Model...")
    set_seed(config.random_seed)
    
    # Project train/val through CGT-GW
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
    
    final_student, _ = train_student_fold(
        student_factory=student_factory,
        train_emb1=proj_train_emb1,
        train_emb2=proj_train_emb2,
        train_scores=train_scores,
        val_emb1=proj_val_emb1,
        val_emb2=proj_val_emb2,
        val_scores=val_scores,
        config=config,
    )
    
    # Step 3: External evaluation
    print(f"\n[3/3] External Dataset Evaluation...")
    external_results = evaluate_cgt_external(
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
# CLI ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    print("CGT Validation Integration Script")
    print("=" * 60)
    print("This script requires:")
    print("  - Trained CGTGWProjector")
    print("  - Student model factory")
    print("  - Preprocessed embeddings")
    print()
    print("Usage:")
    print("  from cgt_validation_integration import run_cgt_validation")
    print("  results = run_cgt_validation(cgt_gw, student_factory, ...)")

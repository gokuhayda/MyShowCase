"""
CGT Experimental Validation Protocol
=====================================

5-Fold Cross-Validation + External Held-Out Evaluation

This script implements the recommended validation protocol for
scientific reproducibility in embedding compression research.

Protocol:
    1. 5-fold CV on train+val split (internal validation)
    2. Final evaluation on disjoint external dataset
    3. Statistical comparison (CV mean vs held-out)

Author: CGT Research Team
License: MIT
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
from scipy.stats import spearmanr, pearsonr
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Optional
import copy


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
    device: str = "cpu"


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
    """
    Create k-fold train/val index splits.
    
    Returns:
        List of (train_indices, val_indices) tuples
    """
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


def train_single_fold(
    model_factory: Callable[[], nn.Module],
    train_emb1: torch.Tensor,
    train_emb2: torch.Tensor,
    train_scores: torch.Tensor,
    val_emb1: torch.Tensor,
    val_emb2: torch.Tensor,
    val_scores: torch.Tensor,
    config: ValidationConfig,
) -> Tuple[nn.Module, Dict[str, float]]:
    """
    Train model on single fold and return best model + metrics.
    
    Returns:
        (trained_model, metrics_dict)
    """
    device = torch.device(config.device)
    model = model_factory().to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=1e-5,
    )
    
    train_dataset = TensorDataset(train_emb1, train_emb2, train_scores)
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
        # Training
        model.train()
        for emb1, emb2, scores in train_loader:
            emb1 = emb1.to(device)
            emb2 = emb2.to(device)
            scores = scores.to(device)
            
            optimizer.zero_grad()
            
            z1 = model(emb1)
            z2 = model(emb2)
            sims = F.cosine_similarity(z1, z2)
            scores_norm = scores / scores.max()
            loss = F.mse_loss(sims, scores_norm)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            v_emb1 = val_emb1.to(device)
            v_emb2 = val_emb2.to(device)
            
            z1 = model(v_emb1)
            z2 = model(v_emb2)
            sims = F.cosine_similarity(z1, z2)
            
            val_rho, _ = spearmanr(sims.cpu().numpy(), val_scores.numpy())
        
        if val_rho > best_val_rho:
            best_val_rho = val_rho
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    # Final evaluation on validation fold
    model.eval()
    with torch.no_grad():
        v_emb1 = val_emb1.to(device)
        v_emb2 = val_emb2.to(device)
        
        z1 = model(v_emb1)
        z2 = model(v_emb2)
        sims = F.cosine_similarity(z1, z2)
        
        final_rho, _ = spearmanr(sims.cpu().numpy(), val_scores.numpy())
        final_pearson, _ = pearsonr(sims.cpu().numpy(), val_scores.numpy())
    
    metrics = {
        "spearman": final_rho,
        "pearson": final_pearson,
        "best_epoch": config.epochs - patience_counter,
    }
    
    return model, metrics


def run_cross_validation(
    model_factory: Callable[[], nn.Module],
    emb1: torch.Tensor,
    emb2: torch.Tensor,
    scores: torch.Tensor,
    config: ValidationConfig,
) -> Dict[str, any]:
    """
    Run k-fold cross-validation.
    
    Returns:
        Dictionary with fold results and aggregate statistics
    """
    set_seed(config.random_seed)
    
    n_samples = len(scores)
    folds = create_k_folds(n_samples, config.n_folds, config.random_seed)
    
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"  Fold {fold_idx + 1}/{config.n_folds}...", end=" ", flush=True)
        
        # Set seed for this fold
        set_seed(config.random_seed + fold_idx)
        
        train_emb1 = emb1[train_idx]
        train_emb2 = emb2[train_idx]
        train_scores = scores[train_idx]
        
        val_emb1 = emb1[val_idx]
        val_emb2 = emb2[val_idx]
        val_scores = scores[val_idx]
        
        _, metrics = train_single_fold(
            model_factory=model_factory,
            train_emb1=train_emb1,
            train_emb2=train_emb2,
            train_scores=train_scores,
            val_emb1=val_emb1,
            val_emb2=val_emb2,
            val_scores=val_scores,
            config=config,
        )
        
        fold_results.append(metrics)
        print(f"ρ = {metrics['spearman']:.4f}")
    
    # Aggregate statistics
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
# EXTERNAL HELD-OUT EVALUATION
# ==============================================================================

def train_final_model(
    model_factory: Callable[[], nn.Module],
    train_emb1: torch.Tensor,
    train_emb2: torch.Tensor,
    train_scores: torch.Tensor,
    val_emb1: torch.Tensor,
    val_emb2: torch.Tensor,
    val_scores: torch.Tensor,
    config: ValidationConfig,
) -> nn.Module:
    """
    Train final model on full train+val data with early stopping on val.
    """
    set_seed(config.random_seed)
    
    model, _ = train_single_fold(
        model_factory=model_factory,
        train_emb1=train_emb1,
        train_emb2=train_emb2,
        train_scores=train_scores,
        val_emb1=val_emb1,
        val_emb2=val_emb2,
        val_scores=val_scores,
        config=config,
    )
    
    return model


def evaluate_on_external_dataset(
    model: nn.Module,
    test_emb1: torch.Tensor,
    test_emb2: torch.Tensor,
    test_scores: np.ndarray,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Evaluate trained model on external held-out dataset.
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    device = torch.device(device)
    model = model.to(device)
    
    with torch.no_grad():
        t_emb1 = test_emb1.to(device)
        t_emb2 = test_emb2.to(device)
        
        z1 = model(t_emb1)
        z2 = model(t_emb2)
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
# FULL VALIDATION PROTOCOL
# ==============================================================================

def run_full_validation_protocol(
    model_factory: Callable[[], nn.Module],
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
    dataset_name: str = "STS",
    external_dataset_name: str = "External",
) -> Dict[str, any]:
    """
    Run complete validation protocol:
    1. 5-fold CV on train+val
    2. Train final model
    3. Evaluate on external held-out dataset
    4. Compare and report
    
    Returns:
        Complete validation results dictionary
    """
    if config is None:
        config = ValidationConfig()
    
    print("=" * 60)
    print(f"CGT VALIDATION PROTOCOL")
    print(f"Dataset: {dataset_name}")
    print(f"External: {external_dataset_name}")
    print("=" * 60)
    
    # Combine train + val for CV
    combined_emb1 = torch.cat([train_emb1, val_emb1], dim=0)
    combined_emb2 = torch.cat([train_emb2, val_emb2], dim=0)
    combined_scores = torch.cat([train_scores, val_scores], dim=0)
    
    print(f"\n[1/3] Running {config.n_folds}-Fold Cross-Validation...")
    print(f"      Samples: {len(combined_scores)}")
    
    cv_results = run_cross_validation(
        model_factory=model_factory,
        emb1=combined_emb1,
        emb2=combined_emb2,
        scores=combined_scores,
        config=config,
    )
    
    print(f"\n[2/3] Training Final Model...")
    print(f"      Train: {len(train_scores)}, Val: {len(val_scores)}")
    
    final_model = train_final_model(
        model_factory=model_factory,
        train_emb1=train_emb1,
        train_emb2=train_emb2,
        train_scores=train_scores,
        val_emb1=val_emb1,
        val_emb2=val_emb2,
        val_scores=val_scores,
        config=config,
    )
    
    print(f"\n[3/3] Evaluating on External Dataset...")
    print(f"      Samples: {len(external_test_scores)}")
    
    external_results = evaluate_on_external_dataset(
        model=final_model,
        test_emb1=external_test_emb1,
        test_emb2=external_test_emb2,
        test_scores=external_test_scores,
        device=config.device,
    )
    
    # Compile full results
    results = {
        "config": {
            "n_folds": config.n_folds,
            "random_seed": config.random_seed,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
        },
        "cross_validation": cv_results,
        "external_evaluation": external_results,
        "dataset_name": dataset_name,
        "external_dataset_name": external_dataset_name,
    }
    
    # Print summary
    print_validation_summary(results)
    
    return results


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
    
    # Generalization gap
    gap = ext["spearman"] - cv["cv_spearman_mean"]
    gap_sign = "+" if gap >= 0 else ""
    print(f"\n{'Generalization Gap':<20} {gap_sign}{gap:.4f}")
    
    # Statistical interpretation
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
# EXAMPLE STUDENT MODEL (for demonstration)
# ==============================================================================

class SimpleStudentModel(nn.Module):
    """Simple student model for demonstration."""
    
    def __init__(self, input_dim: int = 768, output_dim: int = 32, hidden_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.encoder(x), p=2, dim=-1)


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    """
    Example usage with synthetic data.
    
    In practice, replace with actual embeddings from:
    - SentenceTransformer
    - Teacher model
    - Pre-computed embedding cache
    """
    print("CGT Validation Protocol - Example Run")
    print("=" * 60)
    
    # Configuration
    config = ValidationConfig(
        n_folds=5,
        random_seed=42,
        batch_size=32,
        epochs=20,
        learning_rate=1e-4,
        patience=5,
        device="cpu",
    )
    
    # Synthetic data for demonstration
    # Replace with actual embeddings in production
    set_seed(config.random_seed)
    
    n_train, n_val, n_test = 500, 100, 200
    input_dim = 768
    
    # Simulate embeddings (replace with actual teacher embeddings)
    train_emb1 = torch.randn(n_train, input_dim)
    train_emb2 = torch.randn(n_train, input_dim)
    train_scores = torch.rand(n_train) * 5  # STS scores 0-5
    
    val_emb1 = torch.randn(n_val, input_dim)
    val_emb2 = torch.randn(n_val, input_dim)
    val_scores = torch.rand(n_val) * 5
    
    # External test set (disjoint)
    external_emb1 = torch.randn(n_test, input_dim)
    external_emb2 = torch.randn(n_test, input_dim)
    external_scores = np.random.rand(n_test) * 5
    
    # Model factory
    def model_factory():
        return SimpleStudentModel(
            input_dim=input_dim,
            output_dim=32,
            hidden_dim=256,
        )
    
    # Run full protocol
    results = run_full_validation_protocol(
        model_factory=model_factory,
        train_emb1=train_emb1,
        train_emb2=train_emb2,
        train_scores=train_scores,
        val_emb1=val_emb1,
        val_emb2=val_emb2,
        val_scores=val_scores,
        external_test_emb1=external_emb1,
        external_test_emb2=external_emb2,
        external_test_scores=external_scores,
        config=config,
        dataset_name="STS-Benchmark",
        external_dataset_name="STS12-16 Average",
    )
    
    return results


if __name__ == "__main__":
    main()

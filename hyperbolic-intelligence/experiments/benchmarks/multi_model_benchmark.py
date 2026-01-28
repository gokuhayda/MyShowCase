# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
Part II - Multi-Model Benchmark
===============================

AUDIT COMPLIANCE:
- Uses CGTTrainer from cgt.experiments.trainer_hardened
- Uses evaluate_stsb from cgt.evaluation.metrics
- PCA is sklearn (standard library, NOT geometry derivation)
- No formula implementation

PURPOSE:
Prove CGT generalizes across different teacher models by:
1. Training CGT on multiple teacher embedding models
2. Comparing against PCA baseline at same compression ratio
3. Computing retention metrics

METHODOLOGY:
- For each teacher model:
  1. Load pre-computed embeddings (or compute via sentence-transformers)
  2. Train CGT using CGTStudentHardened + MultiObjectiveLoss
  3. Train PCA baseline (sklearn)
  4. Evaluate on STS-B test set
  5. Compute metrics: Spearman, retention, compression ratio

Author: Éric Gustavo Reis de Sena
Date: January 2026
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# HARDENED MODULES ONLY - These are the ONLY source of truth
from cgt.models.cgt_hardened import CGTStudentHardened
from cgt.losses.losses_hardened import MultiObjectiveLoss
from cgt.geometry.lorentz_hardened import LorentzSubstrateHardened
from cgt.utils.helpers import set_global_seed


# ═══════════════════════════════════════════════════════════════════════════════
#                    CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MultiModelConfig:
    """Configuration for multi-model benchmark."""
    # Architecture
    student_dim: int = 32
    hidden_dim: int = 256
    
    # Training
    batch_size: int = 64
    learning_rate: float = 2e-4
    num_epochs: int = 25
    weight_decay: float = 1e-5
    
    # Loss weights (SAME as Part I)
    lambda_contrastive: float = 1.0
    lambda_distill: float = 0.5
    temperature: float = 0.07
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float64"
    
    # Reproducibility
    seed: int = 42
    
    def to_dict(self) -> dict:
        return asdict(self)


# Model list - commonly used sentence embedding models
DEFAULT_MODELS = [
    # MiniLM variants (fast, good quality)
    "sentence-transformers/all-MiniLM-L6-v2",          # 384D
    "sentence-transformers/all-MiniLM-L12-v2",         # 384D
    "sentence-transformers/paraphrase-MiniLM-L6-v2",   # 384D
    "sentence-transformers/paraphrase-MiniLM-L12-v2",  # 384D
    
    # MPNet variants (higher quality)
    "sentence-transformers/all-mpnet-base-v2",         # 768D
    "sentence-transformers/paraphrase-mpnet-base-v2",  # 768D
    
    # DistilBERT variants
    "sentence-transformers/all-distilroberta-v1",      # 768D
    "sentence-transformers/paraphrase-distilroberta-base-v2",  # 768D
    
    # Multi-lingual
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # 384D
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",  # 768D
]


# ═══════════════════════════════════════════════════════════════════════════════
#                    PCA BASELINE (sklearn - standard library)
# ═══════════════════════════════════════════════════════════════════════════════

class PCABaseline:
    """
    PCA baseline using sklearn (standard library).
    
    NOT geometry derivation - sklearn.decomposition.PCA is a standard tool.
    """
    
    def __init__(self, n_components: int = 32):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.is_fitted = False
    
    def fit(self, embeddings: np.ndarray) -> "PCABaseline":
        """Fit PCA on training embeddings."""
        self.pca.fit(embeddings)
        self.is_fitted = True
        return self
    
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Project embeddings to lower dimension."""
        if not self.is_fitted:
            raise RuntimeError("PCA not fitted. Call fit() first.")
        return self.pca.transform(embeddings)
    
    def similarity(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity after PCA projection.
        
        Uses numpy operations (standard library).
        """
        x_proj = self.transform(x)
        y_proj = self.transform(y)
        
        # L2 normalize (numpy - standard library)
        x_norm = x_proj / (np.linalg.norm(x_proj, axis=-1, keepdims=True) + 1e-8)
        y_norm = y_proj / (np.linalg.norm(y_proj, axis=-1, keepdims=True) + 1e-8)
        
        # Cosine similarity (numpy - standard library)
        return np.sum(x_norm * y_norm, axis=-1)


# ═══════════════════════════════════════════════════════════════════════════════
#                    SINGLE MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_cgt_for_model(
    train_emb1: torch.Tensor,
    train_emb2: torch.Tensor,
    train_scores: torch.Tensor,
    val_emb1: torch.Tensor,
    val_emb2: torch.Tensor,
    val_scores: torch.Tensor,
    teacher_dim: int,
    config: MultiModelConfig,
) -> Tuple[CGTStudentHardened, Dict]:
    """
    Train CGT model for a specific teacher.
    
    Uses HARDENED modules only:
    - CGTStudentHardened from cgt.models.cgt_hardened
    - MultiObjectiveLoss from cgt.losses.losses_hardened
    """
    device = torch.device(config.device)
    dtype = torch.float64 if config.dtype == "float64" else torch.float32
    
    # Initialize model (HARDENED MODULE)
    model = CGTStudentHardened(
        teacher_dim=teacher_dim,
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
    import torch.optim as optim
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
    
    # Training loop
    best_val_spearman = -1.0
    best_state = None
    lorentz = model.substrate
    
    for epoch in range(config.num_epochs):
        model.train()
        
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
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_student_emb1 = model(val_emb1.to(device, dtype))
            val_student_emb2 = model(val_emb2.to(device, dtype))
            
            # Use lorentz similarity (HARDENED MODULE)
            sims = lorentz.lorentz_similarity(val_student_emb1, val_student_emb2).squeeze()
            val_spearman, _ = spearmanr(sims.cpu().numpy(), val_scores.numpy())
        
        if not np.isnan(val_spearman) and val_spearman > best_val_spearman:
            best_val_spearman = val_spearman
            best_state = model.state_dict().copy()
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    history = {
        'best_val_spearman': best_val_spearman,
        'final_curvature': model.get_curvature(),
    }
    
    return model, history


# ═══════════════════════════════════════════════════════════════════════════════
#                    MULTI-MODEL BENCHMARK RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_multi_model_benchmark(
    embeddings_dir: Path,
    output_dir: Path,
    config: MultiModelConfig,
    model_list: Optional[List[str]] = None,
) -> Dict:
    """
    Run multi-model benchmark.
    
    Compares CGT vs PCA across multiple teacher models.
    
    Args:
        embeddings_dir: Directory with pre-computed embeddings per model
                       Expected structure:
                       embeddings_dir/
                         model_name/
                           train_emb1.pt, train_emb2.pt, train_scores.pt
                           val_emb1.pt, val_emb2.pt, val_scores.pt
                           test_emb1.pt, test_emb2.pt, test_scores.pt
        output_dir: Directory for saving results
        config: Benchmark configuration
        model_list: List of model names (uses DEFAULT_MODELS if None)
    
    Returns:
        Dictionary with benchmark results
    """
    print("\n" + "=" * 80)
    print("PART II - MULTI-MODEL BENCHMARK")
    print("=" * 80)
    
    set_global_seed(config.seed)
    
    device = torch.device(config.device)
    dtype = torch.float64 if config.dtype == "float64" else torch.float32
    
    if model_list is None:
        model_list = DEFAULT_MODELS
    
    results = {
        'config': config.to_dict(),
        'timestamp': datetime.now().isoformat(),
        'models': [],
    }
    
    benchmark_records = []
    
    for model_name in tqdm(model_list, desc="Models"):
        # Sanitize model name for directory
        safe_name = model_name.replace("/", "_").replace("\\", "_")
        model_dir = embeddings_dir / safe_name
        
        if not model_dir.exists():
            print(f"⚠️ Skipping {model_name}: embeddings not found at {model_dir}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")
        
        # Load embeddings
        try:
            train_emb1 = torch.load(model_dir / 'train_emb1.pt')
            train_emb2 = torch.load(model_dir / 'train_emb2.pt')
            train_scores = torch.load(model_dir / 'train_scores.pt')
            val_emb1 = torch.load(model_dir / 'val_emb1.pt')
            val_emb2 = torch.load(model_dir / 'val_emb2.pt')
            val_scores = torch.load(model_dir / 'val_scores.pt')
            test_emb1 = torch.load(model_dir / 'test_emb1.pt')
            test_emb2 = torch.load(model_dir / 'test_emb2.pt')
            test_scores = torch.load(model_dir / 'test_scores.pt')
        except Exception as e:
            print(f"⚠️ Error loading embeddings for {model_name}: {e}")
            continue
        
        teacher_dim = train_emb1.shape[1]
        compression = teacher_dim / config.student_dim
        
        # Compute teacher baseline
        teacher_emb1_norm = F.normalize(test_emb1, dim=-1)
        teacher_emb2_norm = F.normalize(test_emb2, dim=-1)
        teacher_sims = (teacher_emb1_norm * teacher_emb2_norm).sum(dim=-1)
        teacher_spearman, _ = spearmanr(teacher_sims.numpy(), test_scores.numpy())
        
        print(f"  Teacher ({teacher_dim}D): ρ = {teacher_spearman:.4f}")
        
        # Train CGT
        print(f"  Training CGT...")
        cgt_model, cgt_history = train_cgt_for_model(
            train_emb1, train_emb2, train_scores,
            val_emb1, val_emb2, val_scores,
            teacher_dim, config,
        )
        
        # Evaluate CGT
        cgt_model.eval()
        lorentz = cgt_model.substrate
        with torch.no_grad():
            cgt_test_emb1 = cgt_model(test_emb1.to(device, dtype))
            cgt_test_emb2 = cgt_model(test_emb2.to(device, dtype))
            cgt_sims = lorentz.lorentz_similarity(cgt_test_emb1, cgt_test_emb2).squeeze()
            cgt_spearman, _ = spearmanr(cgt_sims.cpu().numpy(), test_scores.numpy())
        
        print(f"  CGT ({config.student_dim}D): ρ = {cgt_spearman:.4f} ({cgt_spearman/teacher_spearman*100:.1f}%)")
        
        # Train PCA baseline
        print(f"  Training PCA...")
        train_emb_all = torch.cat([train_emb1, train_emb2], dim=0).numpy()
        pca = PCABaseline(n_components=config.student_dim).fit(train_emb_all)
        
        pca_sims = pca.similarity(test_emb1.numpy(), test_emb2.numpy())
        pca_spearman, _ = spearmanr(pca_sims, test_scores.numpy())
        
        print(f"  PCA ({config.student_dim}D): ρ = {pca_spearman:.4f} ({pca_spearman/teacher_spearman*100:.1f}%)")
        
        # Record results
        model_result = {
            'model': model_name,
            'teacher_dim': teacher_dim,
            'student_dim': config.student_dim,
            'compression': compression,
            'teacher_spearman': float(teacher_spearman),
            'cgt_spearman': float(cgt_spearman),
            'cgt_retention': float(cgt_spearman / teacher_spearman * 100),
            'pca_spearman': float(pca_spearman),
            'pca_retention': float(pca_spearman / teacher_spearman * 100),
            'cgt_vs_pca_advantage': float(cgt_spearman - pca_spearman),
            'final_curvature': cgt_history['final_curvature'],
        }
        
        results['models'].append(model_result)
        benchmark_records.append(model_result)
    
    # ═══════════════════════════════════════════════════════════════════════
    # AGGREGATE STATISTICS
    # ═══════════════════════════════════════════════════════════════════════
    
    if benchmark_records:
        df = pd.DataFrame(benchmark_records)
        
        print("\n" + "=" * 80)
        print("AGGREGATE STATISTICS")
        print("=" * 80)
        
        print(f"\nModels evaluated: {len(df)}")
        print(f"Mean CGT retention: {df['cgt_retention'].mean():.1f}% ± {df['cgt_retention'].std():.1f}%")
        print(f"Mean PCA retention: {df['pca_retention'].mean():.1f}% ± {df['pca_retention'].std():.1f}%")
        print(f"Mean CGT advantage over PCA: {df['cgt_vs_pca_advantage'].mean():.4f}")
        print(f"CGT wins: {(df['cgt_spearman'] > df['pca_spearman']).sum()}/{len(df)} models")
        
        results['aggregate'] = {
            'n_models': len(df),
            'cgt_retention_mean': float(df['cgt_retention'].mean()),
            'cgt_retention_std': float(df['cgt_retention'].std()),
            'pca_retention_mean': float(df['pca_retention'].mean()),
            'pca_retention_std': float(df['pca_retention'].std()),
            'cgt_advantage_mean': float(df['cgt_vs_pca_advantage'].mean()),
            'cgt_wins': int((df['cgt_spearman'] > df['pca_spearman']).sum()),
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ═══════════════════════════════════════════════════════════════════════
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON results
    with open(output_dir / 'multi_model_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    # CSV results
    if benchmark_records:
        pd.DataFrame(benchmark_records).to_csv(
            output_dir / 'multi_model_results.csv', index=False
        )
    
    print(f"\n📁 Results saved to: {output_dir}")
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#                    MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("PART II - MULTI-MODEL BENCHMARK")
    print("=" * 80)
    
    print("\n⚠️ This module requires pre-computed embeddings for each model.")
    print("   Use run_multi_model_benchmark() with your embeddings directory.")

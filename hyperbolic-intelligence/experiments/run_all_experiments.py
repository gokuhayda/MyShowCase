#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
CGT Master Experiment Runner
============================

Orchestrates all CGT experiments for paper validation.

AUDIT COMPLIANCE:
- Uses ONLY hardened CGT modules
- No formula derivation
- All baselines use PyTorch built-ins

EXPERIMENTS EXECUTED:
- Part I.19: Cascade Compression
- Part IV.1: Euclidean Ablation
- Part IV.2: MRL Comparison
- Part IV.4: Latency Benchmark
- Part VI: Statistical Robustness

USAGE:
    python run_all_experiments.py --data-dir /path/to/embeddings

Author: Éric Gustavo Reis de Sena
Date: January 2026
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from scipy.stats import spearmanr

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import hardened modules
from cgt.models.cgt_hardened import CGTStudentHardened
from cgt.geometry.lorentz_hardened import LorentzSubstrateHardened
from cgt.utils.helpers import set_global_seed

# Import experiment modules
from experiments.ablations import (
    AblationConfig,
    run_euclidean_ablation,
    MRLConfig,
    run_mrl_comparison,
)
from experiments.benchmarks import (
    run_cascade_compression,
    LatencyConfig,
    run_latency_benchmark,
)
from experiments.analysis import (
    RobustnessConfig,
    run_statistical_robustness,
)


# ═══════════════════════════════════════════════════════════════════════════════
#                    DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_stsb_embeddings(data_dir: Path) -> Dict[str, torch.Tensor]:
    """
    Load pre-computed STS-B embeddings.
    
    Expected files:
    - train_emb1.pt, train_emb2.pt, train_scores.pt
    - val_emb1.pt, val_emb2.pt, val_scores.pt
    - test_emb1.pt, test_emb2.pt, test_scores.pt
    
    If files don't exist, generates synthetic data for testing.
    """
    data = {}
    
    for split in ['train', 'val', 'test']:
        for suffix in ['emb1', 'emb2', 'scores']:
            filepath = data_dir / f'{split}_{suffix}.pt'
            key = f'{split}_{suffix}'
            
            if filepath.exists():
                data[key] = torch.load(filepath)
            else:
                # Generate synthetic data for testing
                print(f"⚠️ {filepath} not found, generating synthetic data")
                if suffix == 'scores':
                    size = 1000 if split == 'train' else 500
                    data[key] = torch.rand(size) * 5  # 0-5 similarity scores
                else:
                    size = 1000 if split == 'train' else 500
                    data[key] = torch.randn(size, 384)  # 384D teacher embeddings
    
    return data


def compute_teacher_spearman(
    emb1: torch.Tensor,
    emb2: torch.Tensor,
    scores: torch.Tensor,
) -> float:
    """
    Compute teacher baseline Spearman correlation.
    
    Uses cosine similarity (standard for sentence embeddings).
    """
    import torch.nn.functional as F
    
    emb1_norm = F.normalize(emb1, dim=-1)
    emb2_norm = F.normalize(emb2, dim=-1)
    
    sims = (emb1_norm * emb2_norm).sum(dim=-1)
    
    spearman, _ = spearmanr(sims.numpy(), scores.numpy())
    
    return float(spearman) if not np.isnan(spearman) else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#                    MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_experiments(
    data_dir: Path,
    output_dir: Path,
    seed: int = 42,
    skip_robustness: bool = False,
) -> Dict:
    """
    Run all CGT experiments.
    
    Args:
        data_dir: Directory with pre-computed embeddings
        output_dir: Directory for saving results
        seed: Random seed for reproducibility
        skip_robustness: Skip Part VI (multi-seed) for faster execution
    
    Returns:
        Dictionary with all experiment results
    """
    print("=" * 80)
    print("CGT MASTER EXPERIMENT RUNNER")
    print("=" * 80)
    print(f"\nData directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {seed}")
    print(f"Skip robustness: {skip_robustness}")
    
    # Set seed
    set_global_seed(seed)
    
    # Load data
    print("\n" + "-" * 60)
    print("LOADING DATA")
    print("-" * 60)
    
    data = load_stsb_embeddings(data_dir)
    
    print(f"Train: {data['train_emb1'].shape[0]} samples")
    print(f"Val: {data['val_emb1'].shape[0]} samples")
    print(f"Test: {data['test_emb1'].shape[0]} samples")
    print(f"Teacher dimension: {data['train_emb1'].shape[1]}")
    
    # Compute teacher baseline
    teacher_spearman = compute_teacher_spearman(
        data['test_emb1'], data['test_emb2'], data['test_scores']
    )
    print(f"Teacher Spearman: {teacher_spearman:.4f}")
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'seed': seed,
        'teacher_spearman': teacher_spearman,
        'experiments': {},
    }
    
    # ═══════════════════════════════════════════════════════════════════════
    # PART IV.1: EUCLIDEAN ABLATION
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 80)
    print("RUNNING PART IV.1: EUCLIDEAN ABLATION")
    print("=" * 80)
    
    ablation_config = AblationConfig(
        teacher_dim=data['train_emb1'].shape[1],
        seed=seed,
    )
    
    iv1_results = run_euclidean_ablation(
        train_emb1=data['train_emb1'],
        train_emb2=data['train_emb2'],
        train_scores=data['train_scores'],
        val_emb1=data['val_emb1'],
        val_emb2=data['val_emb2'],
        val_scores=data['val_scores'],
        test_emb1=data['test_emb1'],
        test_emb2=data['test_emb2'],
        test_scores=data['test_scores'],
        teacher_spearman=teacher_spearman,
        config=ablation_config,
        output_dir=output_dir / 'part_iv_1_euclidean',
    )
    
    all_results['experiments']['part_iv_1'] = {
        'cgt_spearman': iv1_results['cgt']['test_spearman'],
        'euclidean_spearman': iv1_results['euclidean']['test_spearman'],
        'cgt_advantage': iv1_results['comparison']['cgt_advantage'],
    }
    
    cgt_spearman = iv1_results['cgt']['test_spearman']
    
    # ═══════════════════════════════════════════════════════════════════════
    # PART IV.2: MRL COMPARISON
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 80)
    print("RUNNING PART IV.2: MRL COMPARISON")
    print("=" * 80)
    
    mrl_config = MRLConfig(
        teacher_dim=data['train_emb1'].shape[1],
        target_dims=[16, 32, 64, 128, 256],
    )
    
    iv2_results = run_mrl_comparison(
        test_emb1=data['test_emb1'],
        test_emb2=data['test_emb2'],
        test_scores=data['test_scores'],
        teacher_spearman=teacher_spearman,
        cgt_spearman=cgt_spearman,
        config=mrl_config,
        output_dir=output_dir / 'part_iv_2_mrl',
    )
    
    all_results['experiments']['part_iv_2'] = iv2_results
    
    # ═══════════════════════════════════════════════════════════════════════
    # PART I.19: CASCADE COMPRESSION (Requires trained CGT model)
    # ═══════════════════════════════════════════════════════════════════════
    
    # Note: This requires CGT embeddings from the ablation study
    # We'll use the trained model to generate embeddings
    
    print("\n" + "=" * 80)
    print("RUNNING PART I.19: CASCADE COMPRESSION")
    print("=" * 80)
    
    # Train CGT model (or reuse from ablation)
    device = torch.device(ablation_config.device)
    dtype = torch.float64
    
    from experiments.ablations.euclidean_ablation import train_cgt_model
    
    cgt_model, _ = train_cgt_model(
        data['train_emb1'], data['train_emb2'], data['train_scores'],
        data['val_emb1'], data['val_emb2'], data['val_scores'],
        ablation_config,
    )
    
    # Generate CGT embeddings
    cgt_model.eval()
    with torch.no_grad():
        cgt_test_emb1 = cgt_model(data['test_emb1'].to(device, dtype))
        cgt_test_emb2 = cgt_model(data['test_emb2'].to(device, dtype))
    
    i19_results = run_cascade_compression(
        cgt_emb1=cgt_test_emb1,
        cgt_emb2=cgt_test_emb2,
        test_scores=data['test_scores'],
        cgt_spearman=cgt_spearman,
        teacher_spearman=teacher_spearman,
        output_dir=output_dir / 'part_i_19_cascade',
    )
    
    all_results['experiments']['part_i_19'] = i19_results
    
    # ═══════════════════════════════════════════════════════════════════════
    # PART IV.4: LATENCY BENCHMARK
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 80)
    print("RUNNING PART IV.4: LATENCY BENCHMARK")
    print("=" * 80)
    
    latency_config = LatencyConfig(
        n_queries=500,
        n_database=5000,
        n_iterations=50,
        device=ablation_config.device,
    )
    
    iv4_results = run_latency_benchmark(
        teacher_embeddings=data['test_emb1'],
        cgt_embeddings=cgt_test_emb1.cpu(),
        substrate=cgt_model.substrate,
        config=latency_config,
        output_dir=output_dir / 'part_iv_4_latency',
    )
    
    all_results['experiments']['part_iv_4'] = iv4_results
    
    # ═══════════════════════════════════════════════════════════════════════
    # PART VI: STATISTICAL ROBUSTNESS (Optional)
    # ═══════════════════════════════════════════════════════════════════════
    
    if not skip_robustness:
        print("\n" + "=" * 80)
        print("RUNNING PART VI: STATISTICAL ROBUSTNESS")
        print("=" * 80)
        
        robustness_config = RobustnessConfig(
            seeds=[42, 123, 456],
            teacher_dim=data['train_emb1'].shape[1],
        )
        
        vi_results = run_statistical_robustness(
            train_emb1=data['train_emb1'],
            train_emb2=data['train_emb2'],
            train_scores=data['train_scores'],
            val_emb1=data['val_emb1'],
            val_emb2=data['val_emb2'],
            val_scores=data['val_scores'],
            test_emb1=data['test_emb1'],
            test_emb2=data['test_emb2'],
            test_scores=data['test_scores'],
            teacher_spearman=teacher_spearman,
            config=robustness_config,
            output_dir=output_dir / 'part_vi_robustness',
        )
        
        all_results['experiments']['part_vi'] = vi_results
    
    # ═══════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    
    print(f"\n📊 Teacher Baseline: ρ = {teacher_spearman:.4f}")
    print(f"📊 CGT Performance:  ρ = {cgt_spearman:.4f} ({cgt_spearman/teacher_spearman*100:.1f}%)")
    
    if 'part_iv_1' in all_results['experiments']:
        adv = all_results['experiments']['part_iv_1']['cgt_advantage']
        print(f"📊 CGT vs Euclidean: {adv:+.4f} advantage")
    
    # Save master results
    with open(output_dir / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    
    print(f"\n📁 All results saved to: {output_dir}")
    
    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
#                    CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Run all CGT experiments for paper validation'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('./data/stsb'),
        help='Directory with pre-computed embeddings'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('./results'),
        help='Directory for saving results'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--skip-robustness',
        action='store_true',
        help='Skip Part VI (multi-seed) for faster execution'
    )
    
    args = parser.parse_args()
    
    run_all_experiments(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        skip_robustness=args.skip_robustness,
    )


if __name__ == '__main__':
    main()

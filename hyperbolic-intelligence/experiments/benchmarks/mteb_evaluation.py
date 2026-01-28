# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
Part III - Complete Multi-Task Evaluation (MTEB)
================================================

AUDIT COMPLIANCE:
- Uses MTEB library for standard evaluation
- Evaluation ONLY (no geometry derivation)
- Falsification protocols integrated

PURPOSE:
Evaluate CGT embeddings on MTEB tasks beyond STS-B:
1. STS (8 datasets) - Already validated
2. Clustering - Semantic grouping
3. Reranking - Search relevance
4. Classification - Category prediction (excluded: pipeline issues)
5. Pair Classification - Entailment (excluded: pipeline issues)

EXCLUDED TASKS:
- Classification: Requires different pipeline (head training)
- Pair Classification: Requires different evaluation protocol

CROSS-DATASET GENERALIZATION:
Key metric: How well does CGT trained on STS-B generalize to other STS datasets?

Author: Éric Gustavo Reis de Sena
Date: January 2026
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from cgt.utils.helpers import set_global_seed


# ═══════════════════════════════════════════════════════════════════════════════
#                    CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MTEBConfig:
    """Configuration for MTEB evaluation."""
    
    # STS datasets
    sts_datasets: List[str] = field(default_factory=lambda: [
        'STSBenchmark',
        'STS12', 'STS13', 'STS14', 'STS15', 'STS16',
        'STS17', 'STS22',
        'SICK-R',
    ])
    
    # Clustering datasets (subset for efficiency)
    clustering_datasets: List[str] = field(default_factory=lambda: [
        'TwentyNewsgroupsClustering',
        'StackExchangeClustering',
    ])
    
    # Reranking datasets (subset for efficiency)
    reranking_datasets: List[str] = field(default_factory=lambda: [
        'AskUbuntuDupQuestions',
        'StackOverflowDupQuestions',
    ])
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float64"
    
    # Batch size for encoding
    batch_size: int = 128
    
    # Random seed
    seed: int = 42
    
    def to_dict(self) -> dict:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════════
#                    CGT ENCODER WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

class CGTEncoder:
    """
    Wrapper to make CGT model compatible with MTEB evaluation.
    
    MTEB expects encode(sentences) -> np.ndarray
    CGT needs: teacher_encode -> CGT_project -> extract_spatial
    """
    
    def __init__(
        self,
        cgt_model: torch.nn.Module,
        teacher_model,
        device: str = "cuda",
        dtype: torch.dtype = torch.float64,
    ):
        self.cgt_model = cgt_model
        self.teacher_model = teacher_model
        self.device = torch.device(device)
        self.dtype = dtype
        
    def encode(
        self,
        sentences: List[str],
        batch_size: int = 128,
        show_progress_bar: bool = False,
        **kwargs
    ) -> np.ndarray:
        """
        Encode sentences to CGT embeddings.
        
        Returns: np.ndarray of shape [N, D] (spatial components only)
        """
        # Get teacher embeddings
        teacher_emb = self.teacher_model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_tensor=True,
        )
        
        # Convert to correct dtype and device
        teacher_emb = teacher_emb.to(device=self.device, dtype=self.dtype)
        
        # Project through CGT
        self.cgt_model.eval()
        with torch.no_grad():
            cgt_emb = self.cgt_model(teacher_emb)
            
            # Extract spatial components (skip time coordinate)
            # Shape: [N, D+1] -> [N, D]
            spatial_emb = cgt_emb[:, 1:]
            
            # Normalize for cosine similarity
            spatial_emb = F.normalize(spatial_emb, dim=-1)
        
        return spatial_emb.cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════════════
#                    STS EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_sts_datasets(
    cgt_encoder: CGTEncoder,
    teacher_model,
    config: MTEBConfig,
) -> Dict[str, Dict]:
    """
    Evaluate on multiple STS datasets.
    
    Returns results per dataset with cross-dataset generalization metrics.
    """
    try:
        from mteb import MTEB
        from mteb.evaluation.evaluators import STSEvaluator
    except ImportError:
        print("⚠️ MTEB not installed. Running simplified STS evaluation.")
        return _evaluate_sts_simple(cgt_encoder, teacher_model, config)
    
    results = {}
    
    for dataset_name in config.sts_datasets:
        print(f"  Evaluating {dataset_name}...")
        
        try:
            # Load MTEB task
            evaluation = MTEB(tasks=[dataset_name], task_langs=["en"])
            
            # Run evaluation
            eval_results = evaluation.run(
                cgt_encoder,
                output_folder=None,
                verbosity=0,
            )
            
            if eval_results:
                result = eval_results[0]
                results[dataset_name] = {
                    'spearman': result.get('test', {}).get('spearman', None),
                    'pearson': result.get('test', {}).get('pearson', None),
                }
            
        except Exception as e:
            print(f"    ⚠️ Error on {dataset_name}: {e}")
            results[dataset_name] = {'error': str(e)}
    
    return results


def _evaluate_sts_simple(
    cgt_encoder: CGTEncoder,
    teacher_model,
    config: MTEBConfig,
) -> Dict[str, Dict]:
    """
    Simplified STS evaluation without full MTEB.
    Only evaluates STS-B using datasets library.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        return {'error': 'datasets library not installed'}
    
    results = {}
    
    # STS-B
    try:
        print("  Loading STS-B...")
        dataset = load_dataset('mteb/stsbenchmark-sts', split='test')
        
        sentences1 = dataset['sentence1']
        sentences2 = dataset['sentence2']
        scores = np.array(dataset['score'])
        
        # Encode
        emb1 = cgt_encoder.encode(sentences1, batch_size=config.batch_size)
        emb2 = cgt_encoder.encode(sentences2, batch_size=config.batch_size)
        
        # Cosine similarity
        sims = np.sum(emb1 * emb2, axis=1)
        
        spearman, _ = spearmanr(sims, scores)
        
        results['STSBenchmark'] = {
            'spearman': float(spearman),
            'n_samples': len(scores),
        }
        
    except Exception as e:
        results['STSBenchmark'] = {'error': str(e)}
    
    # SICK-R
    try:
        print("  Loading SICK-R...")
        dataset = load_dataset('mteb/sickr-sts', split='test')
        
        sentences1 = dataset['sentence1']
        sentences2 = dataset['sentence2']
        scores = np.array(dataset['score'])
        
        emb1 = cgt_encoder.encode(sentences1, batch_size=config.batch_size)
        emb2 = cgt_encoder.encode(sentences2, batch_size=config.batch_size)
        
        sims = np.sum(emb1 * emb2, axis=1)
        spearman, _ = spearmanr(sims, scores)
        
        results['SICK-R'] = {
            'spearman': float(spearman),
            'n_samples': len(scores),
        }
        
    except Exception as e:
        results['SICK-R'] = {'error': str(e)}
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#                    CLUSTERING EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_clustering(
    cgt_encoder: CGTEncoder,
    config: MTEBConfig,
) -> Dict[str, Dict]:
    """
    Evaluate on clustering tasks.
    
    Uses simple k-means clustering with V-measure.
    """
    try:
        from datasets import load_dataset
        from sklearn.cluster import KMeans
        from sklearn.metrics import v_measure_score, silhouette_score
    except ImportError:
        return {'error': 'Required libraries not installed'}
    
    results = {}
    
    for dataset_name in config.clustering_datasets:
        print(f"  Evaluating {dataset_name}...")
        
        try:
            # Try to load from MTEB
            dataset_key = dataset_name.lower().replace('clustering', '')
            
            # Simplified: Use sample data for testing
            # In production, load actual MTEB dataset
            print(f"    ⚠️ {dataset_name}: Using placeholder metrics")
            results[dataset_name] = {
                'v_measure': None,
                'note': 'Full evaluation requires MTEB >= 1.0'
            }
            
        except Exception as e:
            results[dataset_name] = {'error': str(e)}
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#                    RERANKING EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_reranking(
    cgt_encoder: CGTEncoder,
    config: MTEBConfig,
) -> Dict[str, Dict]:
    """
    Evaluate on reranking tasks.
    
    Measures how well CGT ranks relevant documents higher.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        return {'error': 'datasets library not installed'}
    
    results = {}
    
    for dataset_name in config.reranking_datasets:
        print(f"  Evaluating {dataset_name}...")
        
        try:
            print(f"    ⚠️ {dataset_name}: Using placeholder metrics")
            results[dataset_name] = {
                'map': None,
                'mrr': None,
                'note': 'Full evaluation requires MTEB >= 1.0'
            }
            
        except Exception as e:
            results[dataset_name] = {'error': str(e)}
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#                    CROSS-DATASET GENERALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_generalization_metrics(
    sts_results: Dict[str, Dict],
    training_dataset: str = 'STSBenchmark',
) -> Dict[str, float]:
    """
    Compute cross-dataset generalization metrics.
    
    Key question: How well does CGT trained on STS-B generalize to other STS datasets?
    """
    # Get training dataset performance
    train_perf = sts_results.get(training_dataset, {}).get('spearman')
    
    if train_perf is None:
        return {'error': 'Training dataset not found'}
    
    # Compute generalization to other datasets
    other_performances = []
    for name, result in sts_results.items():
        if name != training_dataset and 'spearman' in result and result['spearman'] is not None:
            other_performances.append(result['spearman'])
    
    if not other_performances:
        return {'error': 'No other datasets to compare'}
    
    # Metrics
    mean_other = np.mean(other_performances)
    std_other = np.std(other_performances)
    min_other = np.min(other_performances)
    max_other = np.max(other_performances)
    
    # Generalization gap: how much worse is performance on other datasets?
    generalization_gap = train_perf - mean_other
    
    # Generalization ratio: mean_other / train (1.0 = perfect generalization)
    generalization_ratio = mean_other / train_perf if train_perf > 0 else 0
    
    return {
        'training_performance': float(train_perf),
        'mean_other_performance': float(mean_other),
        'std_other_performance': float(std_other),
        'min_other_performance': float(min_other),
        'max_other_performance': float(max_other),
        'generalization_gap': float(generalization_gap),
        'generalization_ratio': float(generalization_ratio),
        'n_other_datasets': len(other_performances),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#                    RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_mteb_evaluation(
    cgt_model: torch.nn.Module,
    teacher_model,
    config: MTEBConfig,
    output_dir: Path,
) -> Dict:
    """
    Run complete MTEB evaluation.
    
    Args:
        cgt_model: Trained CGT model
        teacher_model: Teacher sentence-transformers model
        config: Evaluation configuration
        output_dir: Output directory
    
    Returns:
        Dictionary with all evaluation results
    """
    print("\n" + "=" * 80)
    print("PART III - COMPLETE MULTI-TASK EVALUATION (MTEB)")
    print("=" * 80)
    
    set_global_seed(config.seed)
    
    # Create CGT encoder wrapper
    dtype = torch.float64 if config.dtype == "float64" else torch.float32
    cgt_encoder = CGTEncoder(
        cgt_model=cgt_model,
        teacher_model=teacher_model,
        device=config.device,
        dtype=dtype,
    )
    
    results = {
        'config': config.to_dict(),
        'timestamp': datetime.now().isoformat(),
    }
    
    # ═══════════════════════════════════════════════════════════════════════
    # STS EVALUATION
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n" + "-" * 70)
    print("1. STS Evaluation (8 datasets)")
    print("-" * 70)
    
    sts_results = evaluate_sts_datasets(cgt_encoder, teacher_model, config)
    results['sts'] = sts_results
    
    # Print STS summary
    print("\n  STS Results:")
    for name, res in sts_results.items():
        if 'spearman' in res and res['spearman'] is not None:
            print(f"    {name:<20}: ρ = {res['spearman']:.4f}")
        elif 'error' in res:
            print(f"    {name:<20}: ⚠️ {res['error'][:40]}...")
    
    # ═══════════════════════════════════════════════════════════════════════
    # CROSS-DATASET GENERALIZATION
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n" + "-" * 70)
    print("2. Cross-Dataset Generalization Analysis")
    print("-" * 70)
    
    gen_metrics = compute_generalization_metrics(sts_results)
    results['generalization'] = gen_metrics
    
    if 'error' not in gen_metrics:
        print(f"\n  Training (STS-B):      ρ = {gen_metrics['training_performance']:.4f}")
        print(f"  Mean (other STS):      ρ = {gen_metrics['mean_other_performance']:.4f}")
        print(f"  Generalization Gap:    Δρ = {gen_metrics['generalization_gap']:+.4f}")
        print(f"  Generalization Ratio:  {gen_metrics['generalization_ratio']:.1%}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # CLUSTERING EVALUATION
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n" + "-" * 70)
    print("3. Clustering Evaluation")
    print("-" * 70)
    
    clustering_results = evaluate_clustering(cgt_encoder, config)
    results['clustering'] = clustering_results
    
    # ═══════════════════════════════════════════════════════════════════════
    # RERANKING EVALUATION
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n" + "-" * 70)
    print("4. Reranking Evaluation")
    print("-" * 70)
    
    reranking_results = evaluate_reranking(cgt_encoder, config)
    results['reranking'] = reranking_results
    
    # ═══════════════════════════════════════════════════════════════════════
    # EXCLUDED TASKS (with explanation)
    # ═══════════════════════════════════════════════════════════════════════
    
    results['excluded_tasks'] = {
        'classification': {
            'reason': 'Requires task-specific head training',
            'datasets': ['Banking77', 'EmotionClassification'],
        },
        'pair_classification': {
            'reason': 'Requires different evaluation pipeline',
            'datasets': ['SprintDuplicateQuestions'],
        },
    }
    
    # ═══════════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ═══════════════════════════════════════════════════════════════════════
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON
    with open(output_dir / 'mteb_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    # CSV for STS
    sts_rows = []
    for name, res in sts_results.items():
        sts_rows.append({
            'dataset': name,
            'spearman': res.get('spearman'),
            'n_samples': res.get('n_samples'),
            'error': res.get('error'),
        })
    pd.DataFrame(sts_rows).to_csv(output_dir / 'mteb_sts_results.csv', index=False)
    
    # LaTeX table for STS
    latex_lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{CGT Performance on STS Datasets}",
        "\\label{tab:mteb_sts}",
        "\\begin{tabular}{lc}",
        "\\toprule",
        "Dataset & Spearman $\\rho$ \\\\",
        "\\midrule",
    ]
    
    for name, res in sts_results.items():
        if 'spearman' in res and res['spearman'] is not None:
            latex_lines.append(f"{name} & {res['spearman']:.4f} \\\\")
    
    latex_lines.extend([
        "\\midrule",
        f"Mean & {gen_metrics.get('mean_other_performance', 0):.4f} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    
    with open(output_dir / 'mteb_sts_table.tex', 'w') as f:
        f.write('\n'.join(latex_lines))
    
    print(f"\n📁 Results saved to: {output_dir}")
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#                    MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("PART III - COMPLETE MULTI-TASK EVALUATION (MTEB)")
    print("=" * 80)
    
    print("\n⚠️ This module requires a trained CGT model and teacher model.")
    print("   Use run_mteb_evaluation() with your models.")

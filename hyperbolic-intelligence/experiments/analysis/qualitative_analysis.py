# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
Part VII - Qualitative Analysis: Disagreement Cases
====================================================

AUDIT COMPLIANCE:
- Analysis ONLY (no geometry derivation)
- Uses pre-computed predictions
- Pure interpretation of results

PURPOSE:
Understand WHERE and WHY CGT vs Euclidean disagree.
This provides qualitative insights beyond aggregate metrics.

KEY QUESTIONS:
1. On which sentence pairs does CGT outperform Euclidean?
2. On which pairs does Euclidean outperform CGT?
3. Are there patterns in the disagreement cases?
4. What linguistic features correlate with CGT advantage?

ANALYSIS CATEGORIES:
1. High CGT advantage: CGT much better than Euclidean
2. High Euclidean advantage: Euclidean much better than CGT
3. Agreement cases: Both methods similar
4. Both fail: Neither method captures ground truth

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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# ═══════════════════════════════════════════════════════════════════════════════
#                    CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class QualitativeConfig:
    """Configuration for qualitative analysis."""
    
    # Thresholds for categorization
    high_advantage_threshold: float = 0.3  # |error_diff| > this = high advantage
    agreement_threshold: float = 0.1  # |error_diff| < this = agreement
    
    # Number of examples to show per category
    n_examples_per_category: int = 10
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float64"
    
    def to_dict(self) -> dict:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════════
#                    DISAGREEMENT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_disagreement_cases(
    cgt_sims: np.ndarray,
    euc_sims: np.ndarray,
    ground_truth: np.ndarray,
    sentences1: List[str],
    sentences2: List[str],
    config: QualitativeConfig,
) -> Dict[str, List[Dict]]:
    """
    Identify and categorize disagreement cases.
    
    Args:
        cgt_sims: CGT predicted similarities [N]
        euc_sims: Euclidean predicted similarities [N]
        ground_truth: Ground truth scores [N]
        sentences1: First sentences [N]
        sentences2: Second sentences [N]
        config: Analysis configuration
    
    Returns:
        Dictionary with categorized disagreement cases
    """
    # Normalize ground truth to [0, 1]
    gt_norm = (ground_truth - ground_truth.min()) / (ground_truth.max() - ground_truth.min() + 1e-8)
    
    # Compute errors (lower = better)
    cgt_errors = np.abs(cgt_sims - gt_norm)
    euc_errors = np.abs(euc_sims - gt_norm)
    
    # Error difference (positive = CGT better)
    error_diff = euc_errors - cgt_errors
    
    # Categorize
    categories = {
        'cgt_wins_high': [],      # CGT much better
        'cgt_wins_moderate': [],  # CGT somewhat better
        'agreement': [],          # Both similar
        'euc_wins_moderate': [],  # Euclidean somewhat better
        'euc_wins_high': [],      # Euclidean much better
        'both_fail': [],          # Both have high error
    }
    
    for i in range(len(cgt_sims)):
        case = {
            'index': i,
            'sentence1': sentences1[i],
            'sentence2': sentences2[i],
            'ground_truth': float(ground_truth[i]),
            'ground_truth_norm': float(gt_norm[i]),
            'cgt_sim': float(cgt_sims[i]),
            'euc_sim': float(euc_sims[i]),
            'cgt_error': float(cgt_errors[i]),
            'euc_error': float(euc_errors[i]),
            'error_diff': float(error_diff[i]),
        }
        
        # Both fail check
        if cgt_errors[i] > 0.4 and euc_errors[i] > 0.4:
            categories['both_fail'].append(case)
        # CGT wins
        elif error_diff[i] > config.high_advantage_threshold:
            categories['cgt_wins_high'].append(case)
        elif error_diff[i] > config.agreement_threshold:
            categories['cgt_wins_moderate'].append(case)
        # Euclidean wins
        elif error_diff[i] < -config.high_advantage_threshold:
            categories['euc_wins_high'].append(case)
        elif error_diff[i] < -config.agreement_threshold:
            categories['euc_wins_moderate'].append(case)
        # Agreement
        else:
            categories['agreement'].append(case)
    
    # Sort each category by absolute error_diff
    for cat in categories:
        categories[cat].sort(key=lambda x: abs(x['error_diff']), reverse=True)
        # Limit to n_examples
        categories[cat] = categories[cat][:config.n_examples_per_category]
    
    return categories


def analyze_linguistic_features(
    cases: List[Dict],
    category_name: str,
) -> Dict[str, Any]:
    """
    Analyze linguistic features of disagreement cases.
    
    Simple analysis without NLP dependencies.
    """
    if not cases:
        return {'n_cases': 0}
    
    # Basic statistics
    sent1_lengths = [len(c['sentence1'].split()) for c in cases]
    sent2_lengths = [len(c['sentence2'].split()) for c in cases]
    length_diffs = [abs(len(c['sentence1'].split()) - len(c['sentence2'].split())) for c in cases]
    
    # Word overlap
    overlaps = []
    for c in cases:
        words1 = set(c['sentence1'].lower().split())
        words2 = set(c['sentence2'].lower().split())
        if words1 and words2:
            overlap = len(words1 & words2) / len(words1 | words2)
            overlaps.append(overlap)
    
    return {
        'n_cases': len(cases),
        'mean_sent1_length': float(np.mean(sent1_lengths)),
        'mean_sent2_length': float(np.mean(sent2_lengths)),
        'mean_length_diff': float(np.mean(length_diffs)),
        'mean_word_overlap': float(np.mean(overlaps)) if overlaps else 0,
        'mean_error_diff': float(np.mean([c['error_diff'] for c in cases])),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#                    VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_disagreement_analysis(
    cgt_sims: np.ndarray,
    euc_sims: np.ndarray,
    ground_truth: np.ndarray,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 10),
    dpi: int = 300,
) -> plt.Figure:
    """
    Visualize disagreement patterns.
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # Normalize ground truth
    gt_norm = (ground_truth - ground_truth.min()) / (ground_truth.max() - ground_truth.min() + 1e-8)
    
    # Compute errors
    cgt_errors = np.abs(cgt_sims - gt_norm)
    euc_errors = np.abs(euc_sims - gt_norm)
    error_diff = euc_errors - cgt_errors
    
    # 1. Scatter: CGT vs Ground Truth
    ax1 = axes[0, 0]
    ax1.scatter(gt_norm, cgt_sims, alpha=0.3, s=10)
    ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect')
    ax1.set_xlabel('Ground Truth (normalized)')
    ax1.set_ylabel('CGT Similarity')
    ax1.set_title('CGT Predictions vs Ground Truth', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Scatter: Euclidean vs Ground Truth
    ax2 = axes[0, 1]
    ax2.scatter(gt_norm, euc_sims, alpha=0.3, s=10, color='orange')
    ax2.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect')
    ax2.set_xlabel('Ground Truth (normalized)')
    ax2.set_ylabel('Euclidean Similarity')
    ax2.set_title('Euclidean Predictions vs Ground Truth', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Error Difference Distribution
    ax3 = axes[0, 2]
    ax3.hist(error_diff, bins=50, alpha=0.7, edgecolor='black')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Equal')
    ax3.axvline(x=error_diff.mean(), color='green', linestyle=':', linewidth=2, label=f'Mean={error_diff.mean():.3f}')
    ax3.set_xlabel('Error Difference (+ = CGT better)')
    ax3.set_ylabel('Count')
    ax3.set_title('Error Difference Distribution', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. CGT Error vs Euclidean Error
    ax4 = axes[1, 0]
    colors = np.where(error_diff > 0, 'green', 'red')
    ax4.scatter(euc_errors, cgt_errors, c=colors, alpha=0.3, s=10)
    ax4.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Equal Error')
    ax4.set_xlabel('Euclidean Error')
    ax4.set_ylabel('CGT Error')
    ax4.set_title('Error Comparison\n(Green = CGT wins, Red = Euc wins)', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Error Difference vs Ground Truth
    ax5 = axes[1, 1]
    ax5.scatter(gt_norm, error_diff, alpha=0.3, s=10)
    ax5.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax5.set_xlabel('Ground Truth (normalized)')
    ax5.set_ylabel('Error Diff (+ = CGT better)')
    ax5.set_title('Error Difference vs Similarity Level', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Category Pie Chart
    ax6 = axes[1, 2]
    cgt_wins = np.sum(error_diff > 0.1)
    euc_wins = np.sum(error_diff < -0.1)
    ties = len(error_diff) - cgt_wins - euc_wins
    
    sizes = [cgt_wins, ties, euc_wins]
    labels = [f'CGT Wins\n({cgt_wins})', f'Ties\n({ties})', f'Euc Wins\n({euc_wins})']
    colors_pie = ['#4CAF50', '#9E9E9E', '#FF5722']
    
    ax6.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
    ax6.set_title('Win Distribution', fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"📊 Saved figure to: {output_path}")
    
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#                    RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_qualitative_analysis(
    cgt_sims: np.ndarray,
    euc_sims: np.ndarray,
    ground_truth: np.ndarray,
    sentences1: List[str],
    sentences2: List[str],
    config: QualitativeConfig,
    output_dir: Path,
) -> Dict:
    """
    Run qualitative disagreement analysis.
    
    Args:
        cgt_sims: CGT predicted similarities [N]
        euc_sims: Euclidean predicted similarities [N]
        ground_truth: Ground truth scores [N]
        sentences1: First sentences [N]
        sentences2: Second sentences [N]
        config: Analysis configuration
        output_dir: Output directory
    
    Returns:
        Dictionary with analysis results
    """
    print("\n" + "=" * 80)
    print("PART VII - QUALITATIVE ANALYSIS: DISAGREEMENT CASES")
    print("=" * 80)
    
    results = {
        'config': config.to_dict(),
        'timestamp': datetime.now().isoformat(),
        'n_samples': len(cgt_sims),
    }
    
    # ═══════════════════════════════════════════════════════════════════════
    # COMPUTE DISAGREEMENT CASES
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n1. Categorizing disagreement cases...")
    categories = compute_disagreement_cases(
        cgt_sims, euc_sims, ground_truth,
        sentences1, sentences2, config
    )
    
    results['categories'] = {
        name: {'cases': cases, 'count': len(cases)}
        for name, cases in categories.items()
    }
    
    # ═══════════════════════════════════════════════════════════════════════
    # LINGUISTIC ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n2. Analyzing linguistic features...")
    linguistic_analysis = {}
    for name, cases in categories.items():
        linguistic_analysis[name] = analyze_linguistic_features(cases, name)
    
    results['linguistic_analysis'] = linguistic_analysis
    
    # ═══════════════════════════════════════════════════════════════════════
    # PRINT SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n" + "-" * 70)
    print("CATEGORY DISTRIBUTION")
    print("-" * 70)
    
    gt_norm = (ground_truth - ground_truth.min()) / (ground_truth.max() - ground_truth.min() + 1e-8)
    cgt_errors = np.abs(cgt_sims - gt_norm)
    euc_errors = np.abs(euc_sims - gt_norm)
    error_diff = euc_errors - cgt_errors
    
    total = len(cgt_sims)
    cgt_wins = np.sum(error_diff > 0.1)
    euc_wins = np.sum(error_diff < -0.1)
    ties = total - cgt_wins - euc_wins
    
    print(f"  CGT Wins:        {cgt_wins:>5} ({cgt_wins/total*100:.1f}%)")
    print(f"  Euclidean Wins:  {euc_wins:>5} ({euc_wins/total*100:.1f}%)")
    print(f"  Ties:            {ties:>5} ({ties/total*100:.1f}%)")
    
    results['summary'] = {
        'cgt_wins': int(cgt_wins),
        'euc_wins': int(euc_wins),
        'ties': int(ties),
        'cgt_wins_pct': float(cgt_wins/total*100),
        'euc_wins_pct': float(euc_wins/total*100),
        'ties_pct': float(ties/total*100),
    }
    
    # ═══════════════════════════════════════════════════════════════════════
    # EXAMPLE CASES
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n" + "-" * 70)
    print("EXAMPLE DISAGREEMENT CASES")
    print("-" * 70)
    
    # Show top CGT wins
    if categories['cgt_wins_high']:
        print("\n📗 TOP CGT WINS (CGT much better than Euclidean):")
        for i, case in enumerate(categories['cgt_wins_high'][:3]):
            print(f"\n  {i+1}. Ground Truth: {case['ground_truth']:.2f}")
            print(f"     CGT: {case['cgt_sim']:.3f} (error: {case['cgt_error']:.3f})")
            print(f"     Euc: {case['euc_sim']:.3f} (error: {case['euc_error']:.3f})")
            print(f"     Sentence 1: {case['sentence1'][:80]}...")
            print(f"     Sentence 2: {case['sentence2'][:80]}...")
    
    # Show top Euclidean wins
    if categories['euc_wins_high']:
        print("\n📕 TOP EUCLIDEAN WINS (Euclidean much better than CGT):")
        for i, case in enumerate(categories['euc_wins_high'][:3]):
            print(f"\n  {i+1}. Ground Truth: {case['ground_truth']:.2f}")
            print(f"     CGT: {case['cgt_sim']:.3f} (error: {case['cgt_error']:.3f})")
            print(f"     Euc: {case['euc_sim']:.3f} (error: {case['euc_error']:.3f})")
            print(f"     Sentence 1: {case['sentence1'][:80]}...")
            print(f"     Sentence 2: {case['sentence2'][:80]}...")
    
    # ═══════════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ═══════════════════════════════════════════════════════════════════════
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualization
    fig = plot_disagreement_analysis(
        cgt_sims, euc_sims, ground_truth,
        output_path=output_dir / 'disagreement_analysis.png'
    )
    fig.savefig(output_dir / 'disagreement_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # JSON results
    with open(output_dir / 'qualitative_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # CSV for all disagreement cases
    all_cases = []
    for cat_name, cases in categories.items():
        for case in cases:
            case['category'] = cat_name
            all_cases.append(case)
    
    pd.DataFrame(all_cases).to_csv(output_dir / 'disagreement_cases.csv', index=False)
    
    print(f"\n📁 Results saved to: {output_dir}")
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#                    MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("PART VII - QUALITATIVE ANALYSIS: DISAGREEMENT CASES")
    print("=" * 80)
    
    print("\n⚠️ This module requires pre-computed predictions.")
    print("   Use run_qualitative_analysis() with your data.")

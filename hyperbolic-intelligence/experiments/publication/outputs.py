# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
Publication Outputs
===================

Generate publication-ready outputs: CSV, JSON, LaTeX, PDF figures.

RULE 8️⃣: Outputs limpos para paper
- métricas → .csv / .json
- tabelas → .tex
- figuras → .pdf (vetorial)

Author: Éric Gustavo Reis de Sena
Date: January 2026
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

# Optional visualization imports
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ═══════════════════════════════════════════════════════════════════════════════
#                    COLORS (Publication-ready)
# ═══════════════════════════════════════════════════════════════════════════════

COLORS = {
    'cgt': '#2E86AB',           # Blue - CGT main
    'euclidean': '#E74C3C',     # Red - Euclidean baseline
    'mrl': '#9B59B6',           # Purple - MRL
    'pca': '#7F8C8D',           # Gray - PCA
    'teacher': '#1ABC9C',       # Teal - Teacher
    'sq': '#F39C12',            # Orange - Scalar Quantization
    'pq': '#27AE60',            # Green - Product Quantization
    'bq': '#8E44AD',            # Dark Purple - Binary Quantization
}


# ═══════════════════════════════════════════════════════════════════════════════
#                    METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def save_metrics(
    data: Union[Dict, pd.DataFrame],
    filepath: Path,
    fmt: str = 'csv',
) -> None:
    """
    Save metrics to file.
    
    Args:
        data: Dictionary or DataFrame
        filepath: Output path
        fmt: Format ('csv', 'json')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if fmt == 'csv':
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data
        df.to_csv(filepath, index=False)
    
    elif fmt == 'json':
        if isinstance(data, pd.DataFrame):
            data = data.to_dict('records')
        
        # Convert numpy types
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=convert)
    
    print(f"💾 Saved: {filepath}")


# ═══════════════════════════════════════════════════════════════════════════════
#                    LATEX TABLES
# ═══════════════════════════════════════════════════════════════════════════════

def to_latex_table(
    df: pd.DataFrame,
    caption: str,
    label: str,
    bold_max: bool = True,
    precision: int = 4,
    columns: Optional[List[str]] = None,
) -> str:
    """
    Convert DataFrame to LaTeX table.
    
    Args:
        df: Input DataFrame
        caption: Table caption
        label: LaTeX label (tab:xxx)
        bold_max: Bold maximum values per column
        precision: Decimal precision
        columns: Columns to include (None = all)
    
    Returns:
        LaTeX table string
    """
    if columns:
        df = df[columns]
    
    # Format numbers
    df_formatted = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        if bold_max:
            max_val = df[col].max()
            df_formatted[col] = df[col].apply(
                lambda x: f"\\textbf{{{x:.{precision}f}}}" if x == max_val 
                else f"{x:.{precision}f}"
            )
        else:
            df_formatted[col] = df[col].apply(lambda x: f"{x:.{precision}f}")
    
    # Generate LaTeX
    latex = df_formatted.to_latex(
        index=False,
        escape=False,
        column_format='l' + 'c' * (len(df.columns) - 1),
    )
    
    # Wrap in table environment
    full_latex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
{latex}
\\end{{table}}"""
    
    return full_latex


def save_latex_table(
    latex: str,
    filepath: Path,
) -> None:
    """Save LaTeX table to file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        f.write(latex)
    
    print(f"📄 Saved: {filepath}")


# ═══════════════════════════════════════════════════════════════════════════════
#                    FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

def setup_publication_style():
    """Setup matplotlib for publication-quality figures."""
    if not HAS_MATPLOTLIB:
        return
    
    plt.rcParams.update({
        # Figure size
        'figure.figsize': (8, 5),
        'figure.dpi': 150,
        
        # Font
        'font.size': 11,
        'font.family': 'serif',
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        
        # Lines
        'lines.linewidth': 2,
        'lines.markersize': 6,
        
        # Grid
        'axes.grid': True,
        'grid.alpha': 0.3,
        
        # Spines
        'axes.spines.top': False,
        'axes.spines.right': False,
        
        # Save
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })


def save_figure(
    fig,
    filepath: Path,
    formats: List[str] = ['pdf', 'png'],
) -> None:
    """
    Save figure in multiple formats.
    
    Args:
        fig: matplotlib figure
        filepath: Base path (without extension)
        formats: List of formats to save
    """
    if not HAS_MATPLOTLIB:
        print("⚠️ matplotlib not available")
        return
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    for fmt in formats:
        out_path = filepath.with_suffix(f'.{fmt}')
        fig.savefig(out_path, format=fmt)
        print(f"📊 Saved: {out_path}")
    
    plt.close(fig)


def plot_comparison_bar(
    results: Dict[str, float],
    teacher_score: float,
    title: str = "Performance Comparison",
    ylabel: str = "Spearman ρ",
    filepath: Optional[Path] = None,
) -> Optional[Any]:
    """
    Create bar chart comparing methods.
    
    Args:
        results: Dict mapping method name to score
        teacher_score: Teacher baseline score
        title: Figure title
        ylabel: Y-axis label
        filepath: Output path (None = return figure)
    
    Returns:
        matplotlib figure if filepath is None
    """
    if not HAS_MATPLOTLIB:
        return None
    
    setup_publication_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Add teacher baseline
    all_results = {'Teacher': teacher_score, **results}
    
    methods = list(all_results.keys())
    scores = list(all_results.values())
    colors_list = [COLORS.get(m.lower(), '#333333') for m in methods]
    
    bars = ax.bar(methods, scores, color=colors_list, edgecolor='white', linewidth=1.5)
    
    # Add value labels
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f'{score:.4f}',
            ha='center',
            va='bottom',
            fontsize=9,
        )
    
    # Reference line
    ax.axhline(y=teacher_score, color=COLORS['teacher'], linestyle='--', 
               alpha=0.7, label='Teacher baseline')
    
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    
    if filepath:
        save_figure(fig, filepath)
        return None
    
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#                    SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def generate_summary(
    results: Dict[str, Any],
    teacher_spearman: float,
    output_dir: Path,
) -> Dict:
    """
    Generate comprehensive results summary.
    
    Args:
        results: All experiment results
        teacher_spearman: Teacher baseline
        output_dir: Directory for outputs
    
    Returns:
        Summary dictionary
    """
    summary = {
        'timestamp': datetime.now().isoformat(),
        'teacher_spearman': teacher_spearman,
        'experiments': {},
    }
    
    # Part IV.1: Euclidean Ablation
    if 'part_iv_1' in results:
        r = results['part_iv_1']
        summary['experiments']['euclidean_ablation'] = {
            'cgt_spearman': r.get('cgt', {}).get('test_spearman'),
            'euclidean_spearman': r.get('euclidean', {}).get('test_spearman'),
            'cgt_advantage': r.get('comparison', {}).get('cgt_advantage'),
            'verdict': 'CGT > Euclidean' if r.get('comparison', {}).get('cgt_wins') else 'No advantage',
        }
    
    # Part IV.2: MRL Comparison
    if 'part_iv_2' in results:
        summary['experiments']['mrl_comparison'] = results['part_iv_2']
    
    # Part IV.3: BQ Comparison
    if 'part_iv_3' in results:
        summary['experiments']['bq_comparison'] = results['part_iv_3']
    
    # Part I.19: Cascade Compression
    if 'part_i_19' in results:
        summary['experiments']['cascade_compression'] = results['part_i_19']
    
    # Part IV.4: Latency
    if 'part_iv_4' in results:
        summary['experiments']['latency_benchmark'] = results['part_iv_4']
    
    # Part VI: Statistical Robustness
    if 'part_vi' in results:
        summary['experiments']['statistical_robustness'] = results['part_vi']
    
    # Save summary
    save_metrics(summary, output_dir / 'summary.json', fmt='json')
    
    return summary


def print_summary_table(summary: Dict) -> None:
    """Print formatted summary table to console."""
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    
    print(f"\n📊 Teacher Baseline: ρ = {summary.get('teacher_spearman', 'N/A'):.4f}")
    
    exps = summary.get('experiments', {})
    
    if 'euclidean_ablation' in exps:
        e = exps['euclidean_ablation']
        print(f"\n🔷 Euclidean Ablation (Part IV.1)")
        print(f"   CGT:       ρ = {e.get('cgt_spearman', 'N/A'):.4f}")
        print(f"   Euclidean: ρ = {e.get('euclidean_spearman', 'N/A'):.4f}")
        print(f"   Advantage: {e.get('cgt_advantage', 0):+.4f}")
        print(f"   Verdict:   {e.get('verdict', 'N/A')}")
    
    if 'cascade_compression' in exps:
        c = exps['cascade_compression']
        print(f"\n🔷 Cascade Compression (Part I.19)")
        print(f"   See detailed results in output directory")
    
    print("\n" + "=" * 70)

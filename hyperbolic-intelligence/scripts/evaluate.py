# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
CGT Evaluation Script
=====================

Command-line interface for evaluating trained CGT models.

This script is an EXECUTOR ONLY interface. All logic is defined in the cgt package.

Usage:
    python scripts/evaluate.py --checkpoint results/cgt_model.pt
    python scripts/evaluate.py --checkpoint results/cgt_model.pt --run-mteb
    python scripts/evaluate.py --checkpoint results/cgt_model.pt --falsification

Author: Éric Gustavo Reis de Sena
Date: January 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
#                    ACADEMIC USE DISCLAIMER
# ═══════════════════════════════════════════════════════════════════════════════

DISCLAIMER = """
═══════════════════════════════════════════════════════════════════════════════
                    CONTRASTIVE GEOMETRIC TRANSFER (CGT)
                       Evaluation Script v1.0
═══════════════════════════════════════════════════════════════════════════════

Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.

Licensed under CC BY-NC-SA 4.0 - Academic Use Only.
Commercial use requires a separate license: eirikreisena@gmail.com

Patent Pending.
═══════════════════════════════════════════════════════════════════════════════
"""


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained CGT model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=DISCLAIMER,
    )
    
    # Required
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )
    
    # Evaluation options
    parser.add_argument(
        "--stsb",
        action="store_true",
        default=True,
        help="Evaluate on STS-B benchmark (default: True)",
    )
    parser.add_argument(
        "--mteb",
        action="store_true",
        help="Run MTEB evaluation suite",
    )
    parser.add_argument(
        "--mteb-tasks",
        type=str,
        nargs="+",
        default=["STSBenchmark", "SICKRelatedness"],
        help="MTEB tasks to evaluate",
    )
    parser.add_argument(
        "--falsification",
        action="store_true",
        help="Run falsification protocols (F1-F3)",
    )
    
    # Geometric metrics
    parser.add_argument(
        "--compute-gromov",
        action="store_true",
        help="Compute Gromov δ-hyperbolicity",
    )
    parser.add_argument(
        "--compute-distortion",
        action="store_true",
        help="Compute embedding distortion metrics",
    )
    parser.add_argument(
        "--compute-erank",
        action="store_true",
        help="Compute effective rank",
    )
    
    # Output
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="table",
        choices=["table", "json", "latex"],
        help="Output format",
    )
    
    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for evaluation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for encoding",
    )
    
    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress disclaimer banner",
    )
    
    return parser.parse_args()


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple:
    """
    Load model checkpoint.
    
    Returns:
        Tuple of (model, config).
    """
    from cgt.experiments import ExperimentConfig, create_experiment
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Reconstruct config
    config = ExperimentConfig.from_dict(checkpoint["config"])
    
    # Create model with same architecture
    student, _, _, _ = create_experiment(config)
    
    # Load state dict
    student.load_state_dict(checkpoint["model_state_dict"])
    student.to(device)
    student.eval()
    
    return student, config


def format_results_table(results: Dict[str, Any]) -> str:
    """Format results as ASCII table."""
    lines = []
    lines.append("═" * 60)
    lines.append("                    EVALUATION RESULTS")
    lines.append("═" * 60)
    
    for category, metrics in results.items():
        lines.append(f"\n{category}:")
        lines.append("-" * 40)
        
        if isinstance(metrics, dict):
            for name, value in metrics.items():
                if isinstance(value, float):
                    lines.append(f"  {name:30s} {value:.4f}")
                else:
                    lines.append(f"  {name:30s} {value}")
        else:
            lines.append(f"  {metrics}")
    
    lines.append("═" * 60)
    return "\n".join(lines)


def format_results_latex(results: Dict[str, Any]) -> str:
    """Format results as LaTeX table."""
    lines = []
    lines.append("% CGT Evaluation Results")
    lines.append("% Auto-generated by evaluate.py")
    lines.append("")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{CGT Evaluation Results}")
    lines.append("\\begin{tabular}{lc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Metric} & \\textbf{Value} \\\\")
    lines.append("\\midrule")
    
    for category, metrics in results.items():
        lines.append(f"\\multicolumn{{2}}{{l}}{{\\textit{{{category}}}}} \\\\")
        
        if isinstance(metrics, dict):
            for name, value in metrics.items():
                if isinstance(value, float):
                    lines.append(f"{name} & {value:.4f} \\\\")
                else:
                    lines.append(f"{name} & {value} \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def main() -> int:
    """Main entry point for CGT evaluation."""
    args = parse_args()
    
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    if not args.quiet:
        print(DISCLAIMER)
    
    # Device setup
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    #                    LOAD MODEL
    # ═══════════════════════════════════════════════════════════════════════════
    
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    
    try:
        student, config = load_checkpoint(args.checkpoint, device)
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return 1
    
    logger.info(f"Model loaded: {config.hyperbolic_dim}D hyperbolic embeddings")
    
    # Import evaluation modules
    from cgt.evaluation import (
        evaluate_stsb,
        compute_effective_rank,
        compute_gromov_delta,
        compute_distortion,
        FalsificationProtocols,
    )
    
    results = {}
    
    # ═══════════════════════════════════════════════════════════════════════════
    #                    STS-B EVALUATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    if args.stsb:
        logger.info("Evaluating on STS-B benchmark...")
        
        from sentence_transformers import SentenceTransformer
        from datasets import load_dataset
        
        # Load teacher and dataset
        teacher = SentenceTransformer(config.teacher_model, device=str(device))
        dataset = load_dataset("mteb/stsbenchmark-sts")
        
        # Test split evaluation
        test_data = dataset["test"]
        
        with torch.no_grad():
            test_emb1 = teacher.encode(
                test_data["sentence1"],
                convert_to_tensor=True,
                device=str(device),
            )
            test_emb2 = teacher.encode(
                test_data["sentence2"],
                convert_to_tensor=True,
                device=str(device),
            )
        
        stsb_results = evaluate_stsb(
            student=student,
            teacher_emb1=test_emb1,
            teacher_emb2=test_emb2,
            gold_scores=test_data["score"],
        )
        
        results["STS-B Benchmark"] = stsb_results
        logger.info(f"STS-B Spearman ρ: {stsb_results['spearman']:.4f}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    #                    GEOMETRIC METRICS
    # ═══════════════════════════════════════════════════════════════════════════
    
    geometric_results = {}
    
    if args.compute_erank or args.compute_gromov or args.compute_distortion:
        logger.info("Computing geometric metrics...")
        
        # Get sample embeddings
        with torch.no_grad():
            sample_teacher = test_emb1[:500] if args.stsb else None
            if sample_teacher is not None:
                sample_student = student(sample_teacher)
                
                if args.compute_erank:
                    erank = compute_effective_rank(sample_student)
                    geometric_results["Effective Rank"] = erank
                    logger.info(f"Effective Rank: {erank:.2f}")
                
                if args.compute_gromov:
                    gromov = compute_gromov_delta(sample_student[:100], student.manifold)
                    geometric_results["Gromov δ-hyperbolicity"] = gromov
                    logger.info(f"Gromov δ: {gromov:.4f}")
                
                if args.compute_distortion:
                    distortion = compute_distortion(sample_teacher, sample_student, student.manifold)
                    geometric_results.update(distortion)
                    logger.info(f"Distortion: {distortion}")
    
    if geometric_results:
        results["Geometric Metrics"] = geometric_results
    
    # ═══════════════════════════════════════════════════════════════════════════
    #                    FALSIFICATION PROTOCOLS
    # ═══════════════════════════════════════════════════════════════════════════
    
    if args.falsification:
        logger.info("Running falsification protocols...")
        
        fp = FalsificationProtocols(
            student=student,
            perturbation_sigma=config.perturbation_sigma,
            max_amplification=config.f2_max_amplification,
        )
        
        with torch.no_grad():
            sample_batch = test_emb1[:256] if args.stsb else None
            if sample_batch is not None:
                fp_results = fp.run_all(sample_batch.to(device))
                results["Falsification Protocols"] = fp_results
                
                for protocol, result in fp_results.items():
                    logger.info(f"{protocol}: {result}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    #                    MTEB EVALUATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    if args.mteb:
        logger.info("Running MTEB evaluation...")
        logger.warning("MTEB evaluation requires custom wrapper - not yet implemented")
        # TODO: Implement MTEB wrapper for CGT models
    
    # ═══════════════════════════════════════════════════════════════════════════
    #                    OUTPUT RESULTS
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Format output
    if args.format == "json":
        output_str = json.dumps(results, indent=2, default=str)
    elif args.format == "latex":
        output_str = format_results_latex(results)
    else:
        output_str = format_results_table(results)
    
    print(output_str)
    
    # Save to file if requested
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        
        if args.format == "json":
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2, default=str)
        else:
            with open(args.output, "w") as f:
                f.write(output_str)
        
        logger.info(f"Results saved to {args.output}")
    
    logger.info("Evaluation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

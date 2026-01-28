# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
CGT Training Script
===================

Command-line interface for training Contrastive Geometric Transfer models.

This script is an EXECUTOR ONLY interface. All logic is defined in the cgt package.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --epochs 50 --lr 1e-4 --output results/

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
from typing import Any, Dict, Optional

import torch

# ═══════════════════════════════════════════════════════════════════════════════
#                    ACADEMIC USE DISCLAIMER
# ═══════════════════════════════════════════════════════════════════════════════

DISCLAIMER = """
═══════════════════════════════════════════════════════════════════════════════
                    CONTRASTIVE GEOMETRIC TRANSFER (CGT)
                       Training Script v1.0
═══════════════════════════════════════════════════════════════════════════════

Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.

Licensed under CC BY-NC-SA 4.0 - Academic Use Only.
Commercial use requires a separate license: eirikreisena@gmail.com

Patent Pending.
═══════════════════════════════════════════════════════════════════════════════
"""


def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """Configure logging with file and console handlers."""
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train CGT model for sentence embeddings compression.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=DISCLAIMER,
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML configuration file",
    )
    
    # Model Architecture
    parser.add_argument(
        "--teacher-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Teacher model name from sentence-transformers",
    )
    parser.add_argument(
        "--hyperbolic-dim",
        type=int,
        default=32,
        help="Hyperbolic embedding dimension",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden layer dimension",
    )
    
    # Training Parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size",
    )
    parser.add_argument(
        "--lr", "--learning-rate",
        type=float,
        default=2e-4,
        dest="learning_rate",
        help="Learning rate",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=200,
        help="Early stopping patience",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for optimizer",
    )
    
    # Loss Weights
    parser.add_argument(
        "--lambda-contrastive",
        type=float,
        default=1.0,
        help="Weight for contrastive loss",
    )
    parser.add_argument(
        "--lambda-distill",
        type=float,
        default=0.7,
        help="Weight for distillation loss",
    )
    parser.add_argument(
        "--lambda-spectral",
        type=float,
        default=0.2,
        help="Weight for spectral alignment loss",
    )
    parser.add_argument(
        "--lambda-topo",
        type=float,
        default=0.02,
        help="Weight for topological (Betti-0) loss",
    )
    parser.add_argument(
        "--lambda-lipschitz",
        type=float,
        default=0.005,
        help="Weight for Lipschitz regularization",
    )
    
    # Output
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results"),
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--checkpoint-prefix",
        type=str,
        default="cgt_model",
        help="Prefix for checkpoint files",
    )
    
    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for training",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float64",
        choices=["float32", "float64"],
        help="Data type for training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
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


def load_config_file(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file.
        
    Returns:
        Configuration dictionary.
    """
    try:
        import yaml
    except ImportError:
        logging.error("PyYAML not installed. Install with: pip install pyyaml")
        sys.exit(1)
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def merge_configs(file_config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Merge file configuration with command-line arguments.
    
    CLI arguments override file configuration.
    """
    config = file_config.copy()
    
    # Override with CLI args if explicitly provided
    args_dict = vars(args)
    for key, value in args_dict.items():
        if value is not None and key not in ["config", "quiet", "log_level"]:
            # Convert key format (e.g., learning_rate -> learning_rate)
            config[key] = value
    
    return config


def main() -> int:
    """Main entry point for CGT training."""
    args = parse_args()
    
    # Setup logging
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"train_{timestamp}.log"
    
    setup_logging(args.log_level, log_file)
    logger = logging.getLogger(__name__)
    
    # Print disclaimer
    if not args.quiet:
        print(DISCLAIMER)
    
    logger.info("Initializing CGT Training Pipeline")
    logger.info(f"Output directory: {output_dir}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    #                    CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Load configuration
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        file_config = load_config_file(args.config)
        config_dict = merge_configs(file_config, args)
    else:
        config_dict = vars(args)
    
    # Import CGT modules (late import to avoid startup delay)
    logger.info("Importing CGT modules...")
    try:
        from cgt.experiments import ExperimentConfig, create_experiment
        from cgt.experiments.trainer import CGTTrainer
        from cgt.evaluation import evaluate_stsb, FalsificationProtocols
        from cgt.utils import set_global_seed, academic_use_disclaimer
    except ImportError as e:
        logger.error(f"Failed to import CGT modules: {e}")
        logger.error("Make sure the package is installed: pip install -e .")
        return 1
    
    # Create experiment configuration
    config = ExperimentConfig.from_dict(config_dict)
    
    # Device setup
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    logger.info(f"Configuration:\n{json.dumps(config.to_dict(), indent=2, default=str)}")
    
    # Set random seed
    set_global_seed(config.seed)
    logger.info(f"Random seed: {config.seed}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    #                    MODEL CREATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    logger.info("Creating model and optimizer...")
    student, criterion, lipschitz_reg, training_config = create_experiment(config)
    
    logger.info(f"Student model parameters: {sum(p.numel() for p in student.parameters()):,}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    #                    DATA LOADING
    # ═══════════════════════════════════════════════════════════════════════════
    
    logger.info("Loading STS-B dataset and teacher embeddings...")
    
    try:
        from sentence_transformers import SentenceTransformer
        from datasets import load_dataset
    except ImportError as e:
        logger.error(f"Missing required dependency: {e}")
        return 1
    
    # Load teacher model
    teacher = SentenceTransformer(config.teacher_model, device=str(device))
    teacher_dim = teacher.get_sentence_embedding_dimension()
    logger.info(f"Teacher dimension: {teacher_dim}")
    
    # Load STS-B dataset
    dataset = load_dataset("mteb/stsbenchmark-sts")
    
    # Prepare training data
    train_sentences = dataset["train"]["sentence1"] + dataset["train"]["sentence2"]
    logger.info(f"Training sentences: {len(train_sentences):,}")
    
    # Encode with teacher
    logger.info("Encoding training data with teacher model...")
    with torch.no_grad():
        train_embeddings = teacher.encode(
            train_sentences,
            convert_to_tensor=True,
            show_progress_bar=True,
            device=str(device),
        )
    
    logger.info(f"Training embeddings shape: {train_embeddings.shape}")
    
    # Create data loader
    train_dataset = torch.utils.data.TensorDataset(train_embeddings)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    #                    TRAINING
    # ═══════════════════════════════════════════════════════════════════════════
    
    logger.info("Starting training...")
    
    trainer = CGTTrainer(
        student=student,
        criterion=criterion,
        lipschitz_reg=lipschitz_reg,
        **training_config,
    )
    
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=None,  # Validation handled internally
        epochs=config.num_epochs,
    )
    
    # Save training history
    history_path = output_dir / f"{args.checkpoint_prefix}_history_{timestamp}.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to {history_path}")
    
    # Save model checkpoint
    checkpoint_path = output_dir / f"{args.checkpoint_prefix}_{timestamp}.pt"
    torch.save({
        "model_state_dict": student.state_dict(),
        "config": config.to_dict(),
        "history": history,
    }, checkpoint_path)
    logger.info(f"Model checkpoint saved to {checkpoint_path}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    #                    EVALUATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    logger.info("Running final evaluation...")
    
    # Evaluate on STS-B test set
    test_sentences1 = dataset["test"]["sentence1"]
    test_sentences2 = dataset["test"]["sentence2"]
    test_scores = dataset["test"]["score"]
    
    # Encode test data
    with torch.no_grad():
        test_emb1 = teacher.encode(test_sentences1, convert_to_tensor=True, device=str(device))
        test_emb2 = teacher.encode(test_sentences2, convert_to_tensor=True, device=str(device))
    
    # Evaluate
    results = evaluate_stsb(
        student=student,
        teacher_emb1=test_emb1,
        teacher_emb2=test_emb2,
        gold_scores=test_scores,
    )
    
    logger.info(f"STS-B Test Results:")
    logger.info(f"  Spearman ρ: {results['spearman']:.4f}")
    logger.info(f"  Pearson r: {results.get('pearson', 'N/A')}")
    
    # Run falsification protocols
    if config.enable_f1_homotopy or config.enable_f2_stability or config.enable_f3_forman_ricci:
        logger.info("Running falsification protocols...")
        
        fp = FalsificationProtocols(
            student=student,
            perturbation_sigma=config.perturbation_sigma,
            max_amplification=config.f2_max_amplification,
        )
        
        # Sample batch for protocols
        sample_batch = train_embeddings[:256]
        fp_results = fp.run_all(sample_batch.to(device))
        
        for protocol, result in fp_results.items():
            logger.info(f"  {protocol}: {result}")
    
    # Save final results
    results_path = output_dir / f"{args.checkpoint_prefix}_results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump({
            "config": config.to_dict(),
            "stsb_results": results,
            "falsification": fp_results if "fp_results" in locals() else None,
        }, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")
    
    logger.info("Training complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

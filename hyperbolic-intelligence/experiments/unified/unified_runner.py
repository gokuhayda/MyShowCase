# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
Unified Experiment Runner
=========================

Main runner for executing all 5 models with proper isolation.
Supports single and parallel execution modes.

AUDIT COMPLIANCE:
- ✅ float64 enforced end-to-end
- ✅ Process isolation for parallel execution
- ✅ Deterministic seed reset per model
- ✅ GPU fallback to CPU documented
- ✅ Immediate persistence of results
- ✅ Does NOT modify existing runners

EXECUTION ENVIRONMENT: Google Colab compatible
- Ephemeral environment handling
- Incremental result saving
- FINISHED.flag per model
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

# Local imports (after path setup)
from unified.config import (
    ModelType,
    RunMode,
    ExecutionConfig,
    get_model_configs,
    validate_config,
    TrainingHyperparameters,
)
from unified.trainer import UnifiedTrainer

# REUSE existing helpers (NO DUPLICATION)
from cgt.utils.helpers import set_global_seed, get_device, clear_memory


# ═══════════════════════════════════════════════════════════════════════════════
#                         LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging(output_dir: Path, model_name: str = "unified") -> logging.Logger:
    """Setup logging for a specific model/process."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(f"cgt_{model_name}")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers = []
    
    # File handler
    fh = logging.FileHandler(log_dir / f"{model_name}.log")
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


# ═══════════════════════════════════════════════════════════════════════════════
#                         FLOAT64 ENFORCEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def enforce_float64() -> None:
    """
    Enforce float64 precision globally.
    
    CRITICAL: Must be called at the START of each process.
    Colab and PyTorch can silently use TF32/AMP otherwise.
    """
    torch.set_default_dtype(torch.float64)
    
    # Disable TF32 (Colab trap)
    if hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = False
    
    # Disable autocast
    torch.autocast(device_type='cuda', enabled=False)
    torch.autocast(device_type='cpu', enabled=False)


def get_device_info() -> Dict[str, str]:
    """Get device information for logging."""
    info = {
        "device": str(get_device()),
        "dtype": "float64",
        "cuda_available": str(torch.cuda.is_available()),
    }
    
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda or "N/A"
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}"
    
    return info


# ═══════════════════════════════════════════════════════════════════════════════
#                         DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_stsb_data(
    cache_dir: Optional[Path] = None,
    teacher_model: str = "all-MiniLM-L6-v2"
) -> Dict[str, torch.Tensor]:
    """
    Load STS-B dataset with teacher embeddings.
    
    Args:
        cache_dir: Optional cache directory
        teacher_model: Teacher model name (default: all-MiniLM-L6-v2 for 384D,
                       use "all-mpnet-base-v2" for 768D)
    
    Returns:
        Dictionary with train/val/test embeddings and scores
    """
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    
    print("[INFO] Loading STS-B dataset...")
    dataset = load_dataset("mteb/stsbenchmark-sts")
    
    print(f"[INFO] Loading teacher model: {teacher_model}...")
    # Handle both short and full names
    if "/" not in teacher_model:
        teacher_model = f"sentence-transformers/{teacher_model}"
    teacher = SentenceTransformer(teacher_model)
    
    data = {}
    
    for split in ["train", "validation", "test"]:
        print(f"[INFO] Encoding {split} split...")
        ds = dataset[split]
        
        emb1 = teacher.encode(ds["sentence1"], convert_to_tensor=True)
        emb2 = teacher.encode(ds["sentence2"], convert_to_tensor=True)
        scores = torch.tensor(ds["score"], dtype=torch.float64) / 5.0
        
        # Ensure float64
        emb1 = emb1.to(torch.float64)
        emb2 = emb2.to(torch.float64)
        
        data[f"{split}_emb1"] = emb1
        data[f"{split}_emb2"] = emb2
        data[f"{split}_scores"] = scores
    
    # Compute teacher baseline
    from scipy.stats import spearmanr
    
    test_emb1 = data["test_emb1"]
    test_emb2 = data["test_emb2"]
    test_scores = data["test_scores"]
    
    with torch.no_grad():
        teacher_sims = torch.nn.functional.cosine_similarity(test_emb1, test_emb2)
        teacher_rho, _ = spearmanr(teacher_sims.cpu().numpy(), test_scores.cpu().numpy())
    
    data["teacher_spearman"] = teacher_rho
    print(f"[INFO] Teacher baseline Spearman: {teacher_rho:.4f}")
    
    return data


# ═══════════════════════════════════════════════════════════════════════════════
#                         SINGLE MODEL TRAINING (WORKER)
# ═══════════════════════════════════════════════════════════════════════════════

def train_single_model(
    model_type: ModelType,
    data: Dict[str, torch.Tensor],
    output_dir: Path,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Train a single model with full isolation.
    
    This function is designed to run in an isolated process.
    
    Args:
        model_type: Which model to train
        data: Pre-loaded dataset
        output_dir: Output directory for this model
        seed: Random seed
    
    Returns:
        Training results dictionary
    """
    # === CRITICAL: Enforce float64 at process start ===
    enforce_float64()
    
    # Setup logging
    logger = setup_logging(output_dir, model_type.value)
    
    # Get device
    device = get_device()
    device_info = get_device_info()
    
    logger.info(f"Model={model_type.value} | Device={device} | dtype=float64")
    logger.info(f"Device info: {json.dumps(device_info)}")
    
    # Reset seed
    set_global_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Seed reset to {seed}")
    
    try:
        # === PATCH: Select trainer based on model type ===
        if model_type == ModelType.PSI_SLM_FULL:
            # Use PSI-SLM Full trainer with H-AKOrN + Topo
            from unified.psi_slm_trainer import PsiSlmFullTrainer
            trainer = PsiSlmFullTrainer(
                model_type=model_type,
                output_dir=output_dir,
                device=device,
                dtype=torch.float64,
            )
        else:
            # Use standard unified trainer
            trainer = UnifiedTrainer(
                model_type=model_type,
                output_dir=output_dir,
                device=device,
                dtype=torch.float64,
            )
        
        # Train
        results = trainer.train(
            train_emb1=data["train_emb1"],
            train_emb2=data["train_emb2"],
            train_scores=data["train_scores"],
            val_emb1=data["validation_emb1"],
            val_emb2=data["validation_emb2"],
            val_scores=data["validation_scores"],
        )
        
        # Evaluate on test
        test_results = trainer.evaluate(
            test_emb1=data["test_emb1"],
            test_emb2=data["test_emb2"],
            test_scores=data["test_scores"],
        )
        
        results["test_rho"] = test_results["test_rho"]
        results["teacher_spearman"] = data["teacher_spearman"]
        results["retention"] = results["test_rho"] / data["teacher_spearman"]
        
        # Save model
        model_path = trainer.save_model()
        results["model_path"] = str(model_path)
        
        # Save final results
        final_results_path = output_dir / f"{model_type.value}_final.json"
        with open(final_results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create FINISHED flag
        flag_path = output_dir / "FINISHED.flag"
        with open(flag_path, "w") as f:
            f.write(f"Completed: {datetime.now().isoformat()}\n")
            f.write(f"Test ρ: {results['test_rho']:.4f}\n")
            f.write(f"Retention: {results['retention']*100:.1f}%\n")
        
        logger.info(f"Training complete. Test ρ: {results['test_rho']:.4f}")
        logger.info(f"Results saved to: {final_results_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Save error
        error_path = output_dir / "ERROR.log"
        with open(error_path, "w") as f:
            f.write(f"Error: {str(e)}\n")
            f.write(traceback.format_exc())
        
        return {"error": str(e), "model_type": model_type.value}


# ═══════════════════════════════════════════════════════════════════════════════
#                         PARALLEL WORKER WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

def _worker_wrapper(args: Tuple) -> Dict[str, Any]:
    """
    Wrapper for multiprocessing worker.
    
    Ensures complete isolation:
    - New process = new Python interpreter state
    - Fresh RNG state
    - Fresh CUDA context (if available)
    """
    model_type, data_path, output_dir, seed = args
    
    # Load data in worker (avoid pickling large tensors)
    import torch
    data = torch.load(data_path, weights_only=False)
    
    return train_single_model(
        model_type=model_type,
        data=data,
        output_dir=Path(output_dir),
        seed=seed,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#                         UNIFIED RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

class UnifiedRunner:
    """
    Main experiment runner supporting single and parallel execution.
    
    MODES:
    - single: Run one model at a time
    - all_parallel: Run all models in parallel processes
    
    GUARANTEES:
    - float64 precision throughout
    - Process isolation (separate RNG, CUDA context)
    - Immediate persistence of results
    - FINISHED.flag per model
    """
    
    def __init__(self, config: ExecutionConfig):
        """
        Initialize runner.
        
        Args:
            config: Execution configuration
        """
        # Validate config
        errors = validate_config(config)
        if errors:
            raise ValueError(f"Invalid config: {errors}")
        
        self.config = config
        self.output_base = Path(config.output_base)
        self.checkpoint_dir = Path(config.checkpoint_dir)
        
        # Create directories
        self.output_base.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = setup_logging(self.output_base, "unified_runner")
        
        # Model configs
        self.model_configs = get_model_configs()
        
        # Results storage
        self.results: Dict[str, Any] = {}
        
        # Data (loaded lazily)
        self._data: Optional[Dict[str, torch.Tensor]] = None
        self._data_path: Optional[Path] = None
    
    def _get_models_to_run(self) -> List[ModelType]:
        """Get list of models to run based on config."""
        if self.config.mode == RunMode.SINGLE:
            if self.config.target_model is None:
                raise ValueError("SINGLE mode requires target_model")
            return [self.config.target_model]
        
        # All models
        models = list(ModelType)
        
        # Skip PSI_SLM if configured
        if self.config.skip_psi_slm:
            models = [m for m in models if m != ModelType.PSI_SLM]
        
        return models
    
    def _get_output_dir(self, model_type: ModelType) -> Path:
        """Get isolated output directory for a model."""
        model_dir = self.output_base / model_type.value
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir
    
    def _load_data(self) -> Dict[str, torch.Tensor]:
        """Load or return cached data."""
        if self._data is not None:
            return self._data
        
        self.logger.info("Loading data...")
        self._data = load_stsb_data()
        
        # Save data for parallel workers
        self._data_path = self.output_base / "data_cache.pt"
        torch.save(self._data, self._data_path)
        self.logger.info(f"Data cached to: {self._data_path}")
        
        return self._data
    
    def run_single(self, model_type: ModelType) -> Dict[str, Any]:
        """
        Run a single model.
        
        Args:
            model_type: Model to train
        
        Returns:
            Training results
        """
        # Enforce float64
        enforce_float64()
        
        self.logger.info(f"Running single model: {model_type.value}")
        
        # Check if already completed
        output_dir = self._get_output_dir(model_type)
        if (output_dir / "FINISHED.flag").exists():
            self.logger.info(f"Model {model_type.value} already complete. Skipping.")
            
            # Load existing results
            results_path = output_dir / f"{model_type.value}_final.json"
            if results_path.exists():
                with open(results_path) as f:
                    return json.load(f)
            return {"status": "skipped", "model_type": model_type.value}
        
        # Load data
        data = self._load_data()
        
        # Get seed
        hp = TrainingHyperparameters()
        seed = hp.seed
        
        # Train
        results = train_single_model(
            model_type=model_type,
            data=data,
            output_dir=output_dir,
            seed=seed,
        )
        
        self.results[model_type.value] = results
        return results
    
    def run_all_sequential(self) -> Dict[str, Any]:
        """
        Run all models sequentially.
        
        Returns:
            Dictionary of all results
        """
        enforce_float64()
        
        models = self._get_models_to_run()
        self.logger.info(f"Running {len(models)} models sequentially")
        
        for model_type in models:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Starting: {model_type.value}")
            self.logger.info(f"{'='*60}")
            
            results = self.run_single(model_type)
            self.results[model_type.value] = results
            
            # Clear memory between models
            clear_memory()
        
        # Save combined results
        self._save_combined_results()
        
        return self.results
    
    def run_all_parallel(self) -> Dict[str, Any]:
        """
        Run all models in parallel processes.
        
        ISOLATION GUARANTEES:
        - Each model runs in a separate process
        - Fresh Python interpreter state per process
        - Separate RNG state per process
        - Separate CUDA context per process
        
        Returns:
            Dictionary of all results
        """
        enforce_float64()
        
        models = self._get_models_to_run()
        self.logger.info(f"Running {len(models)} models in parallel")
        
        # Load data first (will be shared via file)
        self._load_data()
        
        # Prepare worker arguments
        hp = TrainingHyperparameters()
        base_seed = hp.seed
        
        worker_args = []
        for i, model_type in enumerate(models):
            output_dir = self._get_output_dir(model_type)
            
            # Skip if already complete
            if (output_dir / "FINISHED.flag").exists():
                self.logger.info(f"Skipping {model_type.value} (already complete)")
                continue
            
            # Different seed per model for statistical robustness
            # (but deterministic based on base_seed)
            model_seed = base_seed + i * 111
            
            worker_args.append((
                model_type,
                str(self._data_path),
                str(output_dir),
                model_seed,
            ))
        
        if not worker_args:
            self.logger.info("All models already complete")
            return self._load_existing_results()
        
        # Run in parallel
        n_workers = min(len(worker_args), self.config.max_workers)
        self.logger.info(f"Starting {n_workers} parallel workers")
        
        # Use spawn to ensure complete isolation
        ctx = mp.get_context("spawn")
        
        with ctx.Pool(n_workers) as pool:
            results_list = pool.map(_worker_wrapper, worker_args)
        
        # Collect results
        for args, result in zip(worker_args, results_list):
            model_type = args[0]
            self.results[model_type.value] = result
        
        # Save combined results
        self._save_combined_results()
        
        return self.results
    
    def run(self) -> Dict[str, Any]:
        """
        Run experiments based on config mode.
        
        Returns:
            Dictionary of results
        """
        self.logger.info(f"Starting unified runner in mode: {self.config.mode.value}")
        self.logger.info(f"Output: {self.output_base}")
        self.logger.info(f"Device info: {json.dumps(get_device_info())}")
        
        start_time = time.time()
        
        if self.config.mode == RunMode.SINGLE:
            results = self.run_single(self.config.target_model)
        elif self.config.mode == RunMode.ALL_SEQUENTIAL:
            results = self.run_all_sequential()
        elif self.config.mode == RunMode.ALL_PARALLEL:
            results = self.run_all_parallel()
        else:
            raise ValueError(f"Unknown mode: {self.config.mode}")
        
        elapsed = time.time() - start_time
        self.logger.info(f"Total time: {elapsed/60:.1f} minutes")
        
        return results
    
    def _load_existing_results(self) -> Dict[str, Any]:
        """Load results from completed models."""
        results = {}
        
        for model_type in self._get_models_to_run():
            output_dir = self._get_output_dir(model_type)
            results_path = output_dir / f"{model_type.value}_final.json"
            
            if results_path.exists():
                with open(results_path) as f:
                    results[model_type.value] = json.load(f)
        
        return results
    
    def _save_combined_results(self) -> None:
        """Save combined results from all models."""
        combined_path = self.output_base / "combined_results.json"
        
        with open(combined_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        
        self.logger.info(f"Combined results saved: {combined_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#                         ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment(
    mode: str = "all_sequential",
    target_model: Optional[str] = None,
    output_dir: str = "./outputs",
    checkpoint_dir: str = "./checkpoints",
    skip_psi_slm: bool = True,
    max_workers: int = 4,
) -> Dict[str, Any]:
    """
    Main entry point for running experiments.
    
    Args:
        mode: "single", "all_sequential", or "all_parallel"
        target_model: Model name for single mode
        output_dir: Output directory
        checkpoint_dir: Checkpoint directory
        skip_psi_slm: Whether to skip PSI_SLM model
        max_workers: Max parallel workers
    
    Returns:
        Dictionary of results
    
    Example:
        # Run single model
        results = run_experiment(mode="single", target_model="k_lighting_numerical_parity")
        
        # Run all sequentially
        results = run_experiment(mode="all_sequential")
        
        # Run all in parallel
        results = run_experiment(mode="all_parallel", max_workers=4)
    """
    # Parse mode
    mode_map = {
        "single": RunMode.SINGLE,
        "all_sequential": RunMode.ALL_SEQUENTIAL,
        "all_parallel": RunMode.ALL_PARALLEL,
    }
    
    if mode not in mode_map:
        raise ValueError(f"Unknown mode: {mode}. Use: {list(mode_map.keys())}")
    
    # Parse target model
    target = None
    if target_model:
        model_map = {m.value: m for m in ModelType}
        if target_model not in model_map:
            raise ValueError(f"Unknown model: {target_model}. Use: {list(model_map.keys())}")
        target = model_map[target_model]
    
    # Create config
    config = ExecutionConfig(
        mode=mode_map[mode],
        target_model=target,
        output_base=Path(output_dir),
        checkpoint_dir=Path(checkpoint_dir),
        skip_psi_slm=skip_psi_slm,
        max_workers=max_workers,
    )
    
    # Run
    runner = UnifiedRunner(config)
    return runner.run()


# Module entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CGT Unified Experiment Runner")
    parser.add_argument("--mode", default="all_sequential",
                        choices=["single", "all_sequential", "all_parallel"])
    parser.add_argument("--model", default=None, help="Model for single mode")
    parser.add_argument("--output", default="./outputs")
    parser.add_argument("--checkpoints", default="./checkpoints")
    parser.add_argument("--skip-psi", action="store_true", default=True)
    parser.add_argument("--workers", type=int, default=4)
    
    args = parser.parse_args()
    
    results = run_experiment(
        mode=args.mode,
        target_model=args.model,
        output_dir=args.output,
        checkpoint_dir=args.checkpoints,
        skip_psi_slm=args.skip_psi,
        max_workers=args.workers,
    )
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    for model, res in results.items():
        if "error" in res:
            print(f"{model}: ERROR - {res['error']}")
        else:
            rho = res.get("test_rho", "N/A")
            ret = res.get("retention", "N/A")
            if isinstance(ret, float):
                ret = f"{ret*100:.1f}%"
            print(f"{model}: ρ={rho:.4f if isinstance(rho, float) else rho} | Retention={ret}")

# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.

"""
Final Execution Pipeline
========================

Executes ALL 5 models and generates final analysis.
NO new methodological decisions - only execution, measurement, comparison.

MODELS:
1. K-Lighting AGI v2
2. K-Lighting Numerical Parity
3. CGT Paper Ready
4. Ψ-SLM (optional - different teacher)
5. Hybrid

AUDIT COMPLIANCE:
- ✅ Single evaluation pipeline (k_Lighting_NUMERICAL_PARITY)
- ✅ Same tests, metrics, order for all
- ✅ Automatic ranking (no opinion)
- ✅ Reproducible numerical logs
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cgt.utils.helpers import set_global_seed, get_device, clear_memory
from cgt.models.cgt_hardened import CGTStudentHardened
from cgt.geometry.lorentz_hardened import LorentzSubstrateHardened, LorentzConfig

# Local imports
from .evaluation import UnifiedEvaluator, EvaluationResult, enforce_float64
from .replication_configs import ReplicationModel, get_replication_config
from .hybrid_config import get_hybrid_config


# ═══════════════════════════════════════════════════════════════════════════════
#                    ARCHITECTURE INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def infer_architecture_from_checkpoint(state_dict: dict) -> dict:
    """
    Infer model architecture from checkpoint state_dict.
    
    Returns dict with:
        - teacher_dim: input dimension
        - hidden_dim: hidden layer dimension  
        - student_dim: output dimension (before +1 for time component)
    
    Works with both spectral_norm (weight_orig) and regular (weight) layers.
    """
    # Try spectral norm keys first, then regular
    weight_key_0 = None
    weight_key_6 = None
    
    for key in state_dict.keys():
        if 'projector.0.weight' in key and weight_key_0 is None:
            weight_key_0 = key
        if 'projector.6.weight' in key and weight_key_6 is None:
            weight_key_6 = key
    
    if weight_key_0 is None or weight_key_6 is None:
        # Fallback to defaults
        return {
            "teacher_dim": 384,
            "hidden_dim": 256,
            "student_dim": 32,
        }
    
    # projector.0.weight: (hidden_dim, teacher_dim)
    # projector.6.weight: (student_dim, hidden_dim)
    w0 = state_dict[weight_key_0]
    w6 = state_dict[weight_key_6]
    
    return {
        "teacher_dim": w0.shape[1],
        "hidden_dim": w0.shape[0],
        "student_dim": w6.shape[0],
    }


# ═══════════════════════════════════════════════════════════════════════════════
#                    DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_data_for_evaluation(
    teacher_model: str = "all-MiniLM-L6-v2",
) -> Dict[str, Any]:
    """
    Load STS-B data with specified teacher.
    
    Args:
        teacher_model: Either "all-MiniLM-L6-v2" (384d) or "all-mpnet-base-v2" (768d)
    """
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    from scipy.stats import spearmanr
    
    print(f"[INFO] Loading teacher: {teacher_model}")
    teacher = SentenceTransformer(f"sentence-transformers/{teacher_model}")
    
    print("[INFO] Loading STS-B dataset...")
    dataset = load_dataset("mteb/stsbenchmark-sts")
    
    data = {}
    
    for split in ["train", "validation", "test"]:
        print(f"[INFO] Encoding {split}...")
        ds = dataset[split]
        
        emb1 = teacher.encode(ds["sentence1"], convert_to_tensor=True)
        emb2 = teacher.encode(ds["sentence2"], convert_to_tensor=True)
        scores = torch.tensor(ds["score"], dtype=torch.float64) / 5.0
        
        data[f"{split}_emb1"] = emb1.to(torch.float64)
        data[f"{split}_emb2"] = emb2.to(torch.float64)
        data[f"{split}_scores"] = scores
    
    # Teacher baseline
    with torch.no_grad():
        teacher_sims = torch.nn.functional.cosine_similarity(
            data["test_emb1"], data["test_emb2"]
        )
        teacher_rho, _ = spearmanr(
            teacher_sims.cpu().numpy(),
            data["test_scores"].cpu().numpy()
        )
    
    data["teacher_spearman"] = teacher_rho
    data["teacher_dim"] = data["test_emb1"].shape[1]
    data["teacher_model"] = teacher_model
    
    print(f"[INFO] Teacher baseline: ρ = {teacher_rho:.4f}")
    
    return data


# ═══════════════════════════════════════════════════════════════════════════════
#                    MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_trained_model(
    checkpoint_path: Path,
    input_dim: int,
    output_dim: int = 32,
    hidden_dim: int = 256,
    curvature: float = -1.0,
    device: torch.device = None,
) -> tuple:
    """Load a trained model from checkpoint."""
    if device is None:
        device = get_device()
    
    # Create model (creates its own substrate)
    model = CGTStudentHardened(
        teacher_dim=input_dim,
        student_dim=output_dim,
        hidden_dim=hidden_dim,
        initial_curvature=abs(curvature),
    ).to(device).to(torch.float64)
    
    # Get substrate from model
    substrate = model.substrate
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model, substrate


# ═══════════════════════════════════════════════════════════════════════════════
#                    FINAL EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

class FinalExecutor:
    """
    Executes all 5 models with unified evaluation.
    
    NO new decisions. Only:
    - Execute
    - Measure
    - Compare
    - Document
    """
    
    def __init__(
        self,
        output_base: Path,
        skip_psi_slm: bool = True,
    ):
        enforce_float64()
        
        self.output_base = Path(output_base)
        self.skip_psi_slm = skip_psi_slm
        self.device = get_device()
        
        # Create directories
        self.outputs_dir = self.output_base / "outputs"
        self.tables_dir = self.output_base / "tables"
        self.checkpoints_dir = self.output_base / "checkpoints"
        
        for d in [self.outputs_dir, self.tables_dir, self.checkpoints_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Evaluator
        self.evaluator = UnifiedEvaluator(device=self.device)
        
        # Execution log
        self.execution_log = {
            "start_time": datetime.now().isoformat(),
            "device": str(self.device),
            "models_executed": [],
            "errors": [],
        }
    
    def _get_models_to_run(self) -> List[str]:
        """Get ordered list of models to run."""
        models = [
            "k_light_numerical_parity",  # Reference first
            "k_light_agi_v2",
            "cgt_paper_ready",
            "hybrid",
        ]
        
        if not self.skip_psi_slm:
            models.insert(3, "psi_slm")  # Before hybrid
        
        return models
    
    def train_and_evaluate_model(
        self,
        model_name: str,
        data: Dict[str, Any],
    ) -> Optional[EvaluationResult]:
        """
        Train (if needed) and evaluate a single model.
        """
        # Import at method level to ensure availability
        from .replication_executor import ReplicationTrainer
        from .replication_configs import ReplicationModel
        
        print(f"\n{'#'*70}")
        print(f"# MODEL: {model_name}")
        print(f"{'#'*70}")
        
        model_output_dir = self.outputs_dir / model_name
        checkpoint_path = model_output_dir / "model_checkpoint.pth"
        
        # Check if already trained
        if not checkpoint_path.exists():
            print(f"[WARN] Checkpoint not found: {checkpoint_path}")
            print(f"[INFO] Training required. Running trainer...")
            
            # Import appropriate trainer
            if model_name == "hybrid":
                from .hybrid_executor import HybridTrainer, load_hybrid_data
                
                # Hybrid uses different teacher (768d)
                hybrid_data = load_hybrid_data()
                trainer = HybridTrainer(model_output_dir)
                trainer.train(
                    train_emb1=hybrid_data["train_emb1"],
                    train_emb2=hybrid_data["train_emb2"],
                    train_scores=hybrid_data["train_scores"],
                    val_emb1=hybrid_data["validation_emb1"],
                    val_emb2=hybrid_data["validation_emb2"],
                    val_scores=hybrid_data["validation_scores"],
                )
                
                # Reload data for evaluation
                data = hybrid_data
            else:
                model_enum = ReplicationModel(model_name)
                trainer = ReplicationTrainer(model_enum, model_output_dir)
                trainer.train(
                    train_emb1=data["train_emb1"],
                    train_emb2=data["train_emb2"],
                    train_scores=data["train_scores"],
                    val_emb1=data["validation_emb1"],
                    val_emb2=data["validation_emb2"],
                    val_scores=data["validation_scores"],
                )
        
        # Determine config
        if model_name == "hybrid":
            config = get_hybrid_config()
            input_dim = config.teacher_dim  # 768
        else:
            model_enum = ReplicationModel(model_name)
            config = get_replication_config(model_enum)
            input_dim = config.teacher_dim  # 384
        
        # Load model
        try:
            model, substrate = load_trained_model(
                checkpoint_path=checkpoint_path,
                input_dim=input_dim,
                output_dim=config.student_dim,
                hidden_dim=config.hidden_dim,
                curvature=config.curvature,
                device=self.device,
            )
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            self.execution_log["errors"].append({
                "model": model_name,
                "error": str(e),
            })
            return None
        
        # Evaluate
        result = self.evaluator.evaluate_model(
            model_name=model_name,
            model=model,
            substrate=substrate,
            test_emb1=data["test_emb1"],
            test_emb2=data["test_emb2"],
            test_scores=data["test_scores"],
            val_emb1=data["validation_emb1"],
            val_emb2=data["validation_emb2"],
            val_scores=data["validation_scores"],
            teacher_emb1=data["test_emb1"],
            teacher_emb2=data["test_emb2"],
            teacher_spearman=data["teacher_spearman"],
            teacher_dim=data.get("teacher_dim", data["test_emb1"].shape[1]),
            checkpoint_path=checkpoint_path,
        )
        
        self.execution_log["models_executed"].append(model_name)
        
        # Clear memory
        del model
        clear_memory()
        
        return result
    
    def run_all(self) -> Dict[str, EvaluationResult]:
        """
        Execute all models.
        
        Returns:
            Dictionary of results per model
        """
        print("=" * 70)
        print("FINAL EXECUTION PIPELINE")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Output: {self.output_base}")
        print("=" * 70)
        
        start_time = time.time()
        
        # Load data with standard teacher (384d)
        print("\n[PHASE 1] Loading data (MiniLM, 384d)...")
        data_384 = load_data_for_evaluation("all-MiniLM-L6-v2")
        
        # Load data with hybrid teacher (768d)
        print("\n[PHASE 2] Loading data (mpnet, 768d)...")
        data_768 = load_data_for_evaluation("all-mpnet-base-v2")
        
        # Execute models
        print("\n[PHASE 3] Executing models...")
        models = self._get_models_to_run()
        
        # Models that require 768D teacher embeddings
        MODELS_REQUIRING_768D = {"hybrid", "psi_slm", "psi_slm_full"}
        
        for model_name in models:
            # Choose appropriate data based on architectural requirements
            if model_name in MODELS_REQUIRING_768D:
                data = data_768
            else:
                data = data_384
            
            self.train_and_evaluate_model(model_name, data)
        
        # Generate outputs
        print("\n[PHASE 4] Generating outputs...")
        self._generate_outputs()
        
        # Finalize
        elapsed = time.time() - start_time
        self.execution_log["end_time"] = datetime.now().isoformat()
        self.execution_log["total_time_seconds"] = elapsed
        
        print("\n" + "=" * 70)
        print("EXECUTION COMPLETE")
        print(f"Total time: {elapsed/60:.1f} minutes")
        print("=" * 70)
        
        return self.evaluator.results
    
    def _generate_outputs(self):
        """Generate all output files."""
        # Results table
        table = self.evaluator.get_results_table()
        
        table_path = self.tables_dir / "final_results.txt"
        with open(table_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("FINAL RESULTS TABLE\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            f.write(table)
        
        print(f"Saved: {table_path}")
        
        # JSON results
        self.evaluator.save_results(self.tables_dir)
        
        # Execution log
        log_path = self.outputs_dir / "execution_log.json"
        with open(log_path, "w") as f:
            json.dump(self.execution_log, f, indent=2, default=str)
        
        print(f"Saved: {log_path}")
    
    def create_checkpoints(self):
        """Create phase checkpoints."""
        # 05_execution_results_DONE.md
        exec_checkpoint = self.checkpoints_dir / "05_execution_results_DONE.md"
        
        content = f"""# PHASE 5: EXECUTION RESULTS CHECKPOINT
## Status: COMPLETE ✅

**Generated:** {datetime.now().isoformat()}

---

## 1. MODELS EXECUTED

"""
        for model in self.execution_log.get("models_executed", []):
            result = self.evaluator.results.get(model)
            if result:
                content += f"### {model}\n"
                content += f"- Test Spearman: {result.test_spearman:.4f}\n"
                content += f"- Retention: {result.retention_percent:.1f}%\n"
                content += f"- Falsification: F1={result.f1_projection_passed} F2={result.f2_distance_passed} F3={result.f3_topological_passed}\n\n"
        
        content += f"""
---

## 2. EXECUTION LOG

- Start: {self.execution_log.get('start_time', 'N/A')}
- End: {self.execution_log.get('end_time', 'N/A')}
- Duration: {self.execution_log.get('total_time_seconds', 0)/60:.1f} minutes
- Device: {self.execution_log.get('device', 'N/A')}

---

## 3. OUTPUTS GENERATED

- `outputs/` - Model checkpoints and logs
- `tables/final_results.txt` - Results table
- `tables/evaluation_results.json` - Full results
"""
        
        with open(exec_checkpoint, "w") as f:
            f.write(content)
        
        print(f"Saved: {exec_checkpoint}")


def run_final_execution(
    output_base: Path = Path("./final_output"),
    skip_psi_slm: bool = True,
) -> Dict[str, EvaluationResult]:
    """
    Main entry point for final execution.
    
    Args:
        output_base: Base output directory
        skip_psi_slm: Whether to skip PSI_SLM (different teacher)
    
    Returns:
        Dictionary of evaluation results
    """
    executor = FinalExecutor(output_base, skip_psi_slm)
    results = executor.run_all()
    executor.create_checkpoints()
    
    return results

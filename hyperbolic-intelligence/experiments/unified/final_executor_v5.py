# ==============================================================================
# final_executor_v5.py
# FULL CARTESIAN EXECUTOR - CRASH-RESILIENT WITH CHECKPOINT/RESUME
#
# FEATURES:
# - Automatic resume from last checkpoint
# - Progress saved after EVERY training
# - Atomic writes (no corruption on crash)
# - Detailed progress logging
# - GPU memory optimization
#
# Pipeline:
#   Para cada (Dataset × Teacher):
#       1. Treina CGT-GW (standalone, 7th model) [CHECKPOINT + EVAL]
#       Para cada (Student × Seed):
#           2. Treina Student diretamente (Teacher → Student) [CHECKPOINT]
#           3. Avalia no test split [SAVE RESULT]
#
# Author: CGT Research Team
# ==============================================================================

from pathlib import Path
from datetime import datetime
from enum import Enum
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, field, asdict
import json
import gc
import time
import hashlib
import shutil
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import v_measure_score
from sklearn.cluster import KMeans
from tqdm import tqdm

# CGT imports
from cgt.utils.helpers import set_global_seed, get_device
from cgt.geometry.lorentz_hardened import LorentzSubstrateHardened, LorentzConfig
from cgt.models.cgt_hardened import CGTStudentHardened
from cgt.models.graph_constructor import GraphConstructor, GraphConstructorConfig

# CGTGWProjector - single authoritative implementation (no fallback)
from cgt.models.cgt_gw_projector import CGTGWProjector, CGTGWProjectorConfig

# Assert to fail early if import failed
assert CGTGWProjector is not None, \
    "CGTGWProjector failed to import — installation is inconsistent. Install geoopt: pip install geoopt"

USE_FULL_PROJECTOR = True
print("✅ Using CGTGWProjector (single authoritative implementation)")


# ==============================================================================
# BOUNDARY ENFORCEMENT (CGT-GW REQUIRES float64)
# ==============================================================================

def enforce_cgtgw_boundary(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Enforce dtype boundary for CGT-GW projector.
    
    CGT-GW operates on hyperbolic manifolds requiring float64 precision.
    This function is the SINGLE POINT of dtype conversion before CGT-GW.
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor (any dtype)
    device : torch.device
        Target device
        
    Returns
    -------
    torch.Tensor
        Tensor in float64 on target device
        
    Raises
    ------
    AssertionError
        If conversion fails
    """
    # ═══════════════════════════════════════════════════════════════════
    # PRECISION BOUNDARY: Float32 (Transformer) → Float64 (CGT-GW)
    # .detach() breaks computational graph from encoder
    # .to(dtype=float64) converts to hyperbolic-safe precision
    # ═══════════════════════════════════════════════════════════════════
    x = x.detach().to(device=device, dtype=torch.float64)
    assert x.dtype == torch.float64, \
        f"Boundary enforcement failed: got {x.dtype}, expected float64"
    return x


# ==============================================================================
# CHECKPOINT MANAGER (CRASH-RESILIENT)
# ==============================================================================

class CheckpointManager:
    """
    Manages execution state with crash-resilient checkpointing.
    
    Features:
    - Atomic writes (temp file + rename)
    - Tracks completed: CGT-GW, Students, Evaluations
    - Auto-resume on restart
    - Progress statistics
    """
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.state_file = self.checkpoint_dir / "execution_state.json"
        self.state_backup = self.checkpoint_dir / "execution_state.backup.json"
        self.results_file = self.checkpoint_dir / "results_incremental.json"
        
        self.state = self._load_state()
        self.results = self._load_results()
    
    def _load_state(self) -> Dict:
        """Load state with fallback to backup."""
        for path in [self.state_file, self.state_backup]:
            if path.exists():
                try:
                    with open(path, 'r') as f:
                        state = json.load(f)
                        print(f"✅ Loaded checkpoint from {path.name}")
                        print(f"   CGT-GW completed: {len(state.get('completed_cgt_gw', []))}")
                        print(f"   Students completed: {len(state.get('completed_students', []))}")
                        return state
                except:
                    continue
        
        # Fresh start
        return {
            "completed_cgt_gw": [],      # List of "dataset|teacher" keys
            "completed_students": [],     # List of "dataset|teacher|student|seed" keys
            "failed": [],                 # List of failed items with error
            "start_time": datetime.now().isoformat(),
            "last_update": None,
        }
    
    def _load_results(self) -> List[Dict]:
        """Load incremental results."""
        if self.results_file.exists():
            try:
                with open(self.results_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return []
    
    def _save_state(self):
        """Atomic save with backup."""
        self.state["last_update"] = datetime.now().isoformat()
        
        # Write to temp file first
        temp_file = self.state_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(self.state, f, indent=2)
        
        # Backup existing
        if self.state_file.exists():
            shutil.copy(self.state_file, self.state_backup)
        
        # Atomic rename
        shutil.move(temp_file, self.state_file)
    
    def _save_results(self):
        """Save results incrementally."""
        temp_file = self.results_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        shutil.move(temp_file, self.results_file)
    
    # === CGT-GW ===
    
    def cgt_gw_key(self, dataset: str, teacher: str) -> str:
        return f"{dataset}|{teacher}"
    
    def is_cgt_gw_done(self, dataset: str, teacher: str) -> bool:
        return self.cgt_gw_key(dataset, teacher) in self.state["completed_cgt_gw"]
    
    def mark_cgt_gw_done(self, dataset: str, teacher: str):
        key = self.cgt_gw_key(dataset, teacher)
        if key not in self.state["completed_cgt_gw"]:
            self.state["completed_cgt_gw"].append(key)
            self._save_state()
    
    # === Students ===
    
    def student_key(self, dataset: str, teacher: str, student: str, seed: int) -> str:
        return f"{dataset}|{teacher}|{student}|{seed}"
    
    def is_student_done(self, dataset: str, teacher: str, student: str, seed: int) -> bool:
        return self.student_key(dataset, teacher, student, seed) in self.state["completed_students"]
    
    def mark_student_done(self, dataset: str, teacher: str, student: str, seed: int, result: Dict):
        key = self.student_key(dataset, teacher, student, seed)
        if key not in self.state["completed_students"]:
            self.state["completed_students"].append(key)
            self.results.append(result)
            self._save_state()
            self._save_results()
    
    # === Failures ===
    
    def mark_failed(self, key: str, error: str):
        self.state["failed"].append({
            "key": key,
            "error": error,
            "timestamp": datetime.now().isoformat(),
        })
        self._save_state()
    
    # === Progress ===
    
    def get_progress(self) -> Dict:
        return {
            "cgt_gw_completed": len(self.state["completed_cgt_gw"]),
            "students_completed": len(self.state["completed_students"]),
            "failed": len(self.state["failed"]),
            "results_saved": len(self.results),
        }
    
    def get_results(self) -> List[Dict]:
        return self.results


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class ExecutionConfig:
    """Configuration for execution mode."""
    
    # Scope
    scope: str = "full_cartesian"
    
    # Seeds
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1337])
    
    # GPU Optimization (FULL mode)
    # ═══════════════════════════════════════════════════════════════════
    # CRITICAL: AMP MUST BE DISABLED for CGT-GW hyperbolic geometry
    # AMP is fundamentally incompatible with float64 operations.
    # DO NOT change this to True.
    # ═══════════════════════════════════════════════════════════════════
    use_amp: bool = False  # FORCED OFF - CGT-GW requires float64
    batch_size_train: int = 512
    batch_size_eval: int = 1024
    num_workers: int = 4
    pin_memory: bool = True
    
    # Training
    cgt_gw_epochs: int = 100
    student_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 10
    
    # Architecture
    student_dim: int = 32
    hidden_dim: int = 256
    
    # Resume
    auto_resume: bool = True
    
    # Logging
    verbose: bool = True
    log_every: int = 10


class TaskType(Enum):
    STS = "sts"
    RERANKING = "reranking"
    CLUSTERING = "clustering"


@dataclass
class DatasetConfig:
    name: str
    hf_path: str
    split_train: str
    split_test: str
    task_type: TaskType
    sent1_col: Optional[str] = None
    sent2_col: Optional[str] = None
    score_col: Optional[str] = None
    text_col: Optional[str] = None
    label_col: Optional[str] = None


# ==============================================================================
# DATASETS
# ==============================================================================

STS_DATASETS = [
    DatasetConfig("STS12", "mteb/sts12-sts", "train", "test", TaskType.STS,
                  "sentence1", "sentence2", "score"),
    DatasetConfig("STS13", "mteb/sts13-sts", "train", "test", TaskType.STS,
                  "sentence1", "sentence2", "score"),
    DatasetConfig("STS14", "mteb/sts14-sts", "train", "test", TaskType.STS,
                  "sentence1", "sentence2", "score"),
    DatasetConfig("STS15", "mteb/sts15-sts", "train", "test", TaskType.STS,
                  "sentence1", "sentence2", "score"),
    DatasetConfig("STS16", "mteb/sts16-sts", "train", "test", TaskType.STS,
                  "sentence1", "sentence2", "score"),
    DatasetConfig("STSBenchmark", "mteb/stsbenchmark-sts", "train", "test", TaskType.STS,
                  "sentence1", "sentence2", "score"),
    DatasetConfig("SICK-R", "mteb/sickr-sts", "train", "test", TaskType.STS,
                  "sentence1", "sentence2", "score"),
    DatasetConfig("BIOSSES", "mteb/biosses-sts", "train", "test", TaskType.STS,
                  "sentence1", "sentence2", "score"),
]

RERANKING_DATASETS = [
    DatasetConfig("AskUbuntuDupQuestions", "mteb/askubuntudupquestions-reranking",
                  "train", "test", TaskType.RERANKING),
    DatasetConfig("SciDocsRR", "mteb/scidocs-reranking",
                  "train", "test", TaskType.RERANKING),
    DatasetConfig("StackOverflowDupQuestions", "mteb/stackoverflowdupquestions-reranking",
                  "train", "test", TaskType.RERANKING),
]

CLUSTERING_DATASETS = [
    DatasetConfig("TwentyNewsgroupsClustering", "mteb/twentynewsgroups-clustering",
                  "train", "test", TaskType.CLUSTERING,
                  text_col="text", label_col="label"),
    DatasetConfig("RedditClustering", "mteb/reddit-clustering",
                  "train", "test", TaskType.CLUSTERING,
                  text_col="text", label_col="label"),
]

ALL_DATASET_CONFIGS = STS_DATASETS + RERANKING_DATASETS + CLUSTERING_DATASETS


# ==============================================================================
# TEACHERS
# ==============================================================================

ALL_TEACHERS = [
    ("all-mpnet-base-v2", 768),
    ("all-MiniLM-L6-v2", 384),
    ("all-MiniLM-L12-v2", 384),
    ("all-distilroberta-v1", 768),
    ("multi-qa-mpnet-base-dot-v1", 768),
    ("multi-qa-MiniLM-L6-dot-v1", 384),
    ("BAAI/bge-small-en-v1.5", 384),
    ("BAAI/bge-base-en-v1.5", 768),
    ("BAAI/bge-large-en-v1.5", 1024),
    ("intfloat/e5-small-v2", 384),
    ("intfloat/e5-base-v2", 768),
    ("intfloat/e5-large-v2", 1024),
    ("thenlper/gte-small", 384),
    ("thenlper/gte-base", 768),
    ("thenlper/gte-large", 1024),
    ("paraphrase-multilingual-MiniLM-L12-v2", 384),
    ("distiluse-base-multilingual-cased-v2", 512),
    ("prajjwal1/bert-tiny", 128),
    ("prajjwal1/bert-mini", 256),
    ("prajjwal1/bert-small", 512),
    ("distilbert-base-uncased", 768),
    ("distilbert-base-multilingual-cased", 768),
    ("sentence-transformers/all-MiniLM-L6-v2", 384),
    ("sentence-transformers/all-MiniLM-L12-v2", 384),
    ("huawei-noah/TinyBERT_General_4L_312D", 312),
    ("huawei-noah/TinyBERT_General_6L_768D", 768),
    ("albert-base-v2", 768),
   # ("albert-small-v2", 768),
    ("google/mobilebert-uncased", 512),
    ("google/electra-small-discriminator", 256),
    ("microsoft/mpnet-base", 768),
]

CANONICAL_TEACHERS = ALL_TEACHERS[:6]


# ==============================================================================
# STUDENTS
# ==============================================================================

ALL_STUDENTS = [
    "CGT_PAPER_READY",
    "K_LIGHT_NUMERICAL_PARITY",
    "K_LIGHT_AGI_V2",
    "PSI_SLM",
    "HYBRID",
    "PSI_SLM_FULL",
]


# ==============================================================================
# STUDENT FACTORY
# ==============================================================================

def create_student(
    student_name: str,
    teacher_dim: int,
    student_dim: int = 32,
    hidden_dim: int = 256,
) -> Optional[nn.Module]:
    """Create student model."""
    
    if student_name in ["PSI_SLM", "HYBRID", "PSI_SLM_FULL"]:
        if teacher_dim != 768:
            return None
    
    return CGTStudentHardened(
        teacher_dim=teacher_dim,
        student_dim=student_dim,
        hidden_dim=hidden_dim,
    )


def create_cgt_gw_projector(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    device: torch.device,
) -> nn.Module:
    """
    Create CGT-GW projector (single authoritative implementation).
    
    Uses CGTGWProjector with NCA + Kuramoto dynamics.
    No fallback - requires geoopt to be installed.
    """
    projector_config = CGTGWProjectorConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        gw_embed_dim=min(16, output_dim // 4),
        gw_hidden_dim=hidden_dim,
        gw_coupling_strength=1.0,
        gw_temperature=0.5,
        gw_curvature=1.0,
        gw_num_steps=5,
        graph_method="knn",
        graph_k=8,
        graph_metric="cosine",
        aggregation="final",
    )
    return CGTGWProjector(projector_config).to(device).double()


# ==============================================================================
# TRAINING FUNCTIONS
# ==============================================================================

def train_cgt_gw(
    train_emb1: torch.Tensor,
    train_emb2: torch.Tensor,
    train_scores: torch.Tensor,
    val_emb1: torch.Tensor,
    val_emb2: torch.Tensor,
    val_scores: torch.Tensor,
    config: ExecutionConfig,
    device: torch.device,
) -> Tuple[nn.Module, Dict]:
    """
    Train CGT-GW Projector (single authoritative implementation).
    
    Uses CGTGWProjector with NCA + Kuramoto dynamics.
    No fallback - requires geoopt to be installed.
    """
    input_dim = train_emb1.shape[1]
    output_dim = config.hidden_dim
    
    # Full CGTGWProjector with NCA + Kuramoto dynamics
    projector_config = CGTGWProjectorConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        gw_embed_dim=min(16, output_dim // 4),
        gw_hidden_dim=config.hidden_dim,
        gw_coupling_strength=1.0,
        gw_temperature=0.5,
        gw_curvature=1.0,
        gw_num_steps=5,
        graph_method="knn",
        graph_k=8,
        graph_metric="cosine",
        aggregation="final",
        lambda_topo=0.1,
        lambda_gw=0.05,
        lambda_coherence=0.01,
    )
    model = CGTGWProjector(projector_config).to(device).double()
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.cgt_gw_epochs
    )
    
    scaler = GradScaler() if config.use_amp else None
    
    train_dataset = TensorDataset(train_emb1, train_emb2, train_scores)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size_train,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
    )
    
    best_val_rho = -1.0
    patience_counter = 0
    best_state = None
    
    for epoch in range(config.cgt_gw_epochs):
        model.train()
        
        for emb1, emb2, scores in train_loader:
            # BOUNDARY ENFORCEMENT: Convert to float64 BEFORE CGT-GW
            emb1 = enforce_cgtgw_boundary(emb1, device)
            emb2 = enforce_cgtgw_boundary(emb2, device)
            scores = scores.to(device, non_blocking=True).double()
            
            optimizer.zero_grad(set_to_none=True)
            
            # ============================
            # CGT-GW FORA DO AMP — OBRIGATÓRIO
            # CGTGWProjector opera em float64 por definição científica
            # ============================
            with autocast(enabled=False):
                assert emb1.dtype == torch.float64, f"Expected float64, got {emb1.dtype}"
                assert not torch.is_autocast_enabled(), \
                    "ERROR: AMP scope leaked into CGT-GW execution"
                
                # Project through CGTGWProjector
                z1 = model(emb1)
                z2 = model(emb2)
                
                # Distance-based loss
                dist = model.distance(z1, z2)
                scores_norm = scores / 5.0
                target_dist = 1.0 - scores_norm
                
                # Primary loss: distance preservation
                loss = F.mse_loss(torch.tanh(dist), target_dist)
            
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        
        scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            # BOUNDARY ENFORCEMENT: Convert to float64 BEFORE CGT-GW
            val_emb1_d = enforce_cgtgw_boundary(val_emb1, device)
            val_emb2_d = enforce_cgtgw_boundary(val_emb2, device)
            
            # CGT-GW FORA DO AMP
            with autocast(enabled=False):
                assert not torch.is_autocast_enabled(), \
                    "ERROR: AMP scope leaked into CGT-GW validation"
                z1 = model(val_emb1_d)
                z2 = model(val_emb2_d)
                sims = -model.distance(z1, z2)
            
            val_rho, _ = spearmanr(sims.cpu().numpy(), val_scores.numpy())
        
        if val_rho > best_val_rho:
            best_val_rho = val_rho
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break
    
    if best_state:
        model.load_state_dict(best_state)
    
    return model, {"best_val_rho": best_val_rho}


def train_student(
    train_emb1: torch.Tensor,
    train_emb2: torch.Tensor,
    train_scores: torch.Tensor,
    val_emb1: torch.Tensor,
    val_emb2: torch.Tensor,
    val_scores: torch.Tensor,
    student_name: str,
    config: ExecutionConfig,
    device: torch.device,
    seed: int,
) -> Tuple[Optional[nn.Module], Dict]:
    """Train student model using Teacher embeddings directly."""
    
    set_global_seed(seed)
    
    # Get Teacher output dimension directly
    teacher_dim = train_emb1.shape[1]
    
    student = create_student(
        student_name=student_name,
        teacher_dim=teacher_dim,
        student_dim=config.student_dim,
        hidden_dim=config.hidden_dim,
    )
    
    if student is None:
        return None, {"status": "incompatible"}
    
    student = student.to(device).double()
    
    # Use Teacher embeddings directly (no CGT-GW projection)
    train_teacher1 = train_emb1.cpu()
    train_teacher2 = train_emb2.cpu()
    val_teacher1 = val_emb1.cpu()
    val_teacher2 = val_emb2.cpu()
    
    torch.cuda.empty_cache()
    
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.student_epochs
    )
    
    scaler = GradScaler() if config.use_amp else None
    
    train_dataset = TensorDataset(train_teacher1, train_teacher2, train_scores)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size_train,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
    )
    
    best_val_rho = -1.0
    patience_counter = 0
    best_state = None
    
    for epoch in range(config.student_epochs):
        student.train()
        
        for emb1, emb2, scores in train_loader:
            emb1 = emb1.to(device, non_blocking=True).double()
            emb2 = emb2.to(device, non_blocking=True).double()
            scores = scores.to(device, non_blocking=True).double()
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(enabled=config.use_amp):
                z1 = student(emb1)
                z2 = student(emb2)
                sims = F.cosine_similarity(z1, z2)
                scores_norm = scores / 5.0
                loss = F.mse_loss(sims, scores_norm)
            
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
        
        scheduler.step()
        
        # Validation
        student.eval()
        with torch.no_grad():
            val_t1 = val_teacher1.to(device).double()
            val_t2 = val_teacher2.to(device).double()
            
            with autocast(enabled=config.use_amp):
                z1 = student(val_t1)
                z2 = student(val_t2)
                sims = F.cosine_similarity(z1, z2)
            
            val_rho, _ = spearmanr(sims.cpu().numpy(), val_scores.numpy())
        
        if val_rho > best_val_rho:
            best_val_rho = val_rho
            best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break
    
    if best_state:
        student.load_state_dict(best_state)
    
    return student, {"best_val_rho": best_val_rho, "seed": seed}


# ==============================================================================
# EVALUATION FUNCTIONS
# ==============================================================================

def evaluate_sts(
    student: nn.Module,
    test_emb1: torch.Tensor,
    test_emb2: torch.Tensor,
    test_scores: np.ndarray,
    config: ExecutionConfig,
    device: torch.device,
) -> Dict:
    """Evaluate on STS task."""
    
    student.eval()
    
    with torch.no_grad():
        # Use Teacher embeddings directly
        test_t1 = test_emb1.to(device).double()
        test_t2 = test_emb2.to(device).double()
        
        with autocast(enabled=config.use_amp):
            z1 = student(test_t1)
            z2 = student(test_t2)
            sims = F.cosine_similarity(z1, z2)
        
        rho, pvalue = spearmanr(sims.cpu().numpy(), test_scores)
    
    return {
        "spearman": float(rho),
        "pvalue": float(pvalue),
        "metric_name": "spearman",
        "primary_metric": float(rho),
    }


def evaluate_reranking(
    student: nn.Module,
    embeddings: torch.Tensor,
    queries: List[int],
    positives: List[List[int]],
    config: ExecutionConfig,
    device: torch.device,
) -> Dict:
    """Evaluate on Reranking task."""
    
    student.eval()
    
    with torch.no_grad():
        # Use Teacher embeddings directly
        emb_d = embeddings.to(device).double()
        
        with autocast(enabled=config.use_amp):
            student_emb = student(emb_d)
            student_emb = F.normalize(student_emb, p=2, dim=1)
        
        sim_matrix = torch.mm(student_emb, student_emb.t()).cpu().numpy()
    
    aps = []
    for q_idx, pos_indices in zip(queries, positives):
        if len(pos_indices) == 0:
            continue
        
        sims = sim_matrix[q_idx].copy()
        sims[q_idx] = -np.inf
        ranked = np.argsort(-sims)
        
        pos_set = set(pos_indices)
        hits = 0
        prec_sum = 0.0
        
        for rank, doc_idx in enumerate(ranked, 1):
            if doc_idx in pos_set:
                hits += 1
                prec_sum += hits / rank
        
        if hits > 0:
            aps.append(prec_sum / len(pos_indices))
    
    map_score = float(np.mean(aps)) if aps else 0.0
    
    return {
        "map": map_score,
        "metric_name": "map",
        "primary_metric": map_score,
    }


def evaluate_clustering(
    student: nn.Module,
    embeddings: torch.Tensor,
    labels: np.ndarray,
    config: ExecutionConfig,
    device: torch.device,
) -> Dict:
    """Evaluate on Clustering task."""
    
    student.eval()
    
    with torch.no_grad():
        # Use Teacher embeddings directly
        emb_d = embeddings.to(device).double()
        
        with autocast(enabled=config.use_amp):
            student_emb = student(emb_d)
        
        emb_np = student_emb.cpu().numpy()
    
    n_clusters = len(np.unique(labels))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred_labels = kmeans.fit_predict(emb_np)
    
    v_score = v_measure_score(labels, pred_labels)
    
    return {
        "v_measure": float(v_score),
        "metric_name": "v_measure",
        "primary_metric": float(v_score),
    }


def evaluate_cgt_gw_standalone(
    cgt_gw: CGTGWProjector,
    test_emb1: torch.Tensor,
    test_emb2: torch.Tensor,
    test_scores: np.ndarray,
    config: ExecutionConfig,
    device: torch.device,
) -> Dict:
    """Evaluate CGT-GW as standalone model (7th model)."""
    
    cgt_gw.eval()
    
    with torch.no_grad():
        # Convert to float64 for CGT-GW
        test_emb1_d = enforce_cgtgw_boundary(test_emb1, device)
        test_emb2_d = enforce_cgtgw_boundary(test_emb2, device)
        
        with autocast(enabled=False):
            z1 = cgt_gw(test_emb1_d)
            z2 = cgt_gw(test_emb2_d)
            # Use negative distance as similarity
            sims = -cgt_gw.distance(z1, z2)
        
        rho, pvalue = spearmanr(sims.cpu().numpy(), test_scores)
    
    return {
        "spearman": float(rho),
        "pvalue": float(pvalue),
        "metric_name": "spearman",
        "primary_metric": float(rho),
    }


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_dataset_splits(
    dataset_config: DatasetConfig,
    teacher_model_name: str,
    device: str = "cuda",
    cache_dir: Optional[Path] = None,
) -> Dict:
    """Load and encode dataset splits."""
    from sentence_transformers import SentenceTransformer
    from datasets import load_dataset
    
    # Check cache
    if cache_dir:
        cache_file = cache_dir / f"{dataset_config.name}_{teacher_model_name.replace('/', '_')}.pt"
        if cache_file.exists():
            cached = torch.load(cache_file, weights_only=False)
            # ═══════════════════════════════════════════════════════════════
            # CACHE SANITIZATION: Ensure all embeddings are float64
            # Old cache may contain float32 embeddings from before the fix.
            # CGT-GW hyperbolic geometry requires float64.
            # ═══════════════════════════════════════════════════════════════
            for key in cached:
                if isinstance(cached[key], torch.Tensor) and cached[key].dtype == torch.float32:
                    if "emb" in key.lower() or key == "embeddings":
                        cached[key] = cached[key].double()
            return cached
    
    teacher = SentenceTransformer(teacher_model_name, device=device)
    teacher.eval()
    
    result = {}
    
    if dataset_config.task_type == TaskType.STS:
        # Try loading train split with cascading fallbacks
        train_data = None
        test_only_mode = False
        
        try:
            train_data = load_dataset(dataset_config.hf_path, split="train")
        except Exception:
            try:
                train_data = load_dataset(dataset_config.hf_path, split="validation")
            except Exception:
                # Dataset only has test split - will use test for everything
                test_only_mode = True
        
        test_data = load_dataset(dataset_config.hf_path, split="test")
        
        if test_only_mode:
            # Use test data and split it into train/val/test
            # 60% train, 20% val, 20% test
            with torch.no_grad():
                all_emb1 = teacher.encode(
                    test_data[dataset_config.sent1_col],
                    convert_to_tensor=True, batch_size=128, show_progress_bar=False
                )
                all_emb1 = all_emb1.detach().to(device=device, dtype=torch.float64)
                all_emb2 = teacher.encode(
                    test_data[dataset_config.sent2_col],
                    convert_to_tensor=True, batch_size=128, show_progress_bar=False
                )
                all_emb2 = all_emb2.detach().to(device=device, dtype=torch.float64)
            all_scores = torch.tensor(test_data[dataset_config.score_col], dtype=torch.float32)
            
            n = len(all_scores)
            indices = torch.randperm(n)
            train_end = int(0.6 * n)
            val_end = int(0.8 * n)
            
            result = {
                # ═══════════════════════════════════════════════════════════════
                # BOUNDARY CAST: Teacher embeddings → float64 (CGT-GW requirement)
                # SentenceTransformer outputs float32, but CGT-GW hyperbolic
                # geometry requires float64 for numerical stability.
                # Cast at storage time to avoid downstream dtype leaks.
                # ═══════════════════════════════════════════════════════════════
                "train_emb1": all_emb1[indices[:train_end]].cpu().double(),
                "train_emb2": all_emb2[indices[:train_end]].cpu().double(),
                "train_scores": all_scores[indices[:train_end]],
                "val_emb1": all_emb1[indices[train_end:val_end]].cpu().double(),
                "val_emb2": all_emb2[indices[train_end:val_end]].cpu().double(),
                "val_scores": all_scores[indices[train_end:val_end]],
                "test_emb1": all_emb1[indices[val_end:]].cpu().double(),
                "test_emb2": all_emb2[indices[val_end:]].cpu().double(),
                "test_scores": np.array(all_scores[indices[val_end:]].numpy(), dtype=np.float32),
                "teacher_dim": all_emb1.shape[1],
            }
        else:
            with torch.no_grad():
                train_emb1 = teacher.encode(
                    train_data[dataset_config.sent1_col],
                    convert_to_tensor=True, batch_size=128, show_progress_bar=False
                )
                train_emb1 = train_emb1.detach().to(device=device, dtype=torch.float64)
                train_emb2 = teacher.encode(
                    train_data[dataset_config.sent2_col],
                    convert_to_tensor=True, batch_size=128, show_progress_bar=False
                )
                train_emb2 = train_emb2.detach().to(device=device, dtype=torch.float64)
            train_scores = torch.tensor(train_data[dataset_config.score_col], dtype=torch.float32)
            
            with torch.no_grad():
                test_emb1 = teacher.encode(
                    test_data[dataset_config.sent1_col],
                    convert_to_tensor=True, batch_size=128, show_progress_bar=False
                )
                test_emb1 = test_emb1.detach().to(device=device, dtype=torch.float64)
                test_emb2 = teacher.encode(
                    test_data[dataset_config.sent2_col],
                    convert_to_tensor=True, batch_size=128, show_progress_bar=False
                )
                test_emb2 = test_emb2.detach().to(device=device, dtype=torch.float64)
            test_scores = np.array(test_data[dataset_config.score_col], dtype=np.float32)
            
            # Split train into train/val
            n = len(train_scores)
            indices = torch.randperm(n)
            split = int(0.8 * n)
            
            result = {
                # ═══════════════════════════════════════════════════════════════
                # BOUNDARY CAST: Teacher embeddings → float64 (CGT-GW requirement)
                # SentenceTransformer outputs float32, but CGT-GW hyperbolic
                # geometry requires float64 for numerical stability.
                # Cast at storage time to avoid downstream dtype leaks.
                # ═══════════════════════════════════════════════════════════════
                "train_emb1": train_emb1[indices[:split]].cpu().double(),
                "train_emb2": train_emb2[indices[:split]].cpu().double(),
                "train_scores": train_scores[indices[:split]],
                "val_emb1": train_emb1[indices[split:]].cpu().double(),
                "val_emb2": train_emb2[indices[split:]].cpu().double(),
                "val_scores": train_scores[indices[split:]],
                "test_emb1": test_emb1.cpu().double(),
                "test_emb2": test_emb2.cpu().double(),
                "test_scores": test_scores,
                "teacher_dim": train_emb1.shape[1],
            }
    
    elif dataset_config.task_type == TaskType.RERANKING:
        data = load_dataset(dataset_config.hf_path, split="test")
        
        all_texts = []
        text_to_idx = {}
        queries = []
        positives = []
        
        for item in data:
            query = item["query"]
            pos_list = item.get("positive", [])
            
            if query not in text_to_idx:
                text_to_idx[query] = len(all_texts)
                all_texts.append(query)
            
            queries.append(text_to_idx[query])
            
            pos_indices = []
            for p in pos_list:
                if p not in text_to_idx:
                    text_to_idx[p] = len(all_texts)
                    all_texts.append(p)
                pos_indices.append(text_to_idx[p])
            positives.append(pos_indices)
            
            for n in item.get("negative", []):
                if n not in text_to_idx:
                    text_to_idx[n] = len(all_texts)
                    all_texts.append(n)
        
        with torch.no_grad():
            embeddings = teacher.encode(
                all_texts, convert_to_tensor=True, batch_size=128, show_progress_bar=False
            )
            embeddings = embeddings.detach().to(device=device, dtype=torch.float64)
        
        result = {
            # BOUNDARY CAST: float32 → float64 for CGT-GW hyperbolic geometry
            "embeddings": embeddings.cpu().double(),
            "queries": queries,
            "positives": positives,
            "teacher_dim": embeddings.shape[1],
        }
    
    elif dataset_config.task_type == TaskType.CLUSTERING:
        data = load_dataset(dataset_config.hf_path, split="test")
        
        texts = data[dataset_config.text_col][:10000]
        labels = np.array(data[dataset_config.label_col][:10000])
        
        with torch.no_grad():
            embeddings = teacher.encode(
                texts, convert_to_tensor=True, batch_size=128, show_progress_bar=False
            )
            embeddings = embeddings.detach().to(device=device, dtype=torch.float64)
        
        result = {
            # BOUNDARY CAST: float32 → float64 for CGT-GW hyperbolic geometry
            "embeddings": embeddings.cpu().double(),
            "labels": labels,
            "teacher_dim": embeddings.shape[1],
        }
    
    # Cache
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(result, cache_file)
    
    del teacher
    torch.cuda.empty_cache()
    gc.collect()
    
    return result


# ==============================================================================
# MAIN EXECUTOR (CRASH-RESILIENT)
# ==============================================================================

def run_cartesian_execution_v5(
    output_dir: Path,
    config: Optional[ExecutionConfig] = None,
) -> Dict:
    """
    Full Cartesian Execution with crash-resilient checkpointing.
    
    Automatically resumes from last checkpoint on restart.
    """
    
    if config is None:
        config = ExecutionConfig()
    
    # ═══════════════════════════════════════════════════════════════════
    # CRITICAL: FORCE AMP OFF - CGT-GW hyperbolic geometry requires float64
    # AMP introduces implicit float32 casts that break CGT-GW computation.
    # This override is non-negotiable - do not remove.
    # ═══════════════════════════════════════════════════════════════════
    if config.use_amp:
        print("⚠️  WARNING: AMP was enabled but is incompatible with CGT-GW float64 geometry.")
        print("⚠️  FORCING use_amp=False to prevent dtype mismatches.")
        # Create new config with use_amp forced to False
        from dataclasses import replace
        config = replace(config, use_amp=False)
    
    device = get_device()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Directories
    checkpoint_dir = output_dir / "checkpoints"
    model_dir = output_dir / "models"
    cache_dir = output_dir / "cache"
    
    for d in [checkpoint_dir, model_dir, cache_dir]:
        d.mkdir(exist_ok=True)
    
    # Initialize checkpoint manager
    ckpt_manager = CheckpointManager(checkpoint_dir)
    
    # Select scope
    if config.scope == "minimal":
        datasets = STS_DATASETS[:1]
        teachers = ALL_TEACHERS[:2]
        seeds = [42]
    elif config.scope == "canonical":
        datasets = STS_DATASETS[:3]
        teachers = CANONICAL_TEACHERS
        seeds = [42, 123, 456]
    else:  # full_cartesian
        datasets = ALL_DATASET_CONFIGS
        teachers = ALL_TEACHERS
        seeds = config.seeds
    
    # Calculate totals
    total_cgt_gw = len(datasets) * len(teachers)
    total_students = 0
    for d in datasets:
        for t_name, t_dim in teachers:
            for s in ALL_STUDENTS:
                if s in ["PSI_SLM", "HYBRID", "PSI_SLM_FULL"] and t_dim != 768:
                    continue
                total_students += len(seeds)
    
    start_time = time.time()
    
    # Print status
    progress = ckpt_manager.get_progress()
    print("="*80)
    print("CGT CARTESIAN EXECUTION v5 (CRASH-RESILIENT)")
    print("="*80)
    print(f"Scope: {config.scope}")
    print(f"Device: {device}")
    print(f"AMP: {config.use_amp}")
    print(f"Batch size: {config.batch_size_train}")
    print()
    print(f"📊 PROGRESS:")
    print(f"   CGT-GW: {progress['cgt_gw_completed']}/{total_cgt_gw}")
    print(f"   Students: {progress['students_completed']}/{total_students}")
    print(f"   Failed: {progress['failed']}")
    print()
    if progress['cgt_gw_completed'] > 0 or progress['students_completed'] > 0:
        print("🔄 RESUMING FROM CHECKPOINT...")
    print("="*80)
    
    # Main loop
    for dataset_config in tqdm(datasets, desc="Datasets"):
        dataset_name = dataset_config.name
        
        for teacher_name, teacher_dim in tqdm(teachers, desc=f"Teachers ({dataset_name})", leave=False):
            
            teacher_short = teacher_name.split("/")[-1]
            cgt_gw_key = f"{dataset_name}|{teacher_short}"
            
            # ===== LOAD DATA =====
            try:
                data = load_dataset_splits(
                    dataset_config,
                    teacher_name,
                    device=str(device),
                    cache_dir=cache_dir,
                )
            except Exception as e:
                print(f"\n❌ Error loading {dataset_name}/{teacher_short}: {e}")
                ckpt_manager.mark_failed(cgt_gw_key, str(e))
                continue
            
            # ===== CGT-GW =====
            cgt_gw_path = model_dir / dataset_name / teacher_short / "cgt_gw.pth"
            
            if ckpt_manager.is_cgt_gw_done(dataset_name, teacher_short):
                # Load existing
                projector_config = CGTGWProjectorConfig(
                    input_dim=data["teacher_dim"],
                    output_dim=config.hidden_dim,
                    gw_embed_dim=min(16, config.hidden_dim // 4),
                    gw_hidden_dim=config.hidden_dim,
                )
                cgt_gw = CGTGWProjector(projector_config).to(device).double()
                cgt_gw.load_state_dict(
                    torch.load(cgt_gw_path, map_location=device, weights_only=True)
                )
                
                # ===== EVALUATE EXISTING CGT-GW AS STANDALONE MODEL =====
                cgt_gw_result_path = cgt_gw_path.parent / "cgt_gw_eval.json"
                if not cgt_gw_result_path.exists() and dataset_config.task_type == TaskType.STS:
                    try:
                        test_metrics = evaluate_cgt_gw_standalone(
                            cgt_gw=cgt_gw,
                            test_emb1=data["test_emb1"],
                            test_emb2=data["test_emb2"],
                            test_scores=data["test_scores"],
                            config=config,
                            device=device,
                        )
                        
                        cgt_gw_result = {
                            "dataset": dataset_name,
                            "teacher": teacher_name,
                            "teacher_dim": teacher_dim,
                            "model": "CGT_GW",
                            "model_type": "standalone_projector",
                            "task_type": dataset_config.task_type.value,
                            **test_metrics,
                        }
                        
                        with open(cgt_gw_result_path, 'w') as f:
                            json.dump(cgt_gw_result, f, indent=2)
                        
                        print(f"✅ CGT-GW standalone evaluation (loaded): ρ = {test_metrics['spearman']:.4f}")
                    
                    except Exception as e:
                        print(f"⚠️  CGT-GW standalone evaluation failed: {e}")
            else:
                # Train new
                if dataset_config.task_type == TaskType.STS:
                    try:
                        cgt_gw, cgt_gw_metrics = train_cgt_gw(
                            train_emb1=data["train_emb1"],
                            train_emb2=data["train_emb2"],
                            train_scores=data["train_scores"],
                            val_emb1=data["val_emb1"],
                            val_emb2=data["val_emb2"],
                            val_scores=data["val_scores"],
                            config=config,
                            device=device,
                        )
                        
                        # Save
                        cgt_gw_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(cgt_gw.state_dict(), cgt_gw_path)
                        ckpt_manager.mark_cgt_gw_done(dataset_name, teacher_short)
                        
                        # ===== EVALUATE CGT-GW AS STANDALONE MODEL (7th MODEL) =====
                        # CGT-GW is now evaluated independently, not as part of distillation
                        try:
                            test_metrics = evaluate_cgt_gw_standalone(
                                cgt_gw=cgt_gw,
                                test_emb1=data["test_emb1"],
                                test_emb2=data["test_emb2"],
                                test_scores=data["test_scores"],
                                config=config,
                                device=device,
                            )
                            
                            # Record CGT-GW as independent model result
                            cgt_gw_result = {
                                "dataset": dataset_name,
                                "teacher": teacher_name,
                                "teacher_dim": teacher_dim,
                                "model": "CGT_GW",
                                "model_type": "standalone_projector",
                                "task_type": dataset_config.task_type.value,
                                "train_val_rho": cgt_gw_metrics.get("best_val_rho"),
                                **test_metrics,
                            }
                            
                            # Save CGT-GW result as JSON
                            cgt_gw_result_path = cgt_gw_path.parent / "cgt_gw_eval.json"
                            with open(cgt_gw_result_path, 'w') as f:
                                json.dump(cgt_gw_result, f, indent=2)
                            
                            print(f"✅ CGT-GW standalone evaluation: ρ = {test_metrics['spearman']:.4f}")
                        
                        except Exception as e:
                            print(f"⚠️  CGT-GW standalone evaluation failed: {e}")
                        
                    except Exception as e:
                        print(f"\n❌ CGT-GW training failed: {e}")
                        ckpt_manager.mark_failed(cgt_gw_key, traceback.format_exc())
                        continue
                else:
                    # For non-STS, create without training
                    projector_config = CGTGWProjectorConfig(
                        input_dim=data["teacher_dim"],
                        output_dim=config.hidden_dim,
                        gw_embed_dim=min(16, config.hidden_dim // 4),
                        gw_hidden_dim=config.hidden_dim,
                    )
                    cgt_gw = CGTGWProjector(projector_config).to(device).double()
                    cgt_gw_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(cgt_gw.state_dict(), cgt_gw_path)
                    ckpt_manager.mark_cgt_gw_done(dataset_name, teacher_short)
            
            # ===== STUDENTS =====
            for student_name in ALL_STUDENTS:
                
                # Compatibility check
                if student_name in ["PSI_SLM", "HYBRID", "PSI_SLM_FULL"]:
                    if teacher_dim != 768:
                        continue
                
                for seed in seeds:
                    
                    # Check if already done
                    if ckpt_manager.is_student_done(dataset_name, teacher_short, student_name, seed):
                        continue
                    
                    student_path = (
                        model_dir / dataset_name / teacher_short / 
                        student_name / f"seed_{seed}.pth"
                    )
                    student_key = f"{dataset_name}|{teacher_short}|{student_name}|{seed}"
                    
                    try:
                        # Train
                        if dataset_config.task_type != TaskType.STS:
                            continue  # Skip non-STS for now
                        
                        student, train_metrics = train_student(
                            train_emb1=data["train_emb1"],
                            train_emb2=data["train_emb2"],
                            train_scores=data["train_scores"],
                            val_emb1=data["val_emb1"],
                            val_emb2=data["val_emb2"],
                            val_scores=data["val_scores"],
                            student_name=student_name,
                            config=config,
                            device=device,
                            seed=seed,
                        )
                        
                        if student is None:
                            continue
                        
                        # Save model
                        student_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(student.state_dict(), student_path)
                        
                        # Evaluate
                        if dataset_config.task_type == TaskType.STS:
                            metrics = evaluate_sts(
                                student,
                                data["test_emb1"], data["test_emb2"], data["test_scores"],
                                config, device
                            )
                        elif dataset_config.task_type == TaskType.RERANKING:
                            metrics = evaluate_reranking(
                                student,
                                data["embeddings"], data["queries"], data["positives"],
                                config, device
                            )
                        elif dataset_config.task_type == TaskType.CLUSTERING:
                            metrics = evaluate_clustering(
                                student,
                                data["embeddings"], data["labels"],
                                config, device
                            )
                        
                        # Record result
                        result = {
                            "dataset": dataset_name,
                            "teacher": teacher_name,
                            "teacher_dim": teacher_dim,
                            "student": student_name,
                            "seed": seed,
                            "task_type": dataset_config.task_type.value,
                            "train_val_rho": train_metrics.get("best_val_rho"),
                            **metrics,
                        }
                        
                        # Mark as done (saves checkpoint)
                        ckpt_manager.mark_student_done(
                            dataset_name, teacher_short, student_name, seed, result
                        )
                        
                    except Exception as e:
                        print(f"\n❌ Student training failed: {student_key}")
                        print(f"   Error: {e}")
                        ckpt_manager.mark_failed(student_key, traceback.format_exc())
                        continue
            
            # Clear GPU memory after each teacher
            torch.cuda.empty_cache()
            gc.collect()
    
    # ===== FINAL SUMMARY =====
    elapsed = time.time() - start_time
    final_progress = ckpt_manager.get_progress()
    results = ckpt_manager.get_results()
    
    summary = {
        "config": {
            "scope": config.scope,
            "seeds": seeds,
            "batch_size": config.batch_size_train,
            "use_amp": config.use_amp,
        },
        "statistics": {
            "total_cgt_gw": total_cgt_gw,
            "total_students": total_students,
            "completed_cgt_gw": final_progress["cgt_gw_completed"],
            "completed_students": final_progress["students_completed"],
            "failed": final_progress["failed"],
        },
        "elapsed_seconds": elapsed,
        "timestamp": datetime.utcnow().isoformat(),
        "results": results,
    }
    
    # Save final summary
    summary_path = output_dir / "final_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*80)
    print("EXECUTION COMPLETE")
    print("="*80)
    print(f"CGT-GW: {final_progress['cgt_gw_completed']}/{total_cgt_gw}")
    print(f"Students: {final_progress['students_completed']}/{total_students}")
    print(f"Failed: {final_progress['failed']}")
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Results: {summary_path}")
    print("="*80)
    
    return summary


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    "run_cartesian_execution_v5",
    "ExecutionConfig",
    "CheckpointManager",
    "TaskType",
    "ALL_STUDENTS",
    "ALL_TEACHERS",
    "ALL_DATASET_CONFIGS",
    "STS_DATASETS",
    "RERANKING_DATASETS",
    "CLUSTERING_DATASETS",
]

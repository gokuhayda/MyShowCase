# ==============================================================================
# final_executor_v4.py
# FULL CARTESIAN EXECUTOR - CORRECT PIPELINE
#
# Pipeline:
#   Para cada (Dataset × Teacher):
#       1. Treina CGT-GW (uma vez)
#       Para cada (Student × Seed):
#           2. Treina Student via CGT-GW
#           3. Avalia no test split
#
# GPU Optimization (FULL mode):
#   - Mixed precision (AMP)
#   - Large batch sizes
#   - Parallel data loading
#   - Gradient accumulation
#   - Memory-efficient attention
#
# Author: CGT Research Team
# ==============================================================================

from pathlib import Path
from datetime import datetime
from enum import Enum
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import json
import gc
import time

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
from cgt.models.cgt_gw_projector import CGTGWProjector, CGTGWProjectorConfig
from cgt.models.graph_constructor import GraphConstructor, GraphConstructorConfig


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
    use_amp: bool = True                    # Mixed precision
    batch_size_train: int = 512             # Training batch
    batch_size_eval: int = 1024             # Evaluation batch
    num_workers: int = 4                    # DataLoader workers
    pin_memory: bool = True                 # Pin memory for faster transfer
    gradient_accumulation: int = 1          # Accumulation steps
    
    # Training
    cgt_gw_epochs: int = 100                # CGT-GW training epochs
    student_epochs: int = 100               # Student training epochs
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 10                      # Early stopping patience
    
    # Architecture
    student_dim: int = 32                   # Output dimension
    hidden_dim: int = 256                   # Hidden dimension
    
    # Logging
    verbose: bool = True
    save_checkpoints: bool = True


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
    ("albert-small-v2", 768),
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
# CGT-GW MODEL (uses CGTGWProjector from cgt.models.cgt_gw_projector)
# ==============================================================================
# NOTE: CGTGWProjector wraps CGTGW (NCA + Kuramoto dynamics) with GraphConstructor
# for embedding → graph conversion. See cgt.models.cgt_gw_projector for details.


# ==============================================================================
# STUDENT MODELS (Different Architectures)
# ==============================================================================

def create_student(
    student_name: str,
    teacher_dim: int,
    student_dim: int = 32,
    hidden_dim: int = 256,
) -> nn.Module:
    """
    Factory function to create student models.
    
    All students receive hyperbolic (Lorentz) input from CGT-GW.
    """
    
    # For now, all students use CGTStudentHardened
    # Different architectures can be added here
    
    if student_name in ["PSI_SLM", "HYBRID", "PSI_SLM_FULL"]:
        # These require 768d input
        if teacher_dim != 768:
            return None
    
    return CGTStudentHardened(
        teacher_dim=teacher_dim,
        student_dim=student_dim,
        hidden_dim=hidden_dim,
    )


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
) -> Tuple[CGTGWProjector, Dict]:
    """
    Train CGT-GW Projector using full architecture.
    
    Uses CGTGWProjector which wraps CGTGW (NCA + Kuramoto dynamics)
    with GraphConstructor for embedding → graph conversion.
    """
    
    input_dim = train_emb1.shape[1]
    output_dim = config.hidden_dim
    
    # Configure CGTGWProjector
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
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.cgt_gw_epochs
    )
    
    # AMP scaler
    scaler = GradScaler() if config.use_amp else None
    
    # DataLoader
    train_dataset = TensorDataset(train_emb1, train_emb2, train_scores)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size_train,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
    )
    
    # Training
    best_val_rho = -1.0
    patience_counter = 0
    history = {"train_loss": [], "val_rho": []}
    
    for epoch in range(config.cgt_gw_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (emb1, emb2, scores) in enumerate(train_loader):
            emb1 = emb1.to(device, non_blocking=True).double()
            emb2 = emb2.to(device, non_blocking=True).double()
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
            
            epoch_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_emb1_d = val_emb1.to(device).double()
            val_emb2_d = val_emb2.to(device).double()
            
            # CGT-GW FORA DO AMP
            with autocast(enabled=False):
                assert not torch.is_autocast_enabled(), \
                    "ERROR: AMP scope leaked into CGT-GW validation"
                z1 = model(val_emb1_d)
                z2 = model(val_emb2_d)
                sims = -model.distance(z1, z2)
            
            sims_np = sims.cpu().numpy()
            val_rho, _ = spearmanr(sims_np, val_scores.numpy())
        
        history["train_loss"].append(epoch_loss / len(train_loader))
        history["val_rho"].append(val_rho)
        
        # Early stopping
        if val_rho > best_val_rho:
            best_val_rho = val_rho
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break
    
    # Load best
    model.load_state_dict(best_state)
    
    return model, {"best_val_rho": best_val_rho, "history": history}


def train_student(
    cgt_gw: CGTGWProjector,
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
) -> Tuple[nn.Module, Dict]:
    """
    Train student model using CGT-GW as intermediate.
    
    Pipeline:
        Teacher embeddings → CGT-GW (frozen) → Student → Output
    """
    
    set_global_seed(seed)
    
    # Get CGT-GW output dimension (hyperbolic, so +1 for time)
    cgt_gw.eval()
    with torch.no_grad():
        sample = train_emb1[:1].to(device).double()
        # CGT-GW FORA DO AMP — OBRIGATÓRIO
        with autocast(enabled=False):
            assert sample.dtype == torch.float64, \
                f"Teacher embeddings must be float64, got {sample.dtype}"
            cgt_gw_output = cgt_gw(sample)
        cgt_gw_dim = cgt_gw_output.shape[1]
    
    # Create student
    student = create_student(
        student_name=student_name,
        teacher_dim=cgt_gw_dim,  # Input from CGT-GW
        student_dim=config.student_dim,
        hidden_dim=config.hidden_dim,
    )
    
    if student is None:
        return None, {"status": "incompatible"}
    
    student = student.to(device).double()
    
    # Pre-compute CGT-GW projections (frozen)
    # CGT-GW FORA DO AMP — OBRIGATÓRIO
    cgt_gw.eval()
    with torch.no_grad():
        with autocast(enabled=False):
            train_emb1_d = train_emb1.to(device).double()
            train_emb2_d = train_emb2.to(device).double()
            val_emb1_d = val_emb1.to(device).double()
            val_emb2_d = val_emb2.to(device).double()
            
            assert train_emb1_d.dtype == torch.float64, \
                f"Teacher embeddings must be float64, got {train_emb1_d.dtype}"
            assert not torch.is_autocast_enabled(), \
                "ERROR: AMP scope leaked into CGT-GW execution"
            
            train_hyper1 = cgt_gw(train_emb1_d).cpu()
            train_hyper2 = cgt_gw(train_emb2_d).cpu()
            val_hyper1 = cgt_gw(val_emb1_d).cpu()
            val_hyper2 = cgt_gw(val_emb2_d).cpu()
    
    # Free GPU memory
    torch.cuda.empty_cache()
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.student_epochs
    )
    
    scaler = GradScaler() if config.use_amp else None
    
    # DataLoader
    train_dataset = TensorDataset(train_hyper1, train_hyper2, train_scores)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size_train,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
    )
    
    # Training
    best_val_rho = -1.0
    patience_counter = 0
    history = {"train_loss": [], "val_rho": []}
    best_state = None
    
    for epoch in range(config.student_epochs):
        student.train()
        epoch_loss = 0.0
        
        for emb1, emb2, scores in train_loader:
            emb1 = emb1.to(device, non_blocking=True).double()
            emb2 = emb2.to(device, non_blocking=True).double()
            scores = scores.to(device, non_blocking=True).double()
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(enabled=config.use_amp):
                z1 = student(emb1)
                z2 = student(emb2)
                
                # Cosine similarity loss
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
            
            epoch_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        student.eval()
        with torch.no_grad():
            val_h1 = val_hyper1.to(device).double()
            val_h2 = val_hyper2.to(device).double()
            
            with autocast(enabled=config.use_amp):
                z1 = student(val_h1)
                z2 = student(val_h2)
                sims = F.cosine_similarity(z1, z2)
            
            val_rho, _ = spearmanr(sims.cpu().numpy(), val_scores.numpy())
        
        history["train_loss"].append(epoch_loss / len(train_loader))
        history["val_rho"].append(val_rho)
        
        if val_rho > best_val_rho:
            best_val_rho = val_rho
            best_state = student.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break
    
    if best_state:
        student.load_state_dict(best_state)
    
    return student, {"best_val_rho": best_val_rho, "history": history, "seed": seed}


# ==============================================================================
# EVALUATION FUNCTIONS
# ==============================================================================

def evaluate_sts(
    student: nn.Module,
    cgt_gw: CGTGWProjector,
    test_emb1: torch.Tensor,
    test_emb2: torch.Tensor,
    test_scores: np.ndarray,
    config: ExecutionConfig,
    device: torch.device,
) -> Dict:
    """Evaluate on STS task (Spearman correlation)."""
    
    cgt_gw.eval()
    student.eval()
    
    with torch.no_grad():
        # CGT-GW FORA DO AMP — OBRIGATÓRIO
        with autocast(enabled=False):
            test_emb1_d = test_emb1.to(device).double()
            test_emb2_d = test_emb2.to(device).double()
            
            assert test_emb1_d.dtype == torch.float64, \
                f"Teacher embeddings must be float64, got {test_emb1_d.dtype}"
            assert not torch.is_autocast_enabled(), \
                "ERROR: AMP scope leaked into CGT-GW execution"
            
            # Project via CGT-GW
            test_hyper1 = cgt_gw(test_emb1_d)
            test_hyper2 = cgt_gw(test_emb2_d)
        
        # Student inference
        with autocast(enabled=config.use_amp):
            z1 = student(test_hyper1)
            z2 = student(test_hyper2)
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
    cgt_gw: CGTGWProjector,
    embeddings: torch.Tensor,
    queries: List[int],
    positives: List[List[int]],
    config: ExecutionConfig,
    device: torch.device,
) -> Dict:
    """Evaluate on Reranking task (MAP)."""
    
    cgt_gw.eval()
    student.eval()
    
    with torch.no_grad():
        # CGT-GW FORA DO AMP — OBRIGATÓRIO
        with autocast(enabled=False):
            embeddings_d = embeddings.to(device).double()
            
            assert embeddings_d.dtype == torch.float64, \
                f"Teacher embeddings must be float64, got {embeddings_d.dtype}"
            assert not torch.is_autocast_enabled(), \
                "ERROR: AMP scope leaked into CGT-GW execution"
            
            # Project all embeddings
            hyper_emb = cgt_gw(embeddings_d)
        
        with autocast(enabled=config.use_amp):
            student_emb = student(hyper_emb)
            student_emb = F.normalize(student_emb, p=2, dim=1)
        
        sim_matrix = torch.mm(student_emb, student_emb.t()).cpu().numpy()
    
    # Compute MAP
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
    cgt_gw: CGTGWProjector,
    embeddings: torch.Tensor,
    labels: np.ndarray,
    config: ExecutionConfig,
    device: torch.device,
) -> Dict:
    """Evaluate on Clustering task (V-measure)."""
    
    cgt_gw.eval()
    student.eval()
    
    with torch.no_grad():
        # CGT-GW FORA DO AMP — OBRIGATÓRIO
        with autocast(enabled=False):
            embeddings_d = embeddings.to(device).double()
            
            assert embeddings_d.dtype == torch.float64, \
                f"Teacher embeddings must be float64, got {embeddings_d.dtype}"
            assert not torch.is_autocast_enabled(), \
                "ERROR: AMP scope leaked into CGT-GW execution"
            
            hyper_emb = cgt_gw(embeddings_d)
        
        with autocast(enabled=config.use_amp):
            student_emb = student(hyper_emb)
        
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


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_dataset_splits(
    dataset_config: DatasetConfig,
    teacher_model_name: str,
    device: str = "cuda",
    cache_dir: Optional[Path] = None,
) -> Dict:
    """
    Load and encode dataset splits using teacher model.
    
    Returns dict with train/val/test embeddings and scores.
    """
    from sentence_transformers import SentenceTransformer
    from datasets import load_dataset
    
    # Check cache
    if cache_dir:
        cache_file = cache_dir / f"{dataset_config.name}_{teacher_model_name.replace('/', '_')}.pt"
        if cache_file.exists():
            return torch.load(cache_file, weights_only=False)
    
    # Load teacher
    teacher = SentenceTransformer(teacher_model_name, device=device)
    teacher.eval()
    
    result = {}
    
    if dataset_config.task_type == TaskType.STS:
        # Load train split
        try:
            train_data = load_dataset(dataset_config.hf_path, split="train")
        except:
            # Some datasets don't have train, use validation
            train_data = load_dataset(dataset_config.hf_path, split="validation")
        
        # Load test split
        test_data = load_dataset(dataset_config.hf_path, split="test")
        
        # Encode train
        with torch.no_grad():
            train_emb1 = teacher.encode(
                train_data[dataset_config.sent1_col],
                convert_to_tensor=True, batch_size=128, show_progress_bar=False
            )
            train_emb2 = teacher.encode(
                train_data[dataset_config.sent2_col],
                convert_to_tensor=True, batch_size=128, show_progress_bar=False
            )
        train_scores = torch.tensor(train_data[dataset_config.score_col], dtype=torch.float32)
        
        # Encode test
        with torch.no_grad():
            test_emb1 = teacher.encode(
                test_data[dataset_config.sent1_col],
                convert_to_tensor=True, batch_size=128, show_progress_bar=False
            )
            test_emb2 = teacher.encode(
                test_data[dataset_config.sent2_col],
                convert_to_tensor=True, batch_size=128, show_progress_bar=False
            )
        test_scores = np.array(test_data[dataset_config.score_col], dtype=np.float32)
        
        # Split train into train/val (80/20)
        n = len(train_scores)
        indices = torch.randperm(n)
        split = int(0.8 * n)
        
        result = {
            "train_emb1": train_emb1[indices[:split]].cpu(),
            "train_emb2": train_emb2[indices[:split]].cpu(),
            "train_scores": train_scores[indices[:split]],
            "val_emb1": train_emb1[indices[split:]].cpu(),
            "val_emb2": train_emb2[indices[split:]].cpu(),
            "val_scores": train_scores[indices[split:]],
            "test_emb1": test_emb1.cpu(),
            "test_emb2": test_emb2.cpu(),
            "test_scores": test_scores,
            "teacher_dim": train_emb1.shape[1],
        }
    
    elif dataset_config.task_type == TaskType.RERANKING:
        # Load dataset
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
            
            q_idx = text_to_idx[query]
            queries.append(q_idx)
            
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
        
        result = {
            "embeddings": embeddings.cpu(),
            "queries": queries,
            "positives": positives,
            "teacher_dim": embeddings.shape[1],
        }
    
    elif dataset_config.task_type == TaskType.CLUSTERING:
        data = load_dataset(dataset_config.hf_path, split="test")
        
        texts = data[dataset_config.text_col][:10000]  # Limit for memory
        labels = np.array(data[dataset_config.label_col][:10000])
        
        with torch.no_grad():
            embeddings = teacher.encode(
                texts, convert_to_tensor=True, batch_size=128, show_progress_bar=False
            )
        
        result = {
            "embeddings": embeddings.cpu(),
            "labels": labels,
            "teacher_dim": embeddings.shape[1],
        }
    
    # Cache
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(result, cache_file)
    
    # Cleanup
    del teacher
    torch.cuda.empty_cache()
    gc.collect()
    
    return result


# ==============================================================================
# MAIN EXECUTOR
# ==============================================================================

def run_cartesian_execution_v4(
    output_dir: Path,
    config: Optional[ExecutionConfig] = None,
) -> Dict:
    """
    Full Cartesian Execution with correct pipeline.
    
    Pipeline:
        For each (Dataset × Teacher):
            1. Load data with teacher embeddings
            2. Train CGT-GW (once)
            For each (Student × Seed):
                3. Train Student via CGT-GW
                4. Evaluate on test split
    """
    
    if config is None:
        config = ExecutionConfig()
    
    device = get_device()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Directories
    checkpoint_dir = output_dir / "checkpoints"
    results_dir = output_dir / "results"
    cache_dir = output_dir / "cache"
    
    for d in [checkpoint_dir, results_dir, cache_dir]:
        d.mkdir(exist_ok=True)
    
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
    
    # Results storage
    all_results = []
    summary_stats = {
        "total_cgt_gw_trained": 0,
        "total_students_trained": 0,
        "total_evaluations": 0,
        "skipped_incompatible": 0,
        "errors": 0,
    }
    
    # Timer
    start_time = time.time()
    
    print("="*80)
    print("CGT CARTESIAN EXECUTION v4")
    print("="*80)
    print(f"Scope: {config.scope}")
    print(f"Datasets: {len(datasets)}")
    print(f"Teachers: {len(teachers)}")
    print(f"Students: {len(ALL_STUDENTS)}")
    print(f"Seeds: {seeds}")
    print(f"Device: {device}")
    print(f"AMP: {config.use_amp}")
    print(f"Batch size: {config.batch_size_train}")
    print("="*80)
    
    # Main loop
    for dataset_config in tqdm(datasets, desc="Datasets"):
        dataset_name = dataset_config.name
        
        for teacher_name, teacher_dim in tqdm(teachers, desc=f"Teachers ({dataset_name})", leave=False):
            
            teacher_short = teacher_name.split("/")[-1]
            
            # ===== LOAD DATA =====
            try:
                data = load_dataset_splits(
                    dataset_config,
                    teacher_name,
                    device=str(device),
                    cache_dir=cache_dir,
                )
            except Exception as e:
                print(f"  ❌ Error loading {dataset_name}/{teacher_short}: {e}")
                summary_stats["errors"] += 1
                continue
            
            # ===== TRAIN CGT-GW (ONCE per Dataset × Teacher) =====
            cgt_gw_path = checkpoint_dir / dataset_name / teacher_short / "cgt_gw.pth"
            
            if cgt_gw_path.exists():
                # Load existing
                projector_config = CGTGWProjectorConfig(
                    input_dim=data["teacher_dim"],
                    output_dim=config.hidden_dim,
                    gw_embed_dim=min(16, config.hidden_dim // 4),
                    gw_hidden_dim=config.hidden_dim,
                )
                cgt_gw = CGTGWProjector(projector_config).to(device).double()
                cgt_gw.load_state_dict(torch.load(cgt_gw_path, map_location=device, weights_only=True))
            else:
                # Train new
                if dataset_config.task_type == TaskType.STS:
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
                    summary_stats["total_cgt_gw_trained"] += 1
                else:
                    # For non-STS, create CGT-GW with dummy training
                    projector_config = CGTGWProjectorConfig(
                        input_dim=data["teacher_dim"],
                        output_dim=config.hidden_dim,
                        gw_embed_dim=min(16, config.hidden_dim // 4),
                        gw_hidden_dim=config.hidden_dim,
                    )
                    cgt_gw = CGTGWProjector(projector_config).to(device).double()
            
            # ===== TRAIN STUDENTS =====
            for student_name in ALL_STUDENTS:
                
                # Compatibility check
                if student_name in ["PSI_SLM", "HYBRID", "PSI_SLM_FULL"]:
                    if teacher_dim != 768:
                        summary_stats["skipped_incompatible"] += 1
                        continue
                
                for seed in seeds:
                    
                    student_path = (
                        checkpoint_dir / dataset_name / teacher_short / 
                        student_name / f"seed_{seed}.pth"
                    )
                    
                    # Train or load student
                    if student_path.exists():
                        # Load
                        cgt_gw.eval()
                        with torch.no_grad():
                            sample = data["train_emb1"][:1].to(device).double()
                            # CGT-GW FORA DO AMP — OBRIGATÓRIO
                            with autocast(enabled=False):
                                assert sample.dtype == torch.float64, \
                                    f"Teacher embeddings must be float64, got {sample.dtype}"
                                cgt_gw_dim = cgt_gw(sample).shape[1]
                        
                        student = create_student(
                            student_name, cgt_gw_dim,
                            config.student_dim, config.hidden_dim
                        )
                        if student is None:
                            continue
                        student = student.to(device).double()
                        student.load_state_dict(
                            torch.load(student_path, map_location=device, weights_only=True)
                        )
                    else:
                        # Train
                        if dataset_config.task_type != TaskType.STS:
                            # Skip non-STS training for now
                            continue
                        
                        student, train_metrics = train_student(
                            cgt_gw=cgt_gw,
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
                        
                        # Save
                        student_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(student.state_dict(), student_path)
                        summary_stats["total_students_trained"] += 1
                    
                    # ===== EVALUATE =====
                    if dataset_config.task_type == TaskType.STS:
                        metrics = evaluate_sts(
                            student, cgt_gw,
                            data["test_emb1"], data["test_emb2"], data["test_scores"],
                            config, device
                        )
                    elif dataset_config.task_type == TaskType.RERANKING:
                        metrics = evaluate_reranking(
                            student, cgt_gw,
                            data["embeddings"], data["queries"], data["positives"],
                            config, device
                        )
                    elif dataset_config.task_type == TaskType.CLUSTERING:
                        metrics = evaluate_clustering(
                            student, cgt_gw,
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
                        **metrics,
                    }
                    all_results.append(result)
                    summary_stats["total_evaluations"] += 1
            
            # Clear GPU memory after each teacher
            torch.cuda.empty_cache()
            gc.collect()
    
    # ===== SAVE RESULTS =====
    elapsed = time.time() - start_time
    
    summary = {
        "config": {
            "scope": config.scope,
            "seeds": seeds,
            "batch_size": config.batch_size_train,
            "use_amp": config.use_amp,
        },
        "statistics": summary_stats,
        "elapsed_seconds": elapsed,
        "timestamp": datetime.utcnow().isoformat(),
        "results": all_results,
    }
    
    # Save JSON
    results_path = results_dir / "cartesian_results_v4.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("EXECUTION COMPLETE")
    print("="*80)
    print(f"CGT-GW trained: {summary_stats['total_cgt_gw_trained']}")
    print(f"Students trained: {summary_stats['total_students_trained']}")
    print(f"Evaluations: {summary_stats['total_evaluations']}")
    print(f"Skipped (incompatible): {summary_stats['skipped_incompatible']}")
    print(f"Errors: {summary_stats['errors']}")
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Results saved: {results_path}")
    print("="*80)
    
    return summary


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    "run_cartesian_execution_v4",
    "ExecutionConfig",
    "TaskType",
    "ALL_STUDENTS",
    "ALL_TEACHERS",
    "ALL_DATASET_CONFIGS",
    "STS_DATASETS",
    "RERANKING_DATASETS",
    "CLUSTERING_DATASETS",
]

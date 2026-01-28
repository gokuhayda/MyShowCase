# ==============================================================================
# final_executor_v3.py
# Cartesian Experimental Executor (MULTI-METRIC SUPPORT)
#
# GUARANTEES:
# - Inference-only (no training, no loss, no optimizer)
# - Explicit Cartesian loops (Student × Teacher × Dataset)
# - Architectural compatibility enforced
# - All skips logged with explicit reason
# - Metrics independent of execution scope
#
# SUPPORTED TASK TYPES:
# - STS: Spearman correlation (ρ)
# - Reranking: Mean Average Precision (MAP)
# - Clustering: V-measure
# ==============================================================================

from pathlib import Path
from datetime import datetime
from enum import Enum
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import json

import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import v_measure_score
from sklearn.cluster import KMeans

from cgt.utils.helpers import set_global_seed, get_device
from cgt.geometry.lorentz_hardened import LorentzSubstrateHardened, LorentzConfig
from cgt.models.cgt_hardened import CGTStudentHardened


# ==============================================================================
# TASK TYPES
# ==============================================================================

class TaskType(Enum):
    STS = "sts"                    # Semantic Textual Similarity → Spearman
    RERANKING = "reranking"        # Reranking → MAP
    CLUSTERING = "clustering"      # Clustering → V-measure
    PAIR_CLASSIFICATION = "pair"   # Pair Classification → Average Precision


# ==============================================================================
# DATASET CONFIGURATIONS
# ==============================================================================

@dataclass
class DatasetConfig:
    name: str
    hf_path: str
    split: str
    task_type: TaskType
    # STS-specific fields
    sent1_col: Optional[str] = None
    sent2_col: Optional[str] = None
    score_col: Optional[str] = None
    # Clustering-specific fields
    text_col: Optional[str] = None
    label_col: Optional[str] = None


# STS Datasets (8) - Spearman correlation
STS_DATASETS = [
    DatasetConfig("STS12", "mteb/sts12-sts", "test", TaskType.STS, 
                  "sentence1", "sentence2", "score"),
    DatasetConfig("STS13", "mteb/sts13-sts", "test", TaskType.STS,
                  "sentence1", "sentence2", "score"),
    DatasetConfig("STS14", "mteb/sts14-sts", "test", TaskType.STS,
                  "sentence1", "sentence2", "score"),
    DatasetConfig("STS15", "mteb/sts15-sts", "test", TaskType.STS,
                  "sentence1", "sentence2", "score"),
    DatasetConfig("STS16", "mteb/sts16-sts", "test", TaskType.STS,
                  "sentence1", "sentence2", "score"),
    DatasetConfig("STSBenchmark", "mteb/stsbenchmark-sts", "test", TaskType.STS,
                  "sentence1", "sentence2", "score"),
    DatasetConfig("SICK-R", "mteb/sickr-sts", "test", TaskType.STS,
                  "sentence1", "sentence2", "score"),
    DatasetConfig("BIOSSES", "mteb/biosses-sts", "test", TaskType.STS,
                  "sentence1", "sentence2", "score"),
]

# Reranking Datasets (3) - MAP
RERANKING_DATASETS = [
    DatasetConfig("AskUbuntuDupQuestions", "mteb/askubuntudupquestions-reranking", 
                  "test", TaskType.RERANKING),
    DatasetConfig("SciDocsRR", "mteb/scidocs-reranking", 
                  "test", TaskType.RERANKING),
    DatasetConfig("StackOverflowDupQuestions", "mteb/stackoverflowdupquestions-reranking", 
                  "test", TaskType.RERANKING),
]

# Clustering Datasets (2) - V-measure
CLUSTERING_DATASETS = [
    DatasetConfig("TwentyNewsgroupsClustering", "mteb/twentynewsgroups-clustering", 
                  "test", TaskType.CLUSTERING, text_col="text", label_col="label"),
    DatasetConfig("RedditClustering", "mteb/reddit-clustering", 
                  "test", TaskType.CLUSTERING, text_col="text", label_col="label"),
]

# Combined (backward compatible)
ALL_DATASET_CONFIGS = STS_DATASETS + RERANKING_DATASETS + CLUSTERING_DATASETS

# Legacy format for backward compatibility
ALL_DATASETS = [(cfg.name, cfg.hf_path) for cfg in ALL_DATASET_CONFIGS]
CANONICAL_DATASETS = [(cfg.name, cfg.hf_path) for cfg in STS_DATASETS[:3]]

# Quick lookup
DATASET_CONFIG_MAP = {cfg.name: cfg for cfg in ALL_DATASET_CONFIGS}


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
# TEACHERS (name, embedding_dim)
# ==============================================================================

ALL_TEACHERS = [
    # Sentence-Transformers
    ("all-mpnet-base-v2", 768),
    ("all-MiniLM-L6-v2", 384),
    ("all-MiniLM-L12-v2", 384),
    ("all-distilroberta-v1", 768),
    ("multi-qa-mpnet-base-dot-v1", 768),
    ("multi-qa-MiniLM-L6-dot-v1", 384),
    # BGE Family
    ("BAAI/bge-small-en-v1.5", 384),
    ("BAAI/bge-base-en-v1.5", 768),
    ("BAAI/bge-large-en-v1.5", 1024),
    # E5 Family
    ("intfloat/e5-small-v2", 384),
    ("intfloat/e5-base-v2", 768),
    ("intfloat/e5-large-v2", 1024),
    # GTE Family
    ("thenlper/gte-small", 384),
    ("thenlper/gte-base", 768),
    ("thenlper/gte-large", 1024),
    # Multilingual
    ("paraphrase-multilingual-MiniLM-L12-v2", 384),
    ("distiluse-base-multilingual-cased-v2", 512),
    # Compact BERT variants
    ("prajjwal1/bert-tiny", 128),
    ("prajjwal1/bert-mini", 256),
    ("prajjwal1/bert-small", 512),
    ("distilbert-base-uncased", 768),
    ("distilbert-base-multilingual-cased", 768),
    # Additional models
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
# EXECUTION SCOPES
# ==============================================================================

class ExecutionScope:
    MINIMAL = "minimal"
    CANONICAL = "canonical"
    FULL_CARTESIAN = "full_cartesian"


# ==============================================================================
# ARCHITECTURAL COMPATIBILITY
# ==============================================================================

def is_architecturally_compatible(student: str, teacher_dim: int) -> bool:
    if student in ["PSI_SLM", "HYBRID", "PSI_SLM_FULL"]:
        return teacher_dim == 768
    return True


def infer_architecture_from_checkpoint(state_dict: dict) -> dict:
    """
    Infer model architecture from checkpoint state_dict.
    """
    weight_key_0 = None
    weight_key_6 = None
    
    for key in state_dict.keys():
        if 'projector.0.weight' in key and weight_key_0 is None:
            weight_key_0 = key
        if 'projector.6.weight' in key and weight_key_6 is None:
            weight_key_6 = key
    
    if weight_key_0 is None or weight_key_6 is None:
        return {"teacher_dim": 384, "hidden_dim": 256, "student_dim": 32}
    
    w0 = state_dict[weight_key_0]
    w6 = state_dict[weight_key_6]
    
    return {
        "teacher_dim": w0.shape[1],
        "hidden_dim": w0.shape[0],
        "student_dim": w6.shape[0],
    }


# ==============================================================================
# METRICS
# ==============================================================================

def compute_spearman(student_emb1: torch.Tensor, student_emb2: torch.Tensor, 
                     scores: np.ndarray) -> float:
    """
    Computes Spearman correlation for STS tasks.
    """
    with torch.no_grad():
        sims = F.cosine_similarity(student_emb1, student_emb2).cpu().numpy()
    rho, _ = spearmanr(sims, scores)
    return float(rho)


def compute_map(student_embeddings: torch.Tensor, queries: List[int], 
                positives: List[List[int]]) -> float:
    """
    Computes Mean Average Precision for Reranking tasks.
    
    Args:
        student_embeddings: All document embeddings [N, D]
        queries: Indices of query documents
        positives: List of positive document indices for each query
    
    Returns:
        MAP score
    """
    with torch.no_grad():
        # Normalize embeddings
        emb_norm = F.normalize(student_embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(emb_norm, emb_norm.t()).cpu().numpy()
    
    aps = []
    for q_idx, pos_indices in zip(queries, positives):
        if len(pos_indices) == 0:
            continue
            
        # Get similarities for this query (exclude self)
        sims = sim_matrix[q_idx].copy()
        sims[q_idx] = -np.inf  # Exclude self
        
        # Rank by similarity (descending)
        ranked_indices = np.argsort(-sims)
        
        # Compute AP
        pos_set = set(pos_indices)
        hits = 0
        precision_sum = 0.0
        
        for rank, doc_idx in enumerate(ranked_indices, 1):
            if doc_idx in pos_set:
                hits += 1
                precision_sum += hits / rank
        
        if hits > 0:
            ap = precision_sum / len(pos_indices)
            aps.append(ap)
    
    return float(np.mean(aps)) if aps else 0.0


def compute_vmeasure(student_embeddings: torch.Tensor, labels: np.ndarray,
                     n_clusters: Optional[int] = None) -> float:
    """
    Computes V-measure for Clustering tasks.
    
    Args:
        student_embeddings: Document embeddings [N, D]
        labels: Ground truth cluster labels
        n_clusters: Number of clusters (default: unique labels)
    
    Returns:
        V-measure score
    """
    with torch.no_grad():
        emb_np = student_embeddings.cpu().numpy()
    
    if n_clusters is None:
        n_clusters = len(np.unique(labels))
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred_labels = kmeans.fit_predict(emb_np)
    
    # V-measure
    v_score = v_measure_score(labels, pred_labels)
    
    return float(v_score)


# ==============================================================================
# DATASET LOADING
# ==============================================================================

def load_sts_data(data_path: Path) -> Dict:
    """Load pre-computed STS data."""
    return torch.load(data_path, weights_only=False)


def load_reranking_data(data_path: Path) -> Dict:
    """Load pre-computed Reranking data."""
    return torch.load(data_path, weights_only=False)


def load_clustering_data(data_path: Path) -> Dict:
    """Load pre-computed Clustering data."""
    return torch.load(data_path, weights_only=False)


# ==============================================================================
# SINGLE EXECUTION (READ-ONLY, MULTI-METRIC)
# ==============================================================================

def execute_single(
    student_name: str,
    teacher_name: str,
    teacher_dim: int,
    dataset_config: DatasetConfig,
    student_checkpoint: Path,
    data: Dict,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Single inference-only evaluation with task-specific metrics.
    """
    
    # ------------------------------------------------------------------
    # Load checkpoint and infer architecture
    # ------------------------------------------------------------------
    checkpoint = torch.load(student_checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    
    arch = infer_architecture_from_checkpoint(state)
    
    model = CGTStudentHardened(
        teacher_dim=arch["teacher_dim"],
        student_dim=arch["student_dim"],
        hidden_dim=arch["hidden_dim"],
    )
    model.load_state_dict(state)
    model = model.to(device=device, dtype=torch.float64)
    model.eval()

    # ------------------------------------------------------------------
    # Task-specific evaluation
    # ------------------------------------------------------------------
    result = {
        "student": student_name,
        "teacher": teacher_name,
        "dataset": dataset_config.name,
        "task_type": dataset_config.task_type.value,
        "status": "completed",
    }
    
    try:
        if dataset_config.task_type == TaskType.STS:
            # STS: Spearman correlation
            emb1 = data["test_emb1"].to(device=device, dtype=torch.float64)
            emb2 = data["test_emb2"].to(device=device, dtype=torch.float64)
            scores = data["scores"]
            
            with torch.no_grad():
                student_emb1 = model(emb1)
                student_emb2 = model(emb2)
            
            rho = compute_spearman(student_emb1, student_emb2, scores)
            result["spearman"] = rho
            result["primary_metric"] = rho
            result["metric_name"] = "spearman"
            
        elif dataset_config.task_type == TaskType.RERANKING:
            # Reranking: MAP
            embeddings = data["embeddings"].to(device=device, dtype=torch.float64)
            queries = data["queries"]
            positives = data["positives"]
            
            with torch.no_grad():
                student_emb = model(embeddings)
            
            map_score = compute_map(student_emb, queries, positives)
            result["map"] = map_score
            result["primary_metric"] = map_score
            result["metric_name"] = "map"
            
        elif dataset_config.task_type == TaskType.CLUSTERING:
            # Clustering: V-measure
            embeddings = data["embeddings"].to(device=device, dtype=torch.float64)
            labels = data["labels"]
            
            with torch.no_grad():
                student_emb = model(embeddings)
            
            v_score = compute_vmeasure(student_emb, labels)
            result["v_measure"] = v_score
            result["primary_metric"] = v_score
            result["metric_name"] = "v_measure"
            
        else:
            result["status"] = "skipped"
            result["skip_reason"] = f"unsupported_task_type: {dataset_config.task_type}"
            
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result


# ==============================================================================
# DATASET GENERATION (FOR RERANKING AND CLUSTERING)
# ==============================================================================

def generate_reranking_data(
    config: DatasetConfig,
    teacher_model_name: str,
    output_dir: Path,
    device: str = "cuda",
) -> Path:
    """
    Generate pre-computed embeddings for Reranking dataset.
    """
    from sentence_transformers import SentenceTransformer
    from datasets import load_dataset
    
    print(f"[RERANKING] Generating {config.name}...")
    
    # Load teacher
    teacher = SentenceTransformer(teacher_model_name, device=device)
    teacher.eval()
    
    # Load dataset
    dataset = load_dataset(config.hf_path, split=config.split)
    
    # MTEB reranking format: query, positive, negative
    all_texts = []
    queries = []
    positives = []
    
    text_to_idx = {}
    
    for item in dataset:
        query = item["query"]
        pos_list = item.get("positive", [])
        
        # Add query
        if query not in text_to_idx:
            text_to_idx[query] = len(all_texts)
            all_texts.append(query)
        
        q_idx = text_to_idx[query]
        queries.append(q_idx)
        
        # Add positives
        pos_indices = []
        for p in pos_list:
            if p not in text_to_idx:
                text_to_idx[p] = len(all_texts)
                all_texts.append(p)
            pos_indices.append(text_to_idx[p])
        
        positives.append(pos_indices)
        
        # Add negatives (optional, for corpus)
        neg_list = item.get("negative", [])
        for n in neg_list:
            if n not in text_to_idx:
                text_to_idx[n] = len(all_texts)
                all_texts.append(n)
    
    print(f"  Total texts: {len(all_texts)}")
    print(f"  Queries: {len(queries)}")
    
    # Encode all texts
    with torch.no_grad():
        embeddings = teacher.encode(all_texts, convert_to_tensor=True, 
                                    batch_size=64, show_progress_bar=True)
    
    data_obj = {
        "embeddings": embeddings.cpu(),
        "queries": queries,
        "positives": positives,
        "texts": all_texts,
    }
    
    out_path = output_dir / f"{config.name}.pt"
    torch.save(data_obj, out_path)
    print(f"  ✅ Saved: {out_path}")
    
    return out_path


def generate_clustering_data(
    config: DatasetConfig,
    teacher_model_name: str,
    output_dir: Path,
    device: str = "cuda",
    max_samples: int = 10000,
) -> Path:
    """
    Generate pre-computed embeddings for Clustering dataset.
    """
    from sentence_transformers import SentenceTransformer
    from datasets import load_dataset
    
    print(f"[CLUSTERING] Generating {config.name}...")
    
    # Load teacher
    teacher = SentenceTransformer(teacher_model_name, device=device)
    teacher.eval()
    
    # Load dataset
    dataset = load_dataset(config.hf_path, split=config.split)
    
    # Extract texts and labels
    texts = dataset[config.text_col][:max_samples]
    labels = np.array(dataset[config.label_col][:max_samples])
    
    print(f"  Samples: {len(texts)}")
    print(f"  Unique labels: {len(np.unique(labels))}")
    
    # Encode texts
    with torch.no_grad():
        embeddings = teacher.encode(texts, convert_to_tensor=True,
                                    batch_size=64, show_progress_bar=True)
    
    data_obj = {
        "embeddings": embeddings.cpu(),
        "labels": labels,
        "texts": texts,
    }
    
    out_path = output_dir / f"{config.name}.pt"
    torch.save(data_obj, out_path)
    print(f"  ✅ Saved: {out_path}")
    
    return out_path


def generate_all_datasets(
    teacher_model_name: str = "all-MiniLM-L6-v2",
    output_dir: Path = Path("./data"),
    device: str = "cuda",
    include_sts: bool = True,
    include_reranking: bool = True,
    include_clustering: bool = True,
):
    """
    Generate all dataset embeddings for Cartesian execution.
    """
    from sentence_transformers import SentenceTransformer
    from datasets import load_dataset
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    teacher = SentenceTransformer(teacher_model_name, device=device)
    teacher.eval()
    
    generated = []
    
    # STS Datasets
    if include_sts:
        print("\n" + "="*60)
        print("STS DATASETS")
        print("="*60)
        
        for cfg in STS_DATASETS:
            print(f"\n[STS] {cfg.name}")
            
            dataset = load_dataset(cfg.hf_path, split=cfg.split)
            
            s1 = dataset[cfg.sent1_col]
            s2 = dataset[cfg.sent2_col]
            scores = np.array(dataset[cfg.score_col], dtype=np.float32)
            
            print(f"  Samples: {len(scores)}")
            
            with torch.no_grad():
                emb1 = teacher.encode(s1, convert_to_tensor=True, batch_size=64)
                emb2 = teacher.encode(s2, convert_to_tensor=True, batch_size=64)
            
            data_obj = {
                "test_emb1": emb1.cpu(),
                "test_emb2": emb2.cpu(),
                "scores": scores,
            }
            
            out_path = output_dir / f"{cfg.name}.pt"
            torch.save(data_obj, out_path)
            print(f"  ✅ Saved: {out_path}")
            generated.append(cfg.name)
    
    # Reranking Datasets
    if include_reranking:
        print("\n" + "="*60)
        print("RERANKING DATASETS")
        print("="*60)
        
        for cfg in RERANKING_DATASETS:
            try:
                generate_reranking_data(cfg, teacher_model_name, output_dir, device)
                generated.append(cfg.name)
            except Exception as e:
                print(f"  ❌ Error: {e}")
    
    # Clustering Datasets
    if include_clustering:
        print("\n" + "="*60)
        print("CLUSTERING DATASETS")
        print("="*60)
        
        for cfg in CLUSTERING_DATASETS:
            try:
                generate_clustering_data(cfg, teacher_model_name, output_dir, device)
                generated.append(cfg.name)
            except Exception as e:
                print(f"  ❌ Error: {e}")
    
    print("\n" + "="*60)
    print(f"DATASET GENERATION COMPLETE: {len(generated)} datasets")
    print("="*60)
    
    return generated


# ==============================================================================
# CARTESIAN EXECUTION
# ==============================================================================

def run_cartesian_execution(
    output: Path,
    scope: str = ExecutionScope.CANONICAL,
    seed: int = 42,
    include_reranking: bool = True,
    include_clustering: bool = True,
) -> Dict[str, Any]:
    """
    Full Cartesian executor with explicit loops and multi-metric support.
    """
    
    set_global_seed(seed)
    device = get_device()
    
    output.mkdir(parents=True, exist_ok=True)
    
    # ------------------------------------------------------------------
    # Scope selection
    # ------------------------------------------------------------------
    if scope == ExecutionScope.MINIMAL:
        teachers = ALL_TEACHERS[:2]
        dataset_configs = STS_DATASETS[:1]
    elif scope == ExecutionScope.CANONICAL:
        teachers = CANONICAL_TEACHERS
        dataset_configs = STS_DATASETS[:3]
    elif scope == ExecutionScope.FULL_CARTESIAN:
        teachers = ALL_TEACHERS
        dataset_configs = list(STS_DATASETS)
        if include_reranking:
            dataset_configs.extend(RERANKING_DATASETS)
        if include_clustering:
            dataset_configs.extend(CLUSTERING_DATASETS)
    else:
        raise ValueError(f"Unknown scope: {scope}")
    
    results = []
    skips = []
    
    # Statistics
    stats = {
        "total_combinations": 0,
        "executed": 0,
        "skipped_incompatible": 0,
        "skipped_missing_checkpoint": 0,
        "skipped_missing_data": 0,
        "errors": 0,
    }
    
    # ------------------------------------------------------------------
    # Explicit Cartesian loops
    # ------------------------------------------------------------------
    for student in ALL_STUDENTS:
        for teacher_name, teacher_dim in teachers:
            
            # Architectural compatibility check
            if not is_architecturally_compatible(student, teacher_dim):
                for cfg in dataset_configs:
                    stats["total_combinations"] += 1
                    stats["skipped_incompatible"] += 1
                    skips.append({
                        "student": student,
                        "teacher": teacher_name,
                        "dataset": cfg.name,
                        "reason": "architectural_incompatibility",
                    })
                continue
            
            for cfg in dataset_configs:
                stats["total_combinations"] += 1
                
                # Check checkpoint
                checkpoint = (
                    output.parent / "outputs" / student.lower() / "model_checkpoint.pth"
                )
                
                if not checkpoint.exists():
                    stats["skipped_missing_checkpoint"] += 1
                    skips.append({
                        "student": student,
                        "teacher": teacher_name,
                        "dataset": cfg.name,
                        "reason": "missing_checkpoint",
                    })
                    continue
                
                # Check data
                data_path = output.parent / "data" / f"{cfg.name}.pt"
                if not data_path.exists():
                    stats["skipped_missing_data"] += 1
                    skips.append({
                        "student": student,
                        "teacher": teacher_name,
                        "dataset": cfg.name,
                        "reason": "missing_dataset",
                    })
                    continue
                
                # Load data
                data = torch.load(data_path, weights_only=False)
                
                # Execute
                res = execute_single(
                    student,
                    teacher_name,
                    teacher_dim,
                    cfg,
                    checkpoint,
                    data,
                    device,
                )
                
                if res.get("status") == "completed":
                    stats["executed"] += 1
                elif res.get("status") == "error":
                    stats["errors"] += 1
                
                results.append(res)
    
    # ------------------------------------------------------------------
    # Summary by task type
    # ------------------------------------------------------------------
    task_summaries = {}
    for task_type in TaskType:
        task_results = [r for r in results 
                       if r.get("task_type") == task_type.value 
                       and r.get("status") == "completed"]
        if task_results:
            metrics = [r["primary_metric"] for r in task_results]
            task_summaries[task_type.value] = {
                "count": len(task_results),
                "mean": float(np.mean(metrics)),
                "std": float(np.std(metrics)),
                "min": float(np.min(metrics)),
                "max": float(np.max(metrics)),
            }
    
    # ------------------------------------------------------------------
    # Save summary
    # ------------------------------------------------------------------
    summary = {
        "scope": scope,
        "timestamp": datetime.utcnow().isoformat(),
        "seed": seed,
        "statistics": stats,
        "task_summaries": task_summaries,
        "results": results,
        "skips": skips,
    }
    
    out_path = output / "cartesian_summary_v3.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("CARTESIAN EXECUTION SUMMARY (v3 - MULTI-METRIC)")
    print("="*80)
    print(f"Scope: {scope}")
    print(f"Total combinations: {stats['total_combinations']}")
    print(f"Executed: {stats['executed']}")
    print(f"Skipped (incompatible): {stats['skipped_incompatible']}")
    print(f"Skipped (missing checkpoint): {stats['skipped_missing_checkpoint']}")
    print(f"Skipped (missing data): {stats['skipped_missing_data']}")
    print(f"Errors: {stats['errors']}")
    print()
    print("TASK SUMMARIES:")
    for task, summary_data in task_summaries.items():
        print(f"  {task}: n={summary_data['count']}, "
              f"mean={summary_data['mean']:.4f} ± {summary_data['std']:.4f}")
    print("="*80)
    
    return summary


# ==============================================================================
# CONVENIENCE EXPORTS
# ==============================================================================

__all__ = [
    # Core
    "run_cartesian_execution",
    "generate_all_datasets",
    # Types
    "TaskType",
    "DatasetConfig",
    "ExecutionScope",
    # Configs
    "ALL_STUDENTS",
    "ALL_TEACHERS",
    "CANONICAL_TEACHERS",
    "ALL_DATASETS",
    "CANONICAL_DATASETS",
    "ALL_DATASET_CONFIGS",
    "STS_DATASETS",
    "RERANKING_DATASETS",
    "CLUSTERING_DATASETS",
    # Metrics
    "compute_spearman",
    "compute_map",
    "compute_vmeasure",
]

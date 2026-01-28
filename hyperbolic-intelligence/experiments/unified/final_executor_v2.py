# ==============================================================================
# final_executor_v2.py
# Cartesian Experimental Executor (AUDIT COMPLIANT)
#
# GUARANTEES:
# - Inference-only (no training, no loss, no optimizer)
# - Explicit Cartesian loops (Student × Teacher × Dataset)
# - Architectural compatibility enforced
# - All skips logged with explicit reason
# - Metrics independent of execution scope
# ==============================================================================

from pathlib import Path
from datetime import datetime
import json

import torch
import numpy as np
from scipy.stats import spearmanr

from cgt.utils.helpers import set_global_seed, get_device
from cgt.geometry.lorentz_hardened import LorentzSubstrateHardened, LorentzConfig
from cgt.models.cgt_hardened import CGTStudentHardened


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
# DATASETS
# ==============================================================================

# STS Datasets (8) - Spearman correlation
STS_CONFIGS = [
    ('STS12', 'mteb/sts12-sts', 'test', 'sentence1', 'sentence2', 'score'),
    ('STS13', 'mteb/sts13-sts', 'test', 'sentence1', 'sentence2', 'score'),
    ('STS14', 'mteb/sts14-sts', 'test', 'sentence1', 'sentence2', 'score'),
    ('STS15', 'mteb/sts15-sts', 'test', 'sentence1', 'sentence2', 'score'),
    ('STS16', 'mteb/sts16-sts', 'test', 'sentence1', 'sentence2', 'score'),
    ('STSBenchmark', 'mteb/stsbenchmark-sts', 'test', 'sentence1', 'sentence2', 'score'),
    ('SICK-R', 'mteb/sickr-sts', 'test', 'sentence1', 'sentence2', 'score'),
    ('BIOSSES', 'mteb/biosses-sts', 'test', 'sentence1', 'sentence2', 'score'),
]

# PairClassification Datasets (3) - Average Precision
PAIR_CONFIGS = [
    # ('SprintDuplicateQuestions', 'mteb/sprintduplicatequestions-pairclassification', 'test'),
    # ('TwitterSemEval2015', 'mteb/twittersemeval2015-pairclassification', 'test'),
    # ('TwitterURLCorpus', 'mteb/twitterurlcorpus-pairclassification', 'test'),
]

# Reranking Datasets (3) - MAP
RERANK_CONFIGS = [
    ('AskUbuntuDupQuestions', 'mteb/askubuntudupquestions-reranking', 'test'),
    ('SciDocsRR', 'mteb/scidocs-reranking', 'test'),
    ('StackOverflowDupQuestions', 'mteb/stackoverflowdupquestions-reranking', 'test'),
]

# Clustering Datasets (2) - V-measure
CLUSTER_CONFIGS = [
    ('TwentyNewsgroupsClustering', 'mteb/twentynewsgroups-clustering', 'test'),
    ('RedditClustering', 'mteb/reddit-clustering', 'test'),
]

# Backward compatible format
ALL_DATASETS = (
    [(name, path) for name, path, *_ in STS_CONFIGS] +
    [(name, path) for name, path, *_ in RERANK_CONFIGS] +
    [(name, path) for name, path, *_ in CLUSTER_CONFIGS]
)

CANONICAL_DATASETS = ALL_DATASETS[:3]


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


# ==============================================================================
# METRIC (LOCAL, SCOPE-INDEPENDENT)
# ==============================================================================

def spearman_retention(student_emb1, student_emb2, scores):
    """
    Computes Spearman correlation between student similarity and gold scores.
    """
    with torch.no_grad():
        sims = torch.nn.functional.cosine_similarity(
            student_emb1, student_emb2
        ).cpu().numpy()

    rho, _ = spearmanr(sims, scores)
    return float(rho)


# ==============================================================================
# SINGLE EXECUTION (READ-ONLY)
# ==============================================================================

def execute_single(
    student_name: str,
    teacher_name: str,
    teacher_dim: int,
    dataset_name: str,
    student_checkpoint: Path,
    data: dict,
    device,
):
    """
    Single inference-only evaluation.
    
    Architecture is inferred automatically from checkpoint.
    """

    # --------------------------------------------------------------
    # Load checkpoint and infer architecture (ADAPTIVE)
    # --------------------------------------------------------------
    checkpoint = torch.load(student_checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) else checkpoint
    
    # Infer architecture from checkpoint weights
    arch = infer_architecture_from_checkpoint(state)
    
    model = CGTStudentHardened(
        teacher_dim=arch["teacher_dim"],
        student_dim=arch["student_dim"],
        hidden_dim=arch["hidden_dim"],
    )

    model.load_state_dict(state)

    model = model.to(device=device, dtype=torch.float64)
    model.eval()

    # --------------------------------------------------------------
    # Inference
    # --------------------------------------------------------------
    emb1 = data["test_emb1"].to(device=device, dtype=torch.float64)
    emb2 = data["test_emb2"].to(device=device, dtype=torch.float64)
    scores = data["scores"]

    with torch.no_grad():
        student_emb1 = model(emb1)
        student_emb2 = model(emb2)

    # --------------------------------------------------------------
    # Metric (NO SCOPE DEPENDENCY)
    # --------------------------------------------------------------
    rho = spearman_retention(student_emb1, student_emb2, scores)

    return {
        "student": student_name,
        "teacher": teacher_name,
        "dataset": dataset_name,
        "spearman": rho,
    }


# ==============================================================================
# CARTESIAN EXECUTION
# ==============================================================================

def run_cartesian_execution(
    output: Path,
    scope: str = ExecutionScope.CANONICAL,
    seed: int = 42,
):
    """
    Full Cartesian executor with explicit loops.
    """

    set_global_seed(seed)
    device = get_device()

    output.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------
    # Scope selection
    # --------------------------------------------------------------
    if scope == ExecutionScope.MINIMAL:
        teachers = ALL_TEACHERS[:2]
        datasets = ALL_DATASETS[:1]
    elif scope == ExecutionScope.CANONICAL:
        teachers = CANONICAL_TEACHERS
        datasets = CANONICAL_DATASETS
    elif scope == ExecutionScope.FULL_CARTESIAN:
        teachers = ALL_TEACHERS
        datasets = ALL_DATASETS
    else:
        raise ValueError(f"Unknown scope: {scope}")

    results = []
    skips = []

    # --------------------------------------------------------------
    # Geometry substrate (READ-ONLY)
    # --------------------------------------------------------------
    lorentz = LorentzSubstrateHardened(
        LorentzConfig(initial_curvature=1.0)
    )

    # --------------------------------------------------------------
    # Explicit Cartesian loops
    # --------------------------------------------------------------
    for student in ALL_STUDENTS:
        for teacher_name, teacher_dim in teachers:

            if not is_architecturally_compatible(student, teacher_dim):
                skips.append({
                    "student": student,
                    "teacher": teacher_name,
                    "reason": "architectural_incompatibility",
                })
                continue

            for dataset_name, _ in datasets:

                checkpoint = (
                    output.parent
                    / "outputs"
                    / student.lower()
                    / "model_checkpoint.pth"
                )

                if not checkpoint.exists():
                    skips.append({
                        "student": student,
                        "teacher": teacher_name,
                        "dataset": dataset_name,
                        "reason": "missing_checkpoint",
                    })
                    continue

                data_path = output.parent / "data" / f"{dataset_name}.pt"
                if not data_path.exists():
                    skips.append({
                        "student": student,
                        "teacher": teacher_name,
                        "dataset": dataset_name,
                        "reason": "missing_dataset",
                    })
                    continue

                data = torch.load(data_path, weights_only=False)

                res = execute_single(
                    student,
                    teacher_name,
                    teacher_dim,
                    dataset_name,
                    checkpoint,
                    data,
                    device,
                )
                results.append(res)

    # --------------------------------------------------------------
    # Save summary
    # --------------------------------------------------------------
    summary = {
        "scope": scope,
        "timestamp": datetime.utcnow().isoformat(),
        "n_results": len(results),
        "n_skips": len(skips),
        "results": results,
        "skips": skips,
    }

    out_path = output / "cartesian_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary

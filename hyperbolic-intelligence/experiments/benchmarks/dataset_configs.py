# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
Dataset Configurations for MTEB Evaluation
==========================================

Structured dataset configurations for different task types.

Author: Éric Gustavo Reis de Sena
Date: January 2026
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, List


class TaskType(Enum):
    """MTEB task types."""
    STS = auto()
    RERANKING = auto()
    CLUSTERING = auto()
    CLASSIFICATION = auto()
    PAIR_CLASSIFICATION = auto()
    RETRIEVAL = auto()


@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""
    name: str
    hf_path: str
    train_split: str
    test_split: str
    task_type: TaskType
    text_col: Optional[str] = None
    label_col: Optional[str] = None
    sentence1_col: Optional[str] = None
    sentence2_col: Optional[str] = None
    score_col: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
#                    STS DATASETS
# ═══════════════════════════════════════════════════════════════════════════════

STS_DATASETS = [
    DatasetConfig("STSBenchmark", "mteb/stsbenchmark-sts",
                  "train", "test", TaskType.STS,
                  sentence1_col="sentence1", sentence2_col="sentence2", score_col="score"),
    DatasetConfig("STS12", "mteb/sts12-sts",
                  "train", "test", TaskType.STS),
    DatasetConfig("STS13", "mteb/sts13-sts",
                  "train", "test", TaskType.STS),
    DatasetConfig("STS14", "mteb/sts14-sts",
                  "train", "test", TaskType.STS),
    DatasetConfig("STS15", "mteb/sts15-sts",
                  "train", "test", TaskType.STS),
    DatasetConfig("STS16", "mteb/sts16-sts",
                  "train", "test", TaskType.STS),
    DatasetConfig("SICK-R", "mteb/sickr-sts",
                  "train", "test", TaskType.STS),
    DatasetConfig("BIOSSES", "mteb/biosses-sts",
                  "train", "test", TaskType.STS),
]


# ═══════════════════════════════════════════════════════════════════════════════
#                    RERANKING DATASETS
# ═══════════════════════════════════════════════════════════════════════════════

RERANKING_DATASETS = [
    DatasetConfig("AskUbuntuDupQuestions", "mteb/askubuntudupquestions-reranking",
                  "train", "test", TaskType.RERANKING),
    DatasetConfig("SciDocsRR", "mteb/scidocs-reranking",
                  "train", "test", TaskType.RERANKING),
    DatasetConfig("StackOverflowDupQuestions", "mteb/stackoverflowdupquestions-reranking",
                  "train", "test", TaskType.RERANKING),
]


# ═══════════════════════════════════════════════════════════════════════════════
#                    CLUSTERING DATASETS
# ═══════════════════════════════════════════════════════════════════════════════

CLUSTERING_DATASETS = [
    DatasetConfig("TwentyNewsgroupsClustering", "mteb/twentynewsgroups-clustering",
                  "train", "test", TaskType.CLUSTERING,
                  text_col="text", label_col="label"),
    DatasetConfig("RedditClustering", "mteb/reddit-clustering",
                  "train", "test", TaskType.CLUSTERING,
                  text_col="text", label_col="label"),
    DatasetConfig("StackExchangeClustering", "mteb/stackexchange-clustering",
                  "train", "test", TaskType.CLUSTERING,
                  text_col="text", label_col="label"),
]


# ═══════════════════════════════════════════════════════════════════════════════
#                    ALL DATASETS
# ═══════════════════════════════════════════════════════════════════════════════

ALL_DATASETS = STS_DATASETS + RERANKING_DATASETS + CLUSTERING_DATASETS


def get_datasets_by_type(task_type: TaskType) -> List[DatasetConfig]:
    """Get all datasets for a specific task type."""
    return [d for d in ALL_DATASETS if d.task_type == task_type]


def get_dataset_names_by_type(task_type: TaskType) -> List[str]:
    """Get dataset names for a specific task type."""
    return [d.name for d in get_datasets_by_type(task_type)]

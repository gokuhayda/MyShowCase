"""
dl_trainer: Deep Learning Trainer com Callback Pattern.

Exports principais:
    - Trainer: Motor de treino
    - Callbacks: EarlyStopping, ModelCheckpoint, etc.
"""

from .trainer import Trainer
from .callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    MetricsLogger,
    LearningRateScheduler
)
from .protocols import Callback

__version__ = "0.1.0"
__all__ = [
    "Trainer",
    "Callback",
    "EarlyStopping",
    "ModelCheckpoint",
    "MetricsLogger",
    "LearningRateScheduler",
]

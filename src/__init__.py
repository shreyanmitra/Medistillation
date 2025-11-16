"""
Med-Distillation Source Package

This package contains the core components for medical LLM distillation.

Modules:
- DataLoader: Dataset preparation and loading
- DistillationMethods: Implementation of distillation algorithms
- Trainer: Training loop and evaluation
"""

__version__ = "1.0.0"
__author__ = "CSE 493S Team"

from .DataLoader import create_train_val_dataloaders
from .DistillationMethods import create_distillation_method
from .Trainer import Trainer, TrainingConfig

__all__ = [
    "create_train_val_dataloaders",
    "create_distillation_method",
    "Trainer",
    "TrainingConfig",
]

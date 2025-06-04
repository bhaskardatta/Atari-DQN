"""
Training infrastructure with adaptation detection.
"""

from .trainer import AdaptiveTrainer
from .adaptation import AdaptationDetector

__all__ = [
    'AdaptiveTrainer',
    'AdaptationDetector'
]

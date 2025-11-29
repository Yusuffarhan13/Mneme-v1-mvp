"""
Coconut Training Package
Fine-tune Qwen for latent space reasoning using the Coconut method.
"""

from .data_processor import DataProcessor, CoconutExample
from .coconut_model import CoconutQwen, CoconutTrainer

__all__ = [
    "DataProcessor",
    "CoconutExample",
    "CoconutQwen",
    "CoconutTrainer",
]

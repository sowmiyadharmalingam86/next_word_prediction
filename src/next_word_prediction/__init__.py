"""Next-word prediction package."""

from .generate import GenerationResult, generate_text
from .model import ModelConfig, build_model, perplexity_from_loss
from .train import main as train_main, train

__all__ = [
    "GenerationResult",
    "generate_text",
    "ModelConfig",
    "build_model",
    "perplexity_from_loss",
    "train_main",
    "train",
]

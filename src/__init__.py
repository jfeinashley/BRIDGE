"""Cross-Modal VLM with Bidirectional Cross-Attention"""

from .models import CrossModalVLM, CrossModalInteractionLayer, CaptionDecoder, VisionEncoder, TextEncoder
from .training import VLMTrainer, VLMLosses
from .evaluation import compute_retrieval_metrics, evaluate_retrieval, evaluate_retrieval_ddp

__version__ = "0.1.0"

__all__ = [
    "CrossModalVLM",
    "CrossModalInteractionLayer",
    "CaptionDecoder",
    "VisionEncoder",
    "TextEncoder",
    "VLMTrainer",
    "VLMLosses",
    "compute_retrieval_metrics",
    "evaluate_retrieval",
    "evaluate_retrieval_ddp",
]



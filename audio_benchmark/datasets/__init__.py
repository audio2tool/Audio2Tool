"""
Dataset Loaders for Audio Tool Calling Benchmark

This module provides a registry-based system for dataset loaders.
To add support for a new dataset format, create a class inheriting
from BaseDataset and register it using the @register_dataset decorator.

Available Datasets:
- release: Any tier of the public Audio2Tool dataset release
  (https://huggingface.co/datasets/RVtech/Audio2Tool) — recommended
- tier1: Direct tool calling queries (4,560 samples)
- tier2: Parametric queries with additional tool calls (4,560 samples)
- tier3: Multi-intent queries with multiple tools (4,560 samples)
- tier4: Implicit queries requiring inference (4,560 samples)
- tier5: Needle queries with distracting context (954 samples)
- tier6: Correction queries fixing previous calls (4,560 samples)
- tier7: Multi-turn conversational queries (4,560 samples)
- tier9: Gemma3 noise ablation dataset
- tier10: Kimi noise ablation dataset
- tier11: Qwen3 noise ablation dataset
"""

from .base import (
    BaseDataset, 
    register_dataset, 
    get_dataset, 
    list_datasets,
    DATASET_REGISTRY,
    QuerySample
)
from .tier1_dataset import Tier1Dataset
from .tier2_dataset import Tier2Dataset
from .tier3_dataset import Tier3Dataset
from .tier4_dataset import Tier4Dataset
from .tier5_dataset import Tier5Dataset
from .tier6_dataset import Tier6Dataset
from .tier7_dataset import Tier7Dataset
from .tier8_dataset import Tier8Dataset
from .tier9_dataset import Tier9Dataset
from .tier10_dataset import Tier10Dataset
from .tier11_dataset import Tier11Dataset
from .release_dataset import ReleaseDataset

__all__ = [
    'BaseDataset',
    'register_dataset',
    'get_dataset',
    'list_datasets',
    'DATASET_REGISTRY',
    'QuerySample',
    'Tier1Dataset',
    'Tier2Dataset',
    'Tier3Dataset',
    'Tier4Dataset',
    'Tier5Dataset',
    'Tier6Dataset',
    'Tier7Dataset',
    'Tier8Dataset',
    'Tier9Dataset',
    'Tier10Dataset',
    'Tier11Dataset',
    'ReleaseDataset',
]

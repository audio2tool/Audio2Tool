"""
Audio Tool Calling Benchmark

A modular and extensible framework for benchmarking audio-language models
on tool calling tasks.

Features:
- Registry-based model system for easy extension
- Support for multiple dataset formats
- Comprehensive evaluation metrics
- Configuration-driven benchmark runs

Quick Start:
    from audio_benchmark import get_model, get_dataset, BenchmarkEvaluator
    
    # Initialize model and dataset
    model = get_model("qwen2-audio")
    dataset = get_dataset("tier1", data_dir="/path/to/data")
    dataset.load()
    
    # Run evaluation
    evaluator = BenchmarkEvaluator(model, dataset)
    metrics = evaluator.run()
    print(metrics)

Adding New Models:
    from audio_benchmark.models import BaseAudioModel, register_model
    
    @register_model("my-model")
    class MyModel(BaseAudioModel):
        def load_model(self):
            ...
        def process_audio(self, audio_path):
            ...
        def generate(self, audio_path, tools, **kwargs):
            ...

Adding New Datasets:
    from audio_benchmark.datasets import BaseDataset, register_dataset
    
    @register_dataset("my-dataset")
    class MyDataset(BaseDataset):
        def load(self):
            ...
        def get_tools_schema(self):
            ...
"""

__version__ = "0.1.0"
__author__ = "Audio Benchmark Team"

# Import main components for easy access
from .models import (
    BaseAudioModel,
    register_model,
    get_model,
    list_models,
    MODEL_REGISTRY,
)

from .datasets import (
    BaseDataset,
    register_dataset,
    get_dataset,
    list_datasets,
    DATASET_REGISTRY,
    QuerySample,
)

from .evaluation import (
    BenchmarkEvaluator,
    EvaluationResult,
    ToolCallingMetrics,
    compute_tool_accuracy,
    compute_parameter_accuracy,
)

from .utils import (
    load_config,
    save_config,
    BenchmarkConfig,
    setup_logging,
)

__all__ = [
    # Version
    '__version__',
    
    # Models
    'BaseAudioModel',
    'register_model',
    'get_model',
    'list_models',
    'MODEL_REGISTRY',
    
    # Datasets
    'BaseDataset',
    'register_dataset',
    'get_dataset',
    'list_datasets',
    'DATASET_REGISTRY',
    'QuerySample',
    
    # Evaluation
    'BenchmarkEvaluator',
    'EvaluationResult',
    'ToolCallingMetrics',
    'compute_tool_accuracy',
    'compute_parameter_accuracy',
    
    # Utils
    'load_config',
    'save_config',
    'BenchmarkConfig',
    'setup_logging',
]

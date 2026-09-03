"""
Base Dataset Class and Dataset Registry

This module defines the abstract base class for all dataset loaders
and provides a registry system for easy dataset management.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Iterator, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

# Global dataset registry
DATASET_REGISTRY: Dict[str, type] = {}


def register_dataset(name: str):
    """
    Decorator to register a dataset class in the registry.
    
    Usage:
        @register_dataset("my_dataset")
        class MyDataset(BaseDataset):
            ...
    """
    def decorator(cls):
        if name in DATASET_REGISTRY:
            logger.warning(f"Dataset '{name}' already registered. Overwriting.")
        DATASET_REGISTRY[name] = cls
        cls.dataset_name = name
        return cls
    return decorator


def get_dataset(name: str, **kwargs) -> 'BaseDataset':
    """
    Get a dataset instance by name from the registry.
    
    Args:
        name: Dataset name as registered
        **kwargs: Arguments to pass to the dataset constructor
        
    Returns:
        Instantiated dataset
        
    Raises:
        ValueError: If dataset not found in registry
    """
    if name not in DATASET_REGISTRY:
        available = list(DATASET_REGISTRY.keys())
        raise ValueError(f"Dataset '{name}' not found. Available: {available}")
    return DATASET_REGISTRY[name](**kwargs)


def list_datasets() -> List[str]:
    """List all registered dataset names."""
    return list(DATASET_REGISTRY.keys())


@dataclass
class QuerySample:
    """
    A single query sample from the dataset.
    
    Attributes:
        query_idx: Unique query index
        query_text: The text of the query/command
        tool_name: Expected tool name (ground truth) - primary tool
        tool_call: Expected full tool call string (ground truth)
        extracted_params: Expected parameters as dict (ground truth)
        audio_paths: List of paths to audio files for this query
        domain: Domain category (smart_car, smart_home, wearables)
        category: Sub-category within domain
        tier: Dataset tier (tier1_direct, tier7_multiturn, etc.)
        additional_tool_calls: List of additional tool names required (for multi-tool queries)
        metadata: Additional metadata
    """
    query_idx: int
    query_text: str
    tool_name: str
    tool_call: str
    extracted_params: Dict[str, Any]
    audio_paths: List[str]
    domain: str = ""
    category: str = ""
    tier: str = ""
    additional_tool_calls: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def num_speakers(self) -> int:
        """Number of audio samples (speakers) for this query."""
        return len(self.audio_paths)
    
    @property
    def all_expected_tools(self) -> List[str]:
        """Get all expected tools (primary + additional)."""
        tools = [self.tool_name]
        if self.additional_tool_calls:
            tools.extend(self.additional_tool_calls)
        return tools
    
    @property
    def num_expected_tools(self) -> int:
        """Number of tools expected for this query."""
        return len(self.all_expected_tools)
    
    @property
    def is_multi_tool(self) -> bool:
        """Check if this query requires multiple tools."""
        return len(self.additional_tool_calls) > 0
    
    def get_audio_path(self, speaker_idx: int = 0) -> str:
        """Get audio path for a specific speaker index."""
        if speaker_idx >= len(self.audio_paths):
            raise IndexError(f"Speaker index {speaker_idx} out of range. "
                           f"Only {len(self.audio_paths)} speakers available.")
        return self.audio_paths[speaker_idx]


class BaseDataset(ABC):
    """
    Abstract base class for dataset loaders.
    
    All dataset loaders must implement this interface to be compatible
    with the benchmark system.
    
    Attributes:
        dataset_name: Name of the dataset (set by @register_dataset)
        data_dir: Root directory containing the dataset
        samples: List of loaded query samples
    """
    
    dataset_name: str = "base"
    
    def __init__(
        self,
        data_dir: str,
        filter_domain: Optional[str] = None,
        filter_category: Optional[str] = None,
        max_samples: Optional[int] = None,
        speaker_idx: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize the dataset loader.
        
        Args:
            data_dir: Root directory containing the dataset
            filter_domain: Only include samples from this domain
            filter_category: Only include samples from this category
            max_samples: Maximum number of samples to load
            speaker_idx: If set, only use this speaker index for each query
            **kwargs: Additional dataset-specific arguments
        """
        self.data_dir = Path(data_dir)
        self.filter_domain = filter_domain
        self.filter_category = filter_category
        self.max_samples = max_samples
        self.speaker_idx = speaker_idx
        self.samples: List[QuerySample] = []
        self._loaded = False
        
    @abstractmethod
    def load(self) -> None:
        """
        Load the dataset from disk.
        
        This should populate self.samples with QuerySample objects.
        """
        pass
    
    @abstractmethod
    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """
        Get the schema of available tools.
        
        Returns:
            List of tool definitions with name, description, and parameters
        """
        pass
    
    def __len__(self) -> int:
        """Return number of loaded samples."""
        return len(self.samples)
    
    def __iter__(self) -> Iterator[QuerySample]:
        """Iterate over samples."""
        return iter(self.samples)
    
    def __getitem__(self, idx: int) -> QuerySample:
        """Get sample by index."""
        return self.samples[idx]
    
    def filter_by_domain(self, domain: str) -> List[QuerySample]:
        """Get samples filtered by domain."""
        return [s for s in self.samples if s.domain == domain]
    
    def filter_by_category(self, category: str) -> List[QuerySample]:
        """Get samples filtered by category."""
        return [s for s in self.samples if s.category == category]
    
    def filter_by_tool(self, tool_name: str) -> List[QuerySample]:
        """Get samples filtered by tool name."""
        return [s for s in self.samples if s.tool_name == tool_name]
    
    def get_unique_domains(self) -> List[str]:
        """Get list of unique domains in the dataset."""
        return list(set(s.domain for s in self.samples))
    
    def get_unique_categories(self) -> List[str]:
        """Get list of unique categories in the dataset."""
        return list(set(s.category for s in self.samples))
    
    def get_unique_tools(self) -> List[str]:
        """Get list of unique tool names in the dataset."""
        return list(set(s.tool_name for s in self.samples))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        domains = {}
        categories = {}
        tools = {}
        
        for sample in self.samples:
            domains[sample.domain] = domains.get(sample.domain, 0) + 1
            categories[sample.category] = categories.get(sample.category, 0) + 1
            tools[sample.tool_name] = tools.get(sample.tool_name, 0) + 1
        
        total_audio = sum(s.num_speakers for s in self.samples)
        
        return {
            "total_queries": len(self.samples),
            "total_audio_files": total_audio,
            "domains": domains,
            "categories": categories,
            "tools": tools,
            "unique_domains": len(domains),
            "unique_categories": len(categories),
            "unique_tools": len(tools),
        }
    
    @property
    def is_loaded(self) -> bool:
        """Check if dataset is loaded."""
        return self._loaded
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(data_dir={self.data_dir}, samples={len(self.samples)})"

"""
Base Audio Model Class and Model Registry

This module defines the abstract base class for all audio models
and provides a registry system for easy model management.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import torch
import logging

logger = logging.getLogger(__name__)

# Global model registry
MODEL_REGISTRY: Dict[str, type] = {}


def register_model(name: str):
    """
    Decorator to register a model class in the registry.
    
    Usage:
        @register_model("my_model")
        class MyModel(BaseAudioModel):
            ...
    """
    def decorator(cls):
        if name in MODEL_REGISTRY:
            logger.warning(f"Model '{name}' already registered. Overwriting.")
        MODEL_REGISTRY[name] = cls
        cls.model_name = name
        return cls
    return decorator


def get_model(name: str, **kwargs) -> 'BaseAudioModel':
    """
    Get a model instance by name from the registry.
    
    Args:
        name: Model name as registered
        **kwargs: Arguments to pass to the model constructor
        
    Returns:
        Instantiated model
        
    Raises:
        ValueError: If model not found in registry
    """
    if name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Model '{name}' not found. Available: {available}")
    return MODEL_REGISTRY[name](**kwargs)


def list_models() -> List[str]:
    """List all registered model names."""
    return list(MODEL_REGISTRY.keys())


@dataclass
class ModelOutput:
    """Standardized output from model inference."""
    tool_name: str  # Predicted tool name (primary tool, e.g., "setZoneTemperature")
    tool_call: str  # Full tool call string (e.g., "setZoneTemperature(zone='front')")
    parameters: Dict[str, Any]  # Extracted parameters as dict (for primary tool)
    raw_output: str  # Raw model output before parsing
    confidence: Optional[float] = None  # Optional confidence score
    latency_ms: Optional[float] = None  # Inference latency
    # Multi-tool support
    all_tool_calls: List[Dict[str, Any]] = field(default_factory=list)  # All tool calls returned
    
    @property
    def all_tool_names(self) -> List[str]:
        """Get all predicted tool names."""
        if self.all_tool_calls:
            return [tc.get('name', '') for tc in self.all_tool_calls]
        elif self.tool_name:
            return [self.tool_name]
        return []
    
    @property
    def num_tool_calls(self) -> int:
        """Number of tool calls returned."""
        return len(self.all_tool_calls) if self.all_tool_calls else (1 if self.tool_name else 0)
    

class BaseAudioModel(ABC):
    """
    Abstract base class for audio models used in tool calling benchmark.
    
    All model adapters must implement this interface to be compatible
    with the benchmark system.
    
    Attributes:
        model_name: Name of the model (set by @register_model)
        device: Device to run inference on
        config: Model-specific configuration
    """
    
    model_name: str = "base"
    
    def __init__(
        self,
        device: str = "cuda",
        model_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize the model.
        
        Args:
            device: Device to run on ("cuda", "cpu", or specific GPU)
            model_path: Path to model weights or HuggingFace model ID
            config: Model-specific configuration
            **kwargs: Additional model-specific arguments
        """
        self.device = device
        self.model_path = model_path
        self.config = config or {}
        self.model = None
        self.processor = None
        self._is_loaded = False
        
    @abstractmethod
    def load_model(self) -> None:
        """
        Load model weights and processor.
        
        This should be called before inference. Models should handle
        their own device placement.
        """
        pass
    
    @abstractmethod
    def process_audio(self, audio_path: str) -> Any:
        """
        Process audio file into model-specific input format.
        
        Args:
            audio_path: Path to audio file (WAV format expected)
            
        Returns:
            Processed audio ready for model input
        """
        pass
    
    @abstractmethod
    def generate(
        self,
        audio_path: str,
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> ModelOutput:
        """
        Generate tool call from audio input.
        
        Args:
            audio_path: Path to audio file
            tools: List of available tools with their schemas
            system_prompt: Optional system prompt for the model
            **kwargs: Additional generation parameters
            
        Returns:
            ModelOutput with predicted tool call and parameters
        """
        pass
    
    def parse_tool_call(self, raw_output: str) -> Tuple[str, str, Dict[str, Any]]:
        """
        Parse raw model output into structured tool call.
        
        Args:
            raw_output: Raw string output from the model
            
        Returns:
            Tuple of (tool_name, tool_call_string, parameters_dict)
        """
        import re
        
        # Default parsing logic - can be overridden by subclasses
        # Try to extract function call pattern: tool_name(param1=val1, param2=val2)
        pattern = r'(\w+)\s*\((.*?)\)'
        match = re.search(pattern, raw_output)
        
        if match:
            tool_name = match.group(1)
            params_str = match.group(2)
            tool_call = f"{tool_name}({params_str})"
            
            # Parse parameters
            params = {}
            if params_str.strip():
                # Handle both key=value and key='value' formats
                param_pattern = r"(\w+)\s*=\s*['\"]?([^,'\"]+)['\"]?"
                for param_match in re.finditer(param_pattern, params_str):
                    key = param_match.group(1)
                    value = param_match.group(2).strip()
                    params[key] = value
                    
            return tool_name, tool_call, params
        
        # Fallback: try to extract just tool name
        words = raw_output.split()
        for word in words:
            if word.endswith('()'):
                tool_name = word[:-2]
                return tool_name, word, {}
        
        return "", raw_output, {}
    
    def unload_model(self) -> None:
        """Unload model from memory to free resources."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self._is_loaded = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device={self.device}, loaded={self._is_loaded})"

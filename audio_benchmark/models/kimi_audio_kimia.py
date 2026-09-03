"""
Kimi Audio Model Adapter using Official kimia_infer API

This adapter uses the official MoonshotAI kimia_infer library for inference,
which is more reliable than the HuggingFace transformers approach since
Kimi-Audio has a custom architecture not fully supported by transformers.

Installation:
    pip install git+https://github.com/MoonshotAI/Kimi-Audio.git

HuggingFace: moonshotai/Kimi-Audio-7B-Instruct
GitHub: https://github.com/MoonshotAI/Kimi-Audio
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseAudioModel, ModelOutput, register_model

logger = logging.getLogger(__name__)


@register_model("kimi-audio-kimia")
class KimiAudioKimiaModel(BaseAudioModel):
    """
    Kimi Audio model adapter using the official kimia_infer API.
    
    This adapter wraps the Kimi-Audio-7B-Instruct model from Moonshot AI
    using their official inference library for best compatibility.
    
    Default model: moonshotai/Kimi-Audio-7B-Instruct
    """
    
    DEFAULT_MODEL_PATH = "moonshotai/Kimi-Audio-7B-Instruct"
    
    # Default sampling parameters for Kimi-Audio
    DEFAULT_SAMPLING_PARAMS = {
        "audio_temperature": 0.8,
        "audio_top_k": 10,
        "text_temperature": 0.0,
        "text_top_k": 5,
        "audio_repetition_penalty": 1.0,
        "audio_repetition_window_size": 64,
        "text_repetition_penalty": 1.0,
        "text_repetition_window_size": 16,
    }
    
    def __init__(
        self,
        device: str = "cuda",
        model_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        torch_dtype: str = "auto",
        load_detokenizer: bool = False,
        **kwargs
    ):
        """
        Args:
            device: Device to run on (cuda or cpu)
            model_path: HuggingFace model path or local path
            config: Optional configuration dict
            torch_dtype: Data type (auto, float16, bfloat16)
            load_detokenizer: Whether to load the audio detokenizer (for audio output)
        """
        super().__init__(device, model_path, config, **kwargs)
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.torch_dtype = torch_dtype
        self.load_detokenizer = load_detokenizer
        self._model = None
    
    def load_model(self) -> None:
        """Load Kimi Audio model using kimia_infer."""
        if self._is_loaded:
            logger.info("Model already loaded")
            return
        
        logger.info(f"Loading Kimi Audio from {self.model_path} using kimia_infer...")
        
        try:
            from kimia_infer.api.kimia import KimiAudio
        except ImportError as e:
            raise ImportError(
                "kimia_infer not installed. Install with:\n"
                "  pip install git+https://github.com/MoonshotAI/Kimi-Audio.git"
            ) from e
        
        self._model = KimiAudio(
            model_path=self.model_path,
            load_detokenizer=self.load_detokenizer
        )
        
        self._is_loaded = True
        logger.info(f"Kimi Audio loaded successfully")
    
    def process_audio(self, audio_path: str) -> str:
        """Return the audio path as-is (kimia_infer handles loading)."""
        return audio_path
    
    def _build_tools_in_prompt(self, tools: List[Dict[str, Any]]) -> str:
        """Build a text representation of tools to embed in the system prompt."""
        lines = []
        for tool in tools:
            sig = tool.get("signature", "") or tool.get("name", "")
            desc = tool.get("description", "")
            lines.append(f"- {sig}: {desc}")
        return "\n".join(lines)
    
    def generate(
        self,
        audio_path: str,
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 256,
        **kwargs
    ) -> ModelOutput:
        """
        Generate tool call from audio input using kimia_infer.
        
        Args:
            audio_path: Path to audio file
            tools: List of available tools
            system_prompt: Optional system prompt
            max_new_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            ModelOutput with predicted tool call
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        # Build tools prompt
        tools_text = self._build_tools_in_prompt(tools)
        
        # System prompt for tool calling
        if system_prompt is None:
            system_prompt = (
                "You are a voice command executor with tool calling capabilities.\n"
                "You MUST call a tool for every user request. Do NOT respond with text.\n"
                "Even if the command is vague or missing parameters, call the most appropriate tool.\n"
                "If parameters are unclear, use reasonable defaults or empty values.\n"
                "ALWAYS call a tool. NEVER ask for clarification."
            )
        
        full_system_prompt = (
            f"{system_prompt}\n\n"
            "Available tools:\n"
            f"{tools_text}\n\n"
            "Output ONLY the tool call in the format: tool_name(param1=value1, param2=value2).\n"
            "If no parameters are mentioned, output: tool_name()."
        )
        
        user_prompt = "Convert this voice command to a tool call."
        
        # Build messages in Kimi-Audio format
        messages = [
            {"role": "user", "message_type": "text", "content": full_system_prompt},
            {"role": "user", "message_type": "audio", "content": audio_path},
            {"role": "user", "message_type": "text", "content": user_prompt},
        ]
        
        # Prepare sampling params
        sampling_params = self.DEFAULT_SAMPLING_PARAMS.copy()
        sampling_params["text_temperature"] = kwargs.get("temperature", 0.0)
        
        try:
            # Generate text output only (no audio)
            _, text_output = self._model.generate(
                messages,
                **sampling_params,
                output_type="text"
            )
            raw_output = text_output or ""
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raw_output = f"ERROR: {e}"
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Parse tool call from output
        tool_name, tool_call_str, parameters = self.parse_tool_call(raw_output)
        
        all_tool_calls = []
        if tool_name:
            all_tool_calls.append({
                "name": tool_name,
                "parameters": parameters,
                "call_string": tool_call_str,
            })
        
        return ModelOutput(
            tool_name=tool_name,
            tool_call=tool_call_str,
            parameters=parameters,
            raw_output=raw_output,
            latency_ms=latency_ms,
            all_tool_calls=all_tool_calls,
        )
    
    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        
        # Clear CUDA cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        self._is_loaded = False
        logger.info("Kimi Audio model unloaded")

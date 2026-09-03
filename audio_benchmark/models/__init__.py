"""
Audio Model Adapters for Tool Calling Benchmark

This module provides a registry-based system for audio models.
To add a new model, create a class inheriting from BaseAudioModel
and register it using the @register_model decorator.
"""

from .base import BaseAudioModel, register_model, get_model, list_models, MODEL_REGISTRY
from .qwen2_audio import Qwen2AudioModel
from .qwen3_omni import Qwen3OmniModel
from .kimi_audio import KimiAudioModel
from .kimi_audio_kimia import KimiAudioKimiaModel
from .kimi_audio_vllm import KimiAudioVllmModel
from .step_audio import StepAudio2Model
from .step_audio_vllm import StepAudio2VllmModel
from .qwen3_omni_vllm import Qwen3OmniVllmModel
from .audio_flamingo3_vllm import AudioFlamingo3VllmModel
from .whisper_gemma3_vllm import WhisperGemma3VllmModel, WhisperGemma3VllmGTModel
from .whisper_qwen3_vllm import WhisperQwen3VllmModel, WhisperQwen3VllmGTModel

__all__ = [
    'BaseAudioModel',
    'register_model', 
    'get_model',
    'list_models',
    'MODEL_REGISTRY',
    'Qwen2AudioModel',
    'Qwen3OmniModel',
    'Qwen3OmniVllmModel',
    'KimiAudioModel',
    'KimiAudioKimiaModel',
    'KimiAudioVllmModel',
    'StepAudio2Model',
    'StepAudio2VllmModel',
    'AudioFlamingo3VllmModel',
    'WhisperGemma3VllmModel',
    'WhisperGemma3VllmGTModel',
    'WhisperQwen3VllmModel',
    'WhisperQwen3VllmGTModel',
]

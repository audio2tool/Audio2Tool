"""
Qwen2-Audio Model Adapter

Qwen2-Audio is a large audio-language model from Alibaba that supports
audio understanding and can be used for tool calling tasks.

HuggingFace: Qwen/Qwen2-Audio-7B-Instruct
"""

import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
import librosa

from .base import BaseAudioModel, ModelOutput, register_model
from ..utils.tools_schema import load_tools_from_csv, format_tools_compact

logger = logging.getLogger(__name__)


@register_model("qwen2-audio")
class Qwen2AudioModel(BaseAudioModel):
    """
    Qwen2-Audio model adapter for tool calling benchmark.
    
    This adapter wraps the Qwen2-Audio-7B-Instruct model from Alibaba/Qwen.
    The model supports audio understanding and can generate structured
    tool calls from audio input.
    
    Default model: Qwen/Qwen2-Audio-7B-Instruct
    """
    
    DEFAULT_MODEL_PATH = "Qwen/Qwen2-Audio-7B-Instruct"
    
    # Path to tools taxonomy CSV
    DEFAULT_TOOLS_CSV = Path(__file__).parent.parent / "taxonomy_tools_5.backtracked.csv"
    
    def __init__(
        self,
        device: str = "cuda",
        model_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        torch_dtype: str = "auto",
        tools_csv_path: Optional[str] = None,
        num_gpus: int = 1,
        gpu_ids: Optional[List[int]] = None,
        **kwargs
    ):
        super().__init__(device, model_path, config, **kwargs)
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.torch_dtype = torch_dtype
        self.num_gpus = num_gpus
        self.gpu_ids = gpu_ids  # Specific GPU IDs to use for model parallelism
        
        # Load tools schema
        csv_path = tools_csv_path or self.DEFAULT_TOOLS_CSV
        self.tools_schema = load_tools_from_csv(csv_path)
        self.tools_prompt = format_tools_compact(self.tools_schema)
        logger.info(f"Loaded {len(self.tools_schema)} tools from taxonomy")
        
    def _get_device_map(self):
        """
        Build the device_map for model loading.
        
        Returns:
            device_map: Either a specific device string or "auto" for multi-GPU.
            max_memory: Optional dict constraining per-GPU memory (for multi-GPU).
        """
        if self.num_gpus > 1:
            # Model parallelism: spread model across multiple GPUs
            # When CUDA_VISIBLE_DEVICES is set, GPUs are re-indexed from 0
            max_memory = {}
            for i in range(self.num_gpus):
                max_memory[i] = "38GiB"  # Leave ~1.5GB headroom per A100-40GB
            logger.info(f"Model parallelism: spreading across {self.num_gpus} GPUs, max_memory={max_memory}")
            return "auto", max_memory
        else:
            # Single GPU
            device_map = self.device if self.device != "cpu" else None
            return device_map, None
    
    def load_model(self) -> None:
        """Load Qwen2-Audio model and processor."""
        if self._is_loaded:
            logger.info("Model already loaded")
            return
            
        logger.info(f"Loading Qwen2-Audio from {self.model_path}")
        if self.num_gpus > 1:
            logger.info(f"Using model parallelism across {self.num_gpus} GPUs")
        
        from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
        
        # Determine torch dtype
        if self.torch_dtype == "auto":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif self.torch_dtype == "float16":
            dtype = torch.float16
        elif self.torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
            
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        
        # Determine device placement
        device_map, max_memory = self._get_device_map()
        
        load_kwargs = {
            "torch_dtype": dtype,
            "device_map": device_map,
        }
        if max_memory is not None:
            load_kwargs["max_memory"] = max_memory
        
        # Use Flash Attention 2 for memory efficiency and speed
        try:
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                self.model_path,
                attn_implementation="flash_attention_2",
                **load_kwargs,
            )
            logger.info("Loaded model with Flash Attention 2")
        except Exception as e:
            logger.warning(f"Flash Attention 2 not available: {e}. Falling back to default attention.")
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                self.model_path,
                **load_kwargs,
            )
        
        if self.device == "cpu" and self.num_gpus <= 1:
            self.model = self.model.to("cpu")
        
        self._is_loaded = True
        
        # Log device placement
        if self.num_gpus > 1:
            device_info = {name: str(param.device) for name, param in list(self.model.named_parameters())[:5]}
            logger.info(f"Qwen2-Audio loaded with model parallelism. Sample device placement: {device_info}")
        else:
            logger.info(f"Qwen2-Audio loaded successfully on {self.device}")
        
    def process_audio(self, audio_path: str) -> Any:
        """
        Load and process audio file for Qwen2-Audio.
        
        Args:
            audio_path: Path to WAV audio file
            
        Returns:
            Audio array at 16kHz sample rate
        """
        # Qwen2-Audio expects 16kHz audio
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        return audio
    
    def _build_tool_prompt(self, tools: List[Dict[str, Any]]) -> str:
        """Build tool description prompt."""
        tool_descriptions = []
        for tool in tools:
            name = tool.get("name", "")
            description = tool.get("description", "")
            params = tool.get("parameters", {})
            
            param_str = ""
            if params:
                param_list = []
                for p_name, p_info in params.items():
                    p_type = p_info.get("type", "string")
                    p_desc = p_info.get("description", "")
                    param_list.append(f"  - {p_name} ({p_type}): {p_desc}")
                param_str = "\n".join(param_list)
            
            tool_desc = f"- {name}: {description}"
            if param_str:
                tool_desc += f"\n  Parameters:\n{param_str}"
            tool_descriptions.append(tool_desc)
            
        return "\n".join(tool_descriptions)
    
    def generate(
        self,
        audio_path: str,
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 256,
        **kwargs
    ) -> ModelOutput:
        """
        Generate tool call from audio input.
        
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
        
        # Load audio
        audio = self.process_audio(audio_path)
        
        # Use pre-loaded tools from taxonomy CSV
        tool_prompt = self.tools_prompt
        
        if system_prompt is None:
            system_prompt = (
                "You are an AI assistant that converts voice commands into tool calls. "
                "Listen to the audio and determine which tool should be called. "
                "Output ONLY the tool call in the format: tool_name(param1=value1, param2=value2). "
                "If no parameters are mentioned, output: tool_name()."
            )
        
        # Build conversation with full tool taxonomy
        conversation = [
            {
                "role": "system",
                "content": f"{system_prompt}\n\n## Available Tools:\n{tool_prompt}"
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": audio_path},
                    {"type": "text", "text": "Convert this voice command to a tool call."}
                ]
            }
        ]
        
        # Process inputs
        text = self.processor.apply_chat_template(
            conversation, 
            add_generation_prompt=True, 
            tokenize=False
        )
        
        audios = [audio]
        inputs = self.processor(
            text=text, 
            audios=audios, 
            return_tensors="pt",
            padding=True
        )
        # For model parallelism, inputs go to the first device in the pipeline
        if self.num_gpus > 1:
            first_device = next(self.model.parameters()).device
            inputs = inputs.to(first_device)
        else:
            inputs = inputs.to(self.model.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
        
        # Decode only new tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        raw_output = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Parse output
        tool_name, tool_call, params = self.parse_tool_call(raw_output)
        
        return ModelOutput(
            tool_name=tool_name,
            tool_call=tool_call,
            parameters=params,
            raw_output=raw_output,
            latency_ms=latency_ms
        )

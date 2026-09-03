"""
Kimi Audio Model Adapter

Kimi Audio is a large audio-language model from Moonshot AI that supports
audio understanding, speech-to-text, and conversational tasks.

For our benchmark (audio -> tool call text), we:
- Use Flash Attention 2 for memory efficiency
- Support model parallelism across multiple GPUs
- Pre-load the full tools taxonomy for consistent prompting

HuggingFace: moonshotai/Kimi-Audio-7B-Instruct
"""

import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
import librosa
import soundfile as sf

from .base import BaseAudioModel, ModelOutput, register_model
from ..utils.tools_schema import load_tools_from_csv, format_tools_compact

logger = logging.getLogger(__name__)


@register_model("kimi-audio")
class KimiAudioModel(BaseAudioModel):
    """
    Kimi Audio model adapter for tool calling benchmark.
    
    This adapter wraps the Kimi-Audio-7B-Instruct model from Moonshot AI.
    The model supports audio understanding and can generate structured
    tool calls from audio input.
    
    Default model: moonshotai/Kimi-Audio-7B-Instruct
    """
    
    DEFAULT_MODEL_PATH = "moonshotai/Kimi-Audio-7B-Instruct"
    
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
        self.gpu_ids = gpu_ids
        self.tokenizer = None
        self.feature_extractor = None
        
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
            max_memory = {}
            for i in range(self.num_gpus):
                max_memory[i] = "38GiB"
            logger.info(
                f"Model parallelism: spreading across {self.num_gpus} GPUs, "
                f"max_memory={max_memory}"
            )
            return "auto", max_memory
        else:
            device_map = self.device if self.device != "cpu" else None
            return device_map, None
        
    def load_model(self) -> None:
        """Load Kimi Audio model and processor."""
        if self._is_loaded:
            logger.info("Model already loaded")
            return
            
        logger.info(f"Loading Kimi Audio from {self.model_path}")
        if self.num_gpus > 1:
            logger.info(f"Using model parallelism across {self.num_gpus} GPUs")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
            
            # Determine torch dtype
            if self.torch_dtype == "auto":
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            elif self.torch_dtype == "float16":
                dtype = torch.float16
            elif self.torch_dtype == "bfloat16":
                dtype = torch.bfloat16
            else:
                dtype = torch.float32
            
            # Load tokenizer and processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Determine device placement
            device_map, max_memory = self._get_device_map()
            
            load_kwargs = {
                "torch_dtype": dtype,
                "device_map": device_map,
                "trust_remote_code": True,
            }
            if max_memory is not None:
                load_kwargs["max_memory"] = max_memory
            
            # Use Flash Attention 2 for memory efficiency and speed
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    attn_implementation="flash_attention_2",
                    **load_kwargs,
                )
                logger.info("Loaded model with Flash Attention 2")
            except Exception as e:
                logger.warning(
                    f"Flash Attention 2 not available: {e}. "
                    "Falling back to default attention."
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    **load_kwargs,
                )
            
            if self.device == "cpu" and self.num_gpus <= 1:
                self.model = self.model.to("cpu")
            
            self._is_loaded = True
            
            # Log device placement
            if self.num_gpus > 1:
                device_info = {
                    name: str(param.device)
                    for name, param in list(self.model.named_parameters())[:5]
                }
                logger.info(
                    f"Kimi Audio loaded with model parallelism. "
                    f"Sample device placement: {device_info}"
                )
            else:
                logger.info(f"Kimi Audio loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load Kimi Audio model: {e}")
            logger.info("Attempting alternative loading method...")
            self._load_alternative()
    
    def _load_alternative(self) -> None:
        """Alternative loading method if standard loading fails."""
        from transformers import AutoModel, AutoTokenizer
        
        if self.torch_dtype == "auto":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        # Determine device placement
        device_map, max_memory = self._get_device_map()
        
        load_kwargs = {
            "torch_dtype": dtype,
            "device_map": device_map,
            "trust_remote_code": True,
        }
        if max_memory is not None:
            load_kwargs["max_memory"] = max_memory
        
        # Try Flash Attention 2
        try:
            self.model = AutoModel.from_pretrained(
                self.model_path,
                attn_implementation="flash_attention_2",
                **load_kwargs,
            )
            logger.info("Loaded model (alternative) with Flash Attention 2")
        except Exception:
            self.model = AutoModel.from_pretrained(
                self.model_path,
                **load_kwargs,
            )
        
        self._is_loaded = True
        logger.info(f"Kimi Audio loaded (alternative) on {self.device}")
        
    def process_audio(self, audio_path: str) -> Any:
        """
        Load and process audio file for Kimi Audio.
        
        Args:
            audio_path: Path to WAV audio file
            
        Returns:
            Audio array at appropriate sample rate
        """
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
        
        user_prompt = (
            f"## Available Tools:\n{tool_prompt}\n\n"
            "Convert this voice command to a tool call."
        )
        
        # Build conversation based on processor type
        try:
            if hasattr(self.processor, 'apply_chat_template'):
                conversation = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio": audio},
                            {"type": "text", "text": user_prompt}
                        ]
                    }
                ]
                
                text = self.processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    tokenize=False
                )
                
                inputs = self.processor(
                    text=text,
                    audios=[audio],
                    return_tensors="pt",
                    padding=True
                )
            else:
                # Fallback for tokenizer-based models
                full_prompt = f"{system_prompt}\n\n{user_prompt}\n\nResponse:"
                inputs = self.tokenizer(
                    full_prompt,
                    return_tensors="pt",
                    padding=True
                )
                if hasattr(self.model, 'encode_audio'):
                    audio_features = self.model.encode_audio(
                        torch.tensor(audio).unsqueeze(0).to(self.model.device)
                    )
                    inputs['audio_features'] = audio_features
                    
        except Exception as e:
            logger.warning(f"Error processing with standard method: {e}")
            inputs = self._prepare_inputs_fallback(audio, system_prompt, user_prompt)
        
        # Move inputs to the correct device
        if self.num_gpus > 1:
            first_device = next(self.model.parameters()).device
            inputs = {k: v.to(first_device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        else:
            inputs = {k: v.to(self.model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=kwargs.get('do_sample', False),
                temperature=kwargs.get('temperature', 1.0),
                **{k: v for k, v in kwargs.items() if k not in ['do_sample', 'temperature']}
            )
        
        # Decode
        if hasattr(self.processor, 'batch_decode'):
            raw_output = self.processor.batch_decode(
                generated_ids[:, inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )[0]
        elif self.tokenizer is not None:
            raw_output = self.tokenizer.batch_decode(
                generated_ids[:, inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )[0]
        else:
            raw_output = str(generated_ids)
        
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
    
    def _prepare_inputs_fallback(self, audio, system_prompt, user_prompt):
        """Fallback input preparation method."""
        full_prompt = f"{system_prompt}\n\n{user_prompt}\n\nResponse:"
        
        if self.tokenizer:
            return self.tokenizer(full_prompt, return_tensors="pt", padding=True)
        elif self.processor:
            return self.processor(text=full_prompt, return_tensors="pt", padding=True)
        else:
            raise RuntimeError("No tokenizer or processor available")

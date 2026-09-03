"""
Qwen3-Omni Model Adapter

Qwen3-Omni is a natively end-to-end multilingual omni-modal model from Alibaba
that processes text, images, audio, and video. It uses a MoE-based Thinker-Talker
architecture (30B total / 3B active parameters).

For our benchmark (audio -> tool call text), we:
- Disable the talker component to save ~10GB GPU memory
- Disable thinking/CoT for clean tool call output
- Use Flash Attention 2 for memory efficiency
- Support model parallelism across multiple GPUs

HuggingFace: Qwen/Qwen3-Omni-30B-A3B-Instruct

Requirements:
    pip install git+https://github.com/huggingface/transformers
    pip install accelerate qwen-omni-utils flash-attn --no-build-isolation
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


@register_model("qwen3-omni")
class Qwen3OmniModel(BaseAudioModel):
    """
    Qwen3-Omni model adapter for tool calling benchmark.

    This adapter wraps the Qwen3-Omni-30B-A3B-Instruct model from Alibaba/Qwen.
    The model is a Mixture-of-Experts (MoE) architecture with 30B total params
    and 3B active, supporting audio, image, video, and text understanding.

    For the benchmark we only need text output (tool calls), so the talker
    component is disabled to save ~10GB of GPU memory.

    Default model: Qwen/Qwen3-Omni-30B-A3B-Instruct
    """

    DEFAULT_MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

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
        enable_thinking: bool = False,
        disable_talker: bool = True,
        **kwargs
    ):
        """
        Initialize Qwen3-Omni adapter.

        Args:
            device: Device string (e.g. "cuda", "cuda:0", "cpu").
            model_path: HuggingFace model ID or local path.
            config: Additional model configuration.
            torch_dtype: Data type ("auto", "bfloat16", "float16", "float32").
            tools_csv_path: Path to tools taxonomy CSV.
            num_gpus: Number of GPUs for model parallelism (1 = single GPU).
            gpu_ids: Specific GPU IDs for model parallelism.
            enable_thinking: Whether to enable chain-of-thought reasoning.
                             False (default) gives cleaner tool call output.
            disable_talker: Whether to disable the talker component.
                            True (default) saves ~10GB GPU memory since we
                            only need text output for tool calling.
            **kwargs: Additional arguments passed to base class.
        """
        super().__init__(device, model_path, config, **kwargs)
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.torch_dtype = torch_dtype
        self.num_gpus = num_gpus
        self.gpu_ids = gpu_ids
        self.enable_thinking = enable_thinking
        self.disable_talker = disable_talker

        # Load tools schema
        csv_path = tools_csv_path or self.DEFAULT_TOOLS_CSV
        self.tools_schema = load_tools_from_csv(csv_path)
        self.tools_prompt = format_tools_compact(self.tools_schema)
        logger.info(f"Loaded {len(self.tools_schema)} tools from taxonomy")

        # Check for qwen_omni_utils availability
        self._has_omni_utils = False
        try:
            from qwen_omni_utils import process_mm_info
            self._has_omni_utils = True
            logger.info("qwen_omni_utils available — using official audio processing")
        except ImportError:
            logger.info(
                "qwen_omni_utils not installed — falling back to librosa. "
                "Install with: pip install qwen-omni-utils"
            )

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
            logger.info(
                f"Model parallelism: spreading across {self.num_gpus} GPUs, "
                f"max_memory={max_memory}"
            )
            return "auto", max_memory
        else:
            # Single GPU
            device_map = self.device if self.device != "cpu" else None
            return device_map, None

    def load_model(self) -> None:
        """Load Qwen3-Omni model and processor."""
        if self._is_loaded:
            logger.info("Model already loaded")
            return

        logger.info(f"Loading Qwen3-Omni from {self.model_path}")
        if self.num_gpus > 1:
            logger.info(f"Using model parallelism across {self.num_gpus} GPUs")

        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

        # Determine torch dtype
        if self.torch_dtype == "auto":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif self.torch_dtype == "float16":
            dtype = torch.float16
        elif self.torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        self.processor = Qwen3OmniMoeProcessor.from_pretrained(self.model_path)

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
            self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
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
            self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                self.model_path,
                **load_kwargs,
            )

        # Disable talker component to save ~10GB GPU memory (text-only output)
        if self.disable_talker:
            self.model.disable_talker()
            logger.info("Talker disabled — text-only mode (saves ~10GB GPU memory)")

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
                f"Qwen3-Omni loaded with model parallelism. "
                f"Sample device placement: {device_info}"
            )
        else:
            logger.info(f"Qwen3-Omni loaded successfully on {self.device}")

    def process_audio(self, audio_path: str) -> Any:
        """
        Load and process audio file for Qwen3-Omni.

        Uses qwen_omni_utils if available, otherwise falls back to librosa.

        Args:
            audio_path: Path to WAV audio file

        Returns:
            Audio array at 16kHz sample rate
        """
        # Load with librosa at 16kHz (Whisper-based encoder standard)
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

        # Use pre-loaded tools from taxonomy CSV
        tool_prompt = self.tools_prompt

        if system_prompt is None:
            system_prompt = (
                "You are an AI assistant that converts voice commands into tool calls. "
                "Listen to the audio and determine which tool should be called. "
                "Output ONLY the tool call in the format: tool_name(param1=value1, param2=value2). "
                "If no parameters are mentioned, output: tool_name()."
            )

        # Build conversation — Qwen3-Omni uses "audio" key (not "audio_url")
        conversation = [
            {
                "role": "system",
                "content": f"{system_prompt}\n\n## Available Tools:\n{tool_prompt}"
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": "Convert this voice command to a tool call."}
                ]
            }
        ]

        # Process conversation into model inputs
        text = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=self.enable_thinking,
        )

        # Extract audio data from conversation
        if self._has_omni_utils:
            from qwen_omni_utils import process_mm_info
            audios, images, videos = process_mm_info(
                conversation, use_audio_in_video=False
            )
        else:
            # Fallback: load audio manually with librosa
            audios = [self.process_audio(audio_path)]
            images = None
            videos = None

        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )

        # Move inputs to the correct device
        if self.num_gpus > 1:
            first_device = next(self.model.parameters()).device
            inputs = inputs.to(first_device)
        else:
            inputs = inputs.to(self.model.device)

        # Cast to model dtype for efficiency
        inputs = inputs.to(self.model.dtype)

        # Filter out generation kwargs that conflict with Qwen3-Omni
        gen_kwargs = {k: v for k, v in kwargs.items() if k not in (
            'return_audio', 'enable_thinking', 'thinker_return_dict_in_generate',
            'use_audio_in_video',
        )}

        # Generate text-only output
        # Note: enable_thinking is only for apply_chat_template, not generate()
        with torch.no_grad():
            text_ids, _audio = self.model.generate(
                **inputs,
                return_audio=False,
                thinker_return_dict_in_generate=True,
                use_audio_in_video=False,
                max_new_tokens=max_new_tokens,
                **gen_kwargs,
            )

        # Decode only new tokens (trim input prefix)
        input_len = inputs["input_ids"].shape[1]
        generated_sequences = text_ids.sequences[:, input_len:]

        raw_output = self.processor.batch_decode(
            generated_sequences,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # If thinking was enabled, strip the thinking tags to get clean output
        if self.enable_thinking and "<think>" in raw_output:
            # Extract content after </think> tag
            if "</think>" in raw_output:
                raw_output = raw_output.split("</think>")[-1].strip()

        latency_ms = (time.time() - start_time) * 1000

        # Parse output
        tool_name, tool_call, params = self.parse_tool_call(raw_output)

        return ModelOutput(
            tool_name=tool_name,
            tool_call=tool_call,
            parameters=params,
            raw_output=raw_output,
            latency_ms=latency_ms,
        )

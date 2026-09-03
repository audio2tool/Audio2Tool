"""
Whisper 3 + Qwen3 Cascaded Model Adapter

This adapter implements a two-stage pipeline:
1. Whisper 3 (ASR): Audio → Text transcription
2. Qwen3 (LLM): Text → Tool call

Supports multiple Qwen3 model sizes: 1.7B, 4B, 8B

Single server usage:
    # Start Qwen3 server:
    vllm serve Qwen/Qwen3-8B --port 8801 --dtype bfloat16 --trust-remote-code

    # Run benchmark with Whisper transcription:
    python run_benchmark.py --model whisper-qwen3-vllm ...

    # Run benchmark with ground truth text (no Whisper):
    python run_benchmark.py --model whisper-qwen3-vllm-gt ...

Multi-server usage (8 GPUs: 8 servers x 1 GPU each):
    # Server 1 on GPU 0:
    CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-8B --port 8801
    # ... up to port 8808

    The adapter auto-discovers servers on ports 8801-8808 by default.
"""

import json
import logging
import time
import threading
from typing import Any, Dict, List, Optional

import numpy as np
import requests
import torch

from .base import BaseAudioModel, ModelOutput, register_model

logger = logging.getLogger(__name__)

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False

try:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


def _load_audio(file_path: str, target_rate: int = 16000) -> np.ndarray:
    """Load audio file and resample to target rate."""
    if _HAS_LIBROSA:
        audio, _ = librosa.load(file_path, sr=target_rate, mono=True)
        return audio.astype(np.float32)
    else:
        import soundfile as sf
        audio, sr = sf.read(file_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != target_rate:
            ratio = target_rate / sr
            n_out = int(len(audio) * ratio)
            indices = np.arange(n_out) / ratio
            indices = np.clip(indices.astype(int), 0, len(audio) - 1)
            audio = audio[indices]
        return audio


class WhisperTranscriber:
    """Whisper 3 transcription component (shared with Gemma adapter)."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __init__(
        self,
        model_id: str = "openai/whisper-large-v3",
        device: str = "cuda",
        torch_dtype: str = "float16",
    ):
        self.model_id = model_id
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype, torch.float16)
        self.pipe = None
        self._is_loaded = False
    
    def load(self) -> None:
        """Load Whisper model."""
        if self._is_loaded:
            return
        
        if not _HAS_TRANSFORMERS:
            raise ImportError("transformers required for Whisper. pip install transformers")
        
        logger.info(f"Loading Whisper model: {self.model_id}")
        
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(self.device)
        
        processor = AutoProcessor.from_pretrained(self.model_id)
        
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
        
        self._is_loaded = True
        logger.info("Whisper model loaded successfully")
    
    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file to text."""
        if not self._is_loaded:
            raise RuntimeError("Whisper not loaded. Call load() first.")
        
        result = self.pipe(
            audio_path,
            generate_kwargs={"language": "english", "task": "transcribe"},
            return_timestamps=False,
        )
        return result["text"].strip()
    
    def unload(self) -> None:
        """Unload Whisper model."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        self._is_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class Qwen3VllmClient:
    """Qwen3 vLLM client component."""
    
    DEFAULT_BASE_PORT = 8801
    DEFAULT_NUM_PORTS = 8
    DEFAULT_MODEL_NAME = "Qwen/Qwen3-8B"
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        base_port: int = DEFAULT_BASE_PORT,
        num_ports: int = DEFAULT_NUM_PORTS,
        api_urls: Optional[List[str]] = None,
    ):
        self.served_model_name = model_name or self.DEFAULT_MODEL_NAME
        self.base_port = base_port
        self.num_ports = num_ports
        
        if api_urls:
            self._candidate_urls = api_urls
        else:
            self._candidate_urls = [
                f"http://localhost:{base_port + i}/v1/chat/completions"
                for i in range(num_ports)
            ]
        
        self._live_urls: List[str] = []
        self._rr_counter = 0
        self._rr_lock = threading.Lock()
        self._is_loaded = False
    
    def load(self) -> None:
        """Probe candidate URLs and keep the ones that respond."""
        if self._is_loaded:
            return
        
        self._live_urls = []
        for url in self._candidate_urls:
            health_url = url.replace("/v1/chat/completions", "/health")
            try:
                resp = requests.get(health_url, timeout=5)
                if resp.status_code == 200:
                    self._live_urls.append(url)
            except requests.RequestException:
                pass
        
        if not self._live_urls:
            raise RuntimeError(
                f"No Qwen3 vLLM servers reachable. Tried: {self._candidate_urls}"
            )
        
        self._is_loaded = True
        logger.info(
            f"Connected to {len(self._live_urls)} Qwen3 vLLM server(s): "
            + ", ".join(self._live_urls)
        )
    
    def _next_url(self) -> str:
        """Round-robin URL selection (thread-safe)."""
        with self._rr_lock:
            url = self._live_urls[self._rr_counter % len(self._live_urls)]
            self._rr_counter += 1
        return url
    
    def generate(
        self,
        text: str,
        system_prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.1,
        **kwargs,
    ) -> Optional[dict]:
        """Send text to Qwen3 and get response."""
        payload = {
            "model": self.served_model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": kwargs.get("top_p", 0.95),
            "stream": False,
        }
        
        max_retries = 2
        for attempt in range(max_retries + 1):
            api_url = self._next_url()
            try:
                resp = requests.post(
                    api_url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=60,
                )
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                if attempt < max_retries:
                    logger.warning(f"vLLM API error (attempt {attempt+1}), retrying: {e}")
                else:
                    logger.error(f"vLLM API error ({api_url}): {e}")
        return None
    
    def unload(self) -> None:
        """Disconnect from servers."""
        self._is_loaded = False
        self._live_urls = []


@register_model("whisper-qwen3-vllm")
class WhisperQwen3VllmModel(BaseAudioModel):
    """
    Cascaded Whisper 3 + Qwen3 model for audio tool calling.
    
    Pipeline: Audio → Whisper 3 (ASR) → Qwen3 (Tool Call)
    
    Supports Qwen3 models: 1.7B, 4B, 8B
    """
    
    def __init__(
        self,
        device: str = "cuda",
        model_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        whisper_model: str = "openai/whisper-large-v3",
        whisper_device: Optional[str] = None,
        qwen3_model: str = "Qwen/Qwen3-8B",
        base_port: int = 8801,
        num_ports: int = 8,
        api_urls: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(device, model_path, config, **kwargs)
        
        self.whisper_model_id = whisper_model
        self.whisper_device = whisper_device or device
        self.qwen3_model_name = qwen3_model
        
        self.whisper = WhisperTranscriber(
            model_id=whisper_model,
            device=self.whisper_device,
        )
        
        self.qwen3 = Qwen3VllmClient(
            model_name=qwen3_model,
            base_port=base_port,
            num_ports=num_ports,
            api_urls=api_urls,
        )
    
    def load_model(self) -> None:
        """Load both Whisper and connect to Qwen3 servers."""
        if self._is_loaded:
            return
        
        self.whisper.load()
        self.qwen3.load()
        
        self._is_loaded = True
        logger.info(f"Whisper + Qwen3 ({self.qwen3_model_name}) pipeline loaded")
    
    def process_audio(self, audio_path: str) -> str:
        """Transcribe audio using Whisper 3."""
        return self.whisper.transcribe(audio_path)
    
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
        **kwargs,
    ) -> ModelOutput:
        """
        Process audio through the cascaded pipeline.
        
        1. Transcribe audio with Whisper 3
        2. Send transcription to Qwen3 for tool calling
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        # Stage 1: Whisper transcription
        transcription = self.process_audio(audio_path)
        whisper_time = time.time()
        
        # Stage 2: Qwen3 tool calling
        tools_text = self._build_tools_in_prompt(tools)
        
        if system_prompt is None:
            system_prompt = (
                "You are a voice command executor with tool calling capabilities.\n"
                "You MUST call a tool for every user request. Do NOT respond with text.\n"
                "Even if the command is vague or missing parameters, call the most appropriate tool.\n"
                "If parameters are unclear, use reasonable defaults or empty values.\n"
                "ALWAYS call a tool. NEVER ask for clarification."
            )
        
        full_prompt = (
            f"{system_prompt}\n\n"
            "Available tools:\n"
            f"{tools_text}\n\n"
            "Output ONLY the tool call in the format: tool_name(param1=value1, param2=value2).\n"
            "If no parameters are mentioned, output: tool_name()."
        )
        
        result = self.qwen3.generate(
            text=f"User command (transcribed from audio): {transcription}",
            system_prompt=full_prompt,
            max_tokens=max_new_tokens,
            temperature=kwargs.get("temperature", 0.1),
        )
        
        latency_ms = (time.time() - start_time) * 1000
        whisper_latency_ms = (whisper_time - start_time) * 1000
        
        if result is None:
            return ModelOutput(
                tool_name="",
                tool_call="",
                parameters={},
                raw_output=f"API_ERROR (transcription: {transcription})",
                latency_ms=latency_ms,
            )
        
        message = result["choices"][0]["message"]
        raw_output = message.get("content", "") or ""
        
        # Parse tool call from response
        tool_name, tool_call_str, parameters = self.parse_tool_call(raw_output)
        
        all_tool_calls = []
        if tool_name:
            all_tool_calls.append({
                "name": tool_name,
                "parameters": parameters,
                "call_string": tool_call_str,
            })
        
        # Include transcription in raw output for debugging
        full_raw = f"[Whisper: {transcription}] [Qwen3: {raw_output}]"
        
        return ModelOutput(
            tool_name=tool_name,
            tool_call=tool_call_str,
            parameters=parameters,
            raw_output=full_raw,
            latency_ms=latency_ms,
            all_tool_calls=all_tool_calls,
        )
    
    def unload_model(self) -> None:
        """Unload Whisper and disconnect from Qwen3 servers."""
        self.whisper.unload()
        self.qwen3.unload()
        self._is_loaded = False
        logger.info("Whisper + Qwen3 pipeline unloaded")


@register_model("whisper-qwen3-vllm-gt")
class WhisperQwen3VllmGTModel(BaseAudioModel):
    """
    Qwen3 model with Ground Truth text (no Whisper).
    
    This variant skips ASR and uses the ground truth query text directly.
    Useful for isolating LLM tool-calling performance from ASR errors.
    
    Pipeline: GT Text → Qwen3 (Tool Call)
    
    Supports Qwen3 models: 1.7B, 4B, 8B
    """
    
    def __init__(
        self,
        device: str = "cuda",
        model_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        qwen3_model: str = "Qwen/Qwen3-8B",
        base_port: int = 8801,
        num_ports: int = 8,
        api_urls: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(device, model_path, config, **kwargs)
        
        self.qwen3_model_name = qwen3_model
        
        self.qwen3 = Qwen3VllmClient(
            model_name=qwen3_model,
            base_port=base_port,
            num_ports=num_ports,
            api_urls=api_urls,
        )
    
    def load_model(self) -> None:
        """Connect to Qwen3 servers."""
        if self._is_loaded:
            return
        
        self.qwen3.load()
        self._is_loaded = True
        logger.info(f"Qwen3 ({self.qwen3_model_name}) GT pipeline loaded (no Whisper)")
    
    def process_audio(self, audio_path: str) -> str:
        """No-op for GT mode - audio is not processed."""
        return ""
    
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
        gt_query: Optional[str] = None,
        **kwargs,
    ) -> ModelOutput:
        """
        Process ground truth text through Qwen3.
        
        Args:
            audio_path: Path to audio (used to load GT from metadata if gt_query not provided)
            tools: Available tools
            gt_query: Ground truth query text (if not provided, loads from metadata)
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        # Get ground truth text
        if gt_query is None:
            import os
            query_dir = os.path.dirname(audio_path)
            metadata_path = os.path.join(query_dir, "query_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path) as f:
                    metadata = json.load(f)
                gt_query = metadata.get("query", "")
            else:
                raise ValueError(f"No gt_query provided and no metadata found at {metadata_path}")
        
        # Build prompt
        tools_text = self._build_tools_in_prompt(tools)
        
        if system_prompt is None:
            system_prompt = (
                "You are a voice command executor with tool calling capabilities.\n"
                "You MUST call a tool for every user request. Do NOT respond with text.\n"
                "Even if the command is vague or missing parameters, call the most appropriate tool.\n"
                "If parameters are unclear, use reasonable defaults or empty values.\n"
                "ALWAYS call a tool. NEVER ask for clarification."
            )
        
        full_prompt = (
            f"{system_prompt}\n\n"
            "Available tools:\n"
            f"{tools_text}\n\n"
            "Output ONLY the tool call in the format: tool_name(param1=value1, param2=value2).\n"
            "If no parameters are mentioned, output: tool_name()."
        )
        
        result = self.qwen3.generate(
            text=f"User command: {gt_query}",
            system_prompt=full_prompt,
            max_tokens=max_new_tokens,
            temperature=kwargs.get("temperature", 0.1),
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        if result is None:
            return ModelOutput(
                tool_name="",
                tool_call="",
                parameters={},
                raw_output=f"API_ERROR (GT: {gt_query})",
                latency_ms=latency_ms,
            )
        
        message = result["choices"][0]["message"]
        raw_output = message.get("content", "") or ""
        
        # Parse tool call from response
        tool_name, tool_call_str, parameters = self.parse_tool_call(raw_output)
        
        all_tool_calls = []
        if tool_name:
            all_tool_calls.append({
                "name": tool_name,
                "parameters": parameters,
                "call_string": tool_call_str,
            })
        
        # Include GT query in raw output for debugging
        full_raw = f"[GT: {gt_query}] [Qwen3: {raw_output}]"
        
        return ModelOutput(
            tool_name=tool_name,
            tool_call=tool_call_str,
            parameters=parameters,
            raw_output=full_raw,
            latency_ms=latency_ms,
            all_tool_calls=all_tool_calls,
        )
    
    def unload_model(self) -> None:
        """Disconnect from Qwen3 servers."""
        self.qwen3.unload()
        self._is_loaded = False
        logger.info("Qwen3 GT pipeline unloaded")

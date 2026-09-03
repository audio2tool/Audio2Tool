"""
Kimi Audio vLLM Model Adapter

This adapter connects to one or more running Kimi-Audio-7B servers
and distributes requests across them using round-robin load balancing.

The Kimi-Audio server uses the official kimia_infer API wrapped in
an OpenAI-compatible FastAPI server.

Single server usage:
    # Start server on GPU 0:
    cd <kimi_server_dir> && bash launch_server.sh
    
    # Run benchmark:
    python run_benchmark.py --model kimi-audio-vllm ...

Multi-server usage (8 GPUs):
    # Start 8 servers on ports 8903-8910 (one per GPU):
    for i in {0..7}; do
        CUDA_DEVICE=$i PORT=$((8903+i)) bash launch_server.sh &
    done
    
    The adapter auto-discovers servers on ports 8903-8910 by default.
"""

import base64
import io
import json
import logging
import re
import time
import threading
import wave
from typing import Any, Dict, List, Optional

import numpy as np
import requests
import soundfile as sf

from .base import BaseAudioModel, ModelOutput, register_model

logger = logging.getLogger(__name__)

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False


def _load_audio(file_path: str, target_rate: int = 16000) -> np.ndarray:
    """Load audio file and resample to target rate. Returns float32 numpy array."""
    if _HAS_LIBROSA:
        audio, _ = librosa.load(file_path, sr=target_rate, mono=True)
        return audio.astype(np.float32)
    else:
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


def _audio_to_base64_wav(audio: np.ndarray, sample_rate: int = 16000) -> str:
    """Convert float32 numpy audio to base64-encoded WAV data URL."""
    chunk_int16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(chunk_int16.tobytes())
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:audio/wav;base64,{b64}"


@register_model("kimi-audio-vllm")
class KimiAudioVllmModel(BaseAudioModel):
    """
    Kimi Audio model adapter that connects to running Kimi-Audio server(s).

    Supports multiple servers for parallel inference across GPUs.
    Auto-discovers servers on ports 8903-8910 by default.
    """

    DEFAULT_BASE_PORT = 8903
    DEFAULT_NUM_PORTS = 8
    DEFAULT_MODEL_NAME = "moonshotai/Kimi-Audio-7B-Instruct"

    def __init__(
        self,
        device: str = "cuda",
        model_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        torch_dtype: str = "auto",
        api_url: Optional[str] = None,
        api_urls: Optional[List[str]] = None,
        model_name: Optional[str] = None,
        base_port: int = DEFAULT_BASE_PORT,
        num_ports: int = DEFAULT_NUM_PORTS,
        **kwargs,
    ):
        """
        Args:
            api_url: Single server URL (e.g. http://localhost:8903/v1/chat/completions)
            api_urls: Explicit list of server URLs
            base_port: Base port for auto-discovery (default: 8903)
            num_ports: Number of ports to probe starting from base_port (default: 8)
            model_name: Served model name (default: moonshotai/Kimi-Audio-7B-Instruct)
        """
        super().__init__(device, model_path, config, **kwargs)
        self.served_model_name = model_name or self.DEFAULT_MODEL_NAME
        self.base_port = base_port
        self.num_ports = num_ports

        if api_urls:
            self._candidate_urls = api_urls
        elif api_url:
            self._candidate_urls = [api_url]
        else:
            self._candidate_urls = [
                f"http://localhost:{base_port + i}/v1/chat/completions"
                for i in range(num_ports)
            ]

        self._live_urls: List[str] = []
        self._rr_counter = 0
        self._rr_lock = threading.Lock()

    def load_model(self) -> None:
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
                f"No Kimi-Audio servers reachable. Tried: {self._candidate_urls}"
            )

        self._is_loaded = True
        logger.info(
            f"Connected to {len(self._live_urls)} Kimi-Audio server(s): "
            + ", ".join(self._live_urls)
        )

    def _next_url(self) -> str:
        """Round-robin URL selection (thread-safe)."""
        with self._rr_lock:
            url = self._live_urls[self._rr_counter % len(self._live_urls)]
            self._rr_counter += 1
        return url

    def process_audio(self, audio_path: str) -> str:
        """Load audio and return base64-encoded WAV data URL for the API."""
        audio = _load_audio(audio_path, target_rate=16000)
        return _audio_to_base64_wav(audio, sample_rate=16000)

    def _build_openai_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert tools to OpenAI function calling format."""
        openai_tools = []
        for tool in tools:
            func_def = {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
                },
            }
            openai_tools.append(func_def)
        return openai_tools

    def generate(
        self,
        audio_path: str,
        system_prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        **kwargs,
    ) -> ModelOutput:
        """
        Send audio to a Kimi-Audio server (round-robin) and parse the tool-call response.
        """
        start_time = time.time()

        audio_b64_url = self.process_audio(audio_path)

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": audio_b64_url}},
                    {"type": "text", "text": "Listen to the audio and call the appropriate tool."},
                ],
            },
        ]

        openai_tools = self._build_openai_tools(tools) if tools else None

        payload = {
            "model": self.served_model_name,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }
        if openai_tools:
            payload["tools"] = openai_tools
            payload["tool_choice"] = "auto"

        url = self._next_url()
        max_retries = min(3, len(self._live_urls))
        last_error = None

        for attempt in range(max_retries):
            try:
                resp = requests.post(url, json=payload, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                break
            except requests.RequestException as e:
                last_error = e
                logger.warning(
                    f"Kimi-Audio API error (attempt {attempt+1}) for '{audio_path}', "
                    f"retrying on different server: {e}"
                )
                url = self._next_url()
        else:
            raise RuntimeError(
                f"All Kimi-Audio servers failed for '{audio_path}': {last_error}"
            )

        latency_ms = (time.time() - start_time) * 1000

        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        raw_output = message.get("content", "")
        tool_calls = message.get("tool_calls")

        parsed_tool = None
        parsed_params = {}

        if tool_calls and len(tool_calls) > 0:
            tc = tool_calls[0]
            func = tc.get("function", {})
            parsed_tool = func.get("name")
            args_str = func.get("arguments", "{}")
            try:
                parsed_params = json.loads(args_str) if args_str else {}
            except json.JSONDecodeError:
                parsed_params = {}
        else:
            parsed_tool, parsed_params = self._parse_tool_call_from_text(raw_output, tools)

        # Build tool_call string
        tool_call_str = ""
        if parsed_tool:
            param_parts = ", ".join(f"{k}={v!r}" for k, v in parsed_params.items())
            tool_call_str = f"{parsed_tool}({param_parts})"

        return ModelOutput(
            tool_name=parsed_tool or "",
            tool_call=tool_call_str,
            parameters=parsed_params,
            raw_output=raw_output,
            latency_ms=latency_ms,
        )

    def _parse_tool_call_from_text(
        self, text: str, tools: Optional[List[Dict[str, Any]]] = None
    ) -> tuple:
        """Parse tool call from raw text output."""
        if not text:
            return None, {}

        tool_names = {t["name"] for t in (tools or [])}

        patterns = [
            r"(\w+)\s*\(\s*(.*?)\s*\)",
            r"tool[_\s]*name[:\s]*[\"']?(\w+)[\"']?",
            r"function[:\s]*[\"']?(\w+)[\"']?",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                tool_name = match.group(1)
                if tool_names and tool_name not in tool_names:
                    continue
                
                params = {}
                if len(match.groups()) > 1 and match.group(2):
                    params = self._parse_params(match.group(2))
                return tool_name, params

        json_match = re.search(r'\{[^{}]*"name"[^{}]*\}', text)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                tool_name = parsed.get("name") or parsed.get("function")
                params = parsed.get("arguments") or parsed.get("parameters") or {}
                if isinstance(params, str):
                    try:
                        params = json.loads(params)
                    except json.JSONDecodeError:
                        params = {}
                return tool_name, params
            except json.JSONDecodeError:
                pass

        return None, {}

    def _parse_params(self, params_str: str) -> Dict[str, Any]:
        """Parse parameters from a string like 'key1=value1, key2=value2'."""
        params = {}
        if not params_str.strip():
            return params

        for part in params_str.split(","):
            if "=" in part:
                key, value = part.split("=", 1)
                key = key.strip()
                value = value.strip().strip("\"'")
                
                if value.lower() == "true":
                    value = True
                elif value.lower() == "false":
                    value = False
                else:
                    try:
                        value = int(value)
                    except ValueError:
                        try:
                            value = float(value)
                        except ValueError:
                            pass
                
                params[key] = value

        return params

    def unload_model(self) -> None:
        """Disconnect from servers."""
        self._live_urls = []
        self._is_loaded = False
        logger.info("Disconnected from Kimi-Audio server(s)")

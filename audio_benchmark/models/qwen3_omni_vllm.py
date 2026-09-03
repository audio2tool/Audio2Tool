"""
Qwen3-Omni vLLM Model Adapter

This adapter connects to one or more running Qwen3-Omni vLLM servers
and distributes requests across them using round-robin load balancing.

The Qwen3-Omni model uses the standard OpenAI-compatible chat API with
multimodal inputs (audio sent as base64 data URLs).

Single server usage (4 GPUs, TP=4):
    # Start server:
    vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --port 8901 --dtype bfloat16 \\
        --tensor-parallel-size 4 --trust-remote-code --max-model-len 32768

    # Run benchmark:
    python run_benchmark.py --model qwen3-omni-vllm ...

Multi-server usage (8 GPUs: 2 servers x TP=4):
    # Server 1 on GPUs 0-3:
    CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve ... --port 8901
    # Server 2 on GPUs 4-7:
    CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve ... --port 8902

    The adapter auto-discovers servers on ports 8901-8902 by default.
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


@register_model("qwen3-omni-vllm")
class Qwen3OmniVllmModel(BaseAudioModel):
    """
    Qwen3-Omni model adapter that connects to running vLLM server(s).

    Supports multiple servers for parallel inference across GPUs.
    Auto-discovers servers on ports 8901-8902 by default.
    """

    DEFAULT_BASE_PORT = 8901
    DEFAULT_NUM_PORTS = 2  # Support up to 2 servers (2 x TP=4 = 8 GPUs)
    DEFAULT_MODEL_NAME = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

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
            api_url: Single server URL (e.g. http://localhost:8901/v1/chat/completions)
            api_urls: Explicit list of server URLs
            base_port: Base port for auto-discovery (default: 8901)
            num_ports: Number of ports to probe starting from base_port (default: 2)
            model_name: Served model name (default: Qwen/Qwen3-Omni-30B-A3B-Instruct)
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
                f"No Qwen3-Omni vLLM servers reachable. Tried: {self._candidate_urls}"
            )

        self._is_loaded = True
        logger.info(
            f"Connected to {len(self._live_urls)} Qwen3-Omni vLLM server(s): "
            + ", ".join(self._live_urls)
        )

    def _next_url(self) -> str:
        """Round-robin URL selection (thread-safe)."""
        with self._rr_lock:
            url = self._live_urls[self._rr_counter % len(self._live_urls)]
            self._rr_counter += 1
        return url

    def process_audio(self, audio_path: str) -> str:
        """Load audio and return a base64-encoded WAV data URL for the API."""
        audio_tensor = _load_audio(audio_path, target_rate=16000)
        return _audio_to_base64_wav(audio_tensor)

    def _parse_signature_params(self, signature: str) -> Dict[str, Dict[str, Any]]:
        """Parse parameters from a function signature string like 'func(a: Int, b: String)'."""
        params = {}
        match = re.search(r'\(([^)]*)\)', signature)
        if not match:
            return params

        param_str = match.group(1).strip()
        if not param_str:
            return params

        for param in param_str.split(','):
            param = param.strip()
            if not param:
                continue

            if ':' in param:
                name, type_str = param.split(':', 1)
                name = name.strip()
                type_str = type_str.strip()

                type_mapping = {
                    'int': 'integer',
                    'float': 'number',
                    'bool': 'boolean',
                    'boolean': 'boolean',
                    'string': 'string',
                    'str': 'string',
                    'list': 'array',
                    'array': 'array',
                }
                json_type = type_mapping.get(type_str.lower(), 'string')
                params[name] = {"type": json_type, "description": f"Parameter: {name}"}
            else:
                params[param] = {"type": "string", "description": f"Parameter: {param}"}

        return params

    def _build_openai_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert benchmark tool schema to OpenAI function-calling format."""
        openai_tools = []
        for tool in tools:
            params = tool.get("parameters", {})

            if not params and tool.get("signature"):
                params = self._parse_signature_params(tool["signature"])

            properties = {}
            required = []
            for p_name, p_info in params.items():
                if isinstance(p_info, dict):
                    properties[p_name] = {
                        "type": p_info.get("type", "string"),
                        "description": p_info.get("description", "").strip(),
                    }
                else:
                    properties[p_name] = {"type": "string", "description": ""}
                required.append(p_name)

            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"].strip(),
                        "description": tool.get("description", "").strip(),
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                            "additionalProperties": False,
                        },
                    },
                }
            )
        return openai_tools

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
        Send audio to a Qwen3-Omni vLLM server (round-robin) and parse the tool-call response.

        Uses the prompt-based approach by default: tool definitions are embedded in the
        system prompt and the model returns text that gets parsed into structured calls.
        This is more reliable than the OpenAI tools API which requires --tool-call-parser.

        If use_openai_tools=True is passed, tries the structured API first with fallback.
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start_time = time.time()

        # Prepare audio as base64 data URL
        audio_data_url = self.process_audio(audio_path)

        # System prompt for tool calling
        if system_prompt is None:
            system_prompt = (
                "You are a voice command executor with tool calling capabilities.\n"
                "You MUST call a tool for every user request. Do NOT respond with text.\n"
                "Even if the command is vague or missing parameters, call the most appropriate tool.\n"
                "If parameters are unclear, use reasonable defaults or empty values.\n"
                "ALWAYS call a tool. NEVER ask for clarification."
            )

        use_openai_tools = kwargs.pop("use_openai_tools", False)

        if use_openai_tools:
            result = self._try_openai_tools(
                audio_data_url, audio_path, tools, system_prompt, max_new_tokens, **kwargs
            )
        else:
            result = None

        # Default path: prompt-based tool calling (most reliable)
        if result is None:
            result = self._prompt_based_request(
                audio_data_url, tools, system_prompt, max_new_tokens, **kwargs
            )

        if result is None:
            latency_ms = (time.time() - start_time) * 1000
            return ModelOutput(
                tool_name="",
                tool_call="",
                parameters={},
                raw_output="API_ERROR: all requests failed",
                latency_ms=latency_ms,
            )

        latency_ms = (time.time() - start_time) * 1000

        message = result["choices"][0]["message"]
        raw_output = message.get("content", "") or ""

        # Parse tool calls from the response
        tool_name = ""
        tool_call_str = ""
        parameters = {}
        all_tool_calls = []

        tool_calls = message.get("tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                tc_name = tc["function"]["name"]
                try:
                    tc_params = json.loads(tc["function"]["arguments"])
                except (json.JSONDecodeError, TypeError):
                    tc_params = {}
                tc_param_parts = ", ".join(f"{k}={v!r}" for k, v in tc_params.items())
                tc_call_str = f"{tc_name}({tc_param_parts})"
                all_tool_calls.append({
                    "name": tc_name,
                    "parameters": tc_params,
                    "call_string": tc_call_str,
                })

            if all_tool_calls:
                tool_name = all_tool_calls[0]["name"]
                parameters = all_tool_calls[0]["parameters"]
                tool_call_str = all_tool_calls[0]["call_string"]
            raw_output = json.dumps(tool_calls)
        else:
            # Parse from text content
            tool_name, tool_call_str, parameters = self.parse_tool_call(raw_output)
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

    def _prompt_based_request(
        self,
        audio_data_url: str,
        tools: List[Dict[str, Any]],
        system_prompt: str,
        max_new_tokens: int,
        **kwargs,
    ) -> Optional[dict]:
        """Send request with tools embedded in the system prompt (most reliable approach)."""
        tools_text = self._build_tools_in_prompt(tools)
        full_prompt = (
            f"{system_prompt}\n\n"
            "Available tools:\n"
            f"{tools_text}\n\n"
            "Output ONLY the tool call in the format: tool_name(param1=value1, param2=value2).\n"
            "If no parameters are mentioned, output: tool_name()."
        )

        payload = {
            "model": self.served_model_name,
            "messages": [
                {"role": "system", "content": full_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "audio_url", "audio_url": {"url": audio_data_url}},
                        {"type": "text", "text": "Convert this voice command to a tool call."},
                    ],
                },
            ],
            "max_tokens": max_new_tokens,
            "temperature": kwargs.get("temperature", 0.3),
            "top_p": kwargs.get("top_p", 0.95),
            "stream": False,
        }

        # Log payload size on first call
        if not hasattr(self, '_logged_payload_size'):
            payload_json = json.dumps(payload)
            logger.info(f"Payload size: {len(payload_json)/1024:.1f}KB, {len(tools)} tools")
            self._logged_payload_size = True

        max_retries = 2
        for attempt in range(max_retries + 1):
            api_url = self._next_url()
            try:
                resp = requests.post(
                    api_url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=180,
                )
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                if attempt < max_retries:
                    logger.warning(
                        f"vLLM API error (attempt {attempt+1}) for audio, "
                        f"retrying: {e}"
                    )
                else:
                    logger.error(f"vLLM API error ({api_url}): {e}")
        return None

    def _try_openai_tools(
        self,
        audio_data_url: str,
        audio_path: str,
        tools: List[Dict[str, Any]],
        system_prompt: str,
        max_new_tokens: int,
        **kwargs,
    ) -> Optional[dict]:
        """Try the OpenAI-compatible structured tools API. Returns None on failure."""
        openai_tools = self._build_openai_tools(tools)

        payload = {
            "model": self.served_model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "audio_url", "audio_url": {"url": audio_data_url}},
                        {"type": "text", "text": "Convert this voice command to a tool call."},
                    ],
                },
            ],
            "tools": openai_tools,
            "tool_choice": "required",
            "max_tokens": max_new_tokens,
            "temperature": kwargs.get("temperature", 0.3),
            "top_p": kwargs.get("top_p", 0.95),
            "stream": False,
        }

        api_url = self._next_url()
        try:
            resp = requests.post(
                api_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=180,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.info(f"OpenAI tools API not available, using prompt-based: {e}")
            return None

    def unload_model(self) -> None:
        """No local model to unload."""
        self._is_loaded = False
        self._live_urls = []
        logger.info("Disconnected from Qwen3-Omni vLLM server(s)")

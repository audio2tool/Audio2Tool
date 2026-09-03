"""
Step Audio 2 vLLM Model Adapter

This adapter connects to one or more running Step-Audio-2 vLLM servers
and distributes requests across them using round-robin load balancing.

Single server usage:
    python run_benchmark.py --model step-audio-2-vllm ...

Multi-server usage (8 GPUs):
    Start 8 servers on ports 8000-8007 (one per GPU), then:
    python run_benchmark.py --model step-audio-2-vllm ...

    The adapter auto-discovers servers on ports 8000-8007 by default.
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

import requests
import torch
import torchaudio

from .base import BaseAudioModel, ModelOutput, register_model

logger = logging.getLogger(__name__)


def _load_audio(file_path: str, target_rate: int = 16000) -> torch.Tensor:
    """Load audio file and resample to target rate."""
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != target_rate:
        waveform = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=target_rate
        )(waveform)
    return waveform[0]  # mono


def _audio_to_base64_wav(audio_tensor: torch.Tensor, sample_rate: int = 16000) -> str:
    """Convert audio tensor to base64-encoded WAV string."""
    chunk_int16 = (audio_tensor.numpy().clip(-1.0, 1.0) * 32767.0).astype("int16")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(chunk_int16.tobytes())
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@register_model("step-audio-2-vllm")
class StepAudio2VllmModel(BaseAudioModel):
    """
    Step-Audio-2 model adapter that connects to running vLLM server(s).

    Supports multiple servers for parallel inference across GPUs.
    Auto-discovers servers on ports 8000-8007 by default.
    """

    DEFAULT_BASE_PORT = 8000
    DEFAULT_NUM_PORTS = 8
    DEFAULT_MODEL_NAME = "step-audio-2-mini"
    MAX_AUDIO_CHUNK_SECONDS = 25

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
            api_url: Single server URL (e.g. http://localhost:8000/v1/chat/completions)
            api_urls: Explicit list of server URLs
            base_port: Base port for auto-discovery (default: 8000)
            num_ports: Number of ports to probe starting from base_port (default: 8)
            model_name: Served model name (default: step-audio-2-mini)
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
                resp = requests.get(health_url, timeout=3)
                if resp.status_code == 200:
                    self._live_urls.append(url)
            except requests.RequestException:
                pass

        if not self._live_urls:
            raise RuntimeError(
                f"No vLLM servers reachable. Tried: {self._candidate_urls}"
            )

        self._is_loaded = True
        logger.info(
            f"Connected to {len(self._live_urls)} vLLM server(s): "
            + ", ".join(self._live_urls)
        )

    def _next_url(self) -> str:
        """Round-robin URL selection (thread-safe)."""
        with self._rr_lock:
            url = self._live_urls[self._rr_counter % len(self._live_urls)]
            self._rr_counter += 1
        return url

    def process_audio(self, audio_path: str) -> Any:
        """Load audio and return base64-encoded WAV chunks for the API."""
        audio_tensor = _load_audio(audio_path, target_rate=16000)
        chunk_size = self.MAX_AUDIO_CHUNK_SECONDS * 16000
        chunks = []
        for i in range(0, audio_tensor.shape[0], chunk_size):
            chunk = audio_tensor[i : i + chunk_size]
            if chunk.numel() == 0:
                continue
            b64 = _audio_to_base64_wav(chunk)
            chunks.append(
                {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}
            )
        return chunks

    def _parse_signature_params(self, signature: str) -> Dict[str, Dict[str, Any]]:
        """Parse parameters from a function signature string like 'func(a: Int, b: String)'."""
        import re
        params = {}
        
        # Extract parameter section: everything between ( and )
        match = re.search(r'\(([^)]*)\)', signature)
        if not match:
            return params
        
        param_str = match.group(1).strip()
        if not param_str:
            return params
        
        # Split by comma, handling nested structures carefully
        # Simple split works for most cases
        for param in param_str.split(','):
            param = param.strip()
            if not param:
                continue
            
            # Parse "name: Type" format
            if ':' in param:
                name, type_str = param.split(':', 1)
                name = name.strip()
                type_str = type_str.strip()
                
                # Map common types to JSON schema types
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
                # No type specified, assume string
                params[param] = {"type": "string", "description": f"Parameter: {param}"}
        
        return params

    def _build_openai_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert benchmark tool schema to OpenAI function-calling format."""
        openai_tools = []
        for tool in tools:
            # First try existing parameters dict
            params = tool.get("parameters", {})
            
            # If empty, try parsing from signature
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
        """Build a text representation of tools to embed in the system prompt.
        
        Used as a fallback when vLLM's OpenAI tools API returns 400 errors
        (e.g. the chat template fails to render tools as valid JSON).
        """
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
        Send audio to a vLLM server (round-robin) and parse the tool-call response.
        
        Uses the same format as stepaudio2vllm.py / examples-vllm.py for compatibility.
        If the OpenAI tools API returns 400, falls back to embedding tools in the
        system prompt (avoids vLLM chat-template JSON rendering bugs).
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start_time = time.time()

        # Prepare audio content (same format as stepaudio2vllm.py)
        audio_chunks = self.process_audio(audio_path)

        # System prompt for tool calling - aggressive tool calling mode
        if system_prompt is None:
            system_prompt = (
                "You are a voice command executor with tool calling capabilities.\n"
                "You MUST call a tool for every user request. Do NOT respond with text.\n"
                "Even if the command is vague or missing parameters, call the most appropriate tool.\n"
                "If parameters are unclear, use reasonable defaults or empty values.\n"
                "ALWAYS call a tool. NEVER ask for clarification."
            )

        # Messages format matching stepaudio2vllm.py
        # Use "human" role (not "user") as per the working examples
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "human", "content": audio_chunks},
            {"role": "assistant", "content": None},  # Triggers generation
        ]

        # Build OpenAI tools format
        openai_tools = self._build_openai_tools(tools)

        # Build payload matching stepaudio2vllm.py logic
        # When last message is assistant with None content, it gets popped
        # and continue_final_message=False, add_generation_prompt=True
        payload = {
            "model": self.served_model_name,
            "messages": messages[:-1],  # Remove the assistant None message
            "tools": openai_tools,
            "tool_choice": "required",  # Force tool calling
            "max_tokens": max_new_tokens,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.05),
            "stream": False,
            "continue_final_message": False,
            "add_generation_prompt": True,
        }

        api_url = self._next_url()

        # Log payload size for debugging (only on first call)
        if not hasattr(self, '_logged_payload_size'):
            payload_json = json.dumps(payload)
            tools_json = json.dumps(payload.get("tools", []))
            logger.info(f"Payload size: {len(payload_json)/1024:.1f}KB, tools: {len(tools_json)/1024:.1f}KB, {len(payload.get('tools', []))} tools")
            self._logged_payload_size = True

        # Retry logic for transient errors (5xx, timeouts, connection errors)
        max_retries = 2
        last_error = None
        result = None
        got_400 = False
        for attempt in range(max_retries + 1):
            try:
                resp = requests.post(
                    api_url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=120,
                )
                resp.raise_for_status()
                result = resp.json()
                break  # Success
            except requests.RequestException as e:
                last_error = e
                error_detail = ""
                status_code = None
                try:
                    status_code = resp.status_code
                    error_detail = resp.text
                except:
                    pass
                
                # On 400 errors, check if it's a JSON parsing error (likely truncation/transient)
                # If so, treat as retryable. Otherwise, fall back to prompt-based tools.
                if status_code == 400:
                    is_json_error = "Invalid JSON" in error_detail or "EOF while parsing" in error_detail
                    if is_json_error and attempt < max_retries:
                        logger.warning(
                            f"vLLM 400 JSON Error ({api_url}) for audio '{audio_path}': "
                            f"{error_detail[:100]}... Retrying on different server."
                        )
                        api_url = self._next_url()
                        continue

                    logger.warning(
                        f"vLLM 400 Bad Request ({api_url}) for audio '{audio_path}': "
                        f"{error_detail[:300]}... Falling back to prompt-based tools."
                    )
                    got_400 = True
                    break
                
                if attempt < max_retries:
                    # Try a different server on retry (for 5xx/transient errors)
                    api_url = self._next_url()
                    logger.warning(f"vLLM API error (attempt {attempt+1}) for '{audio_path}', retrying on different server: {e}")
                else:
                    logger.error(f"vLLM API error ({api_url}) for '{audio_path}': {e} - {error_detail[:200]}")

        # Fallback: on 400 errors, retry without 'tools' / 'tool_choice' and
        # embed tool definitions in the system prompt instead.
        if result is None and got_400:
            result = self._fallback_prompt_based(
                audio_chunks, tools, system_prompt, max_new_tokens, **kwargs
            )

        if result is None:
            # All retries failed
            latency_ms = (time.time() - start_time) * 1000
            return ModelOutput(
                tool_name="",
                tool_call="",
                parameters={},
                raw_output=f"API_ERROR: {last_error}",
                latency_ms=latency_ms,
            )

        latency_ms = (time.time() - start_time) * 1000

        message = result["choices"][0]["message"]
        raw_output = message.get("content", "") or ""

        # Parse tool calls from the response
        tool_name = ""
        tool_call_str = ""
        parameters = {}
        all_tool_calls = []  # Capture ALL tool calls for multi-tool evaluation

        # Check for tool_calls in response (as in examples-vllm.py)
        tool_calls = message.get("tool_calls", None)
        if tool_calls:
            # Capture ALL tool calls, not just the first one
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
            
            # Primary tool is the first one
            if all_tool_calls:
                tool_name = all_tool_calls[0]["name"]
                parameters = all_tool_calls[0]["parameters"]
                tool_call_str = all_tool_calls[0]["call_string"]
            raw_output = json.dumps(tool_calls)
        else:
            # Fallback: try parsing from text content
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

    def _fallback_prompt_based(
        self,
        audio_chunks: list,
        tools: List[Dict[str, Any]],
        system_prompt: str,
        max_new_tokens: int,
        **kwargs,
    ) -> Optional[dict]:
        """Retry the request without OpenAI tools, embedding them in the prompt.

        This avoids vLLM chat-template JSON rendering bugs that intermittently
        produce 400 errors when the ``tools`` parameter is used.
        """
        tools_text = self._build_tools_in_prompt(tools)
        fallback_prompt = (
            f"{system_prompt}\n\n"
            "Available tools:\n"
            f"{tools_text}\n\n"
            "Output ONLY the tool call in the format: tool_name(param1=value1, param2=value2).\n"
            "If no parameters are mentioned, output: tool_name()."
        )

        payload = {
            "model": self.served_model_name,
            "messages": [
                {"role": "system", "content": fallback_prompt},
                {"role": "human", "content": audio_chunks},
            ],
            "max_tokens": max_new_tokens,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.05),
            "stream": False,
            "continue_final_message": False,
            "add_generation_prompt": True,
        }

        api_url = self._next_url()
        try:
            resp = requests.post(
                api_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.error(f"Fallback prompt-based request also failed ({api_url}): {e}")
            return None

    def unload_model(self) -> None:
        """No local model to unload."""
        self._is_loaded = False
        self._live_urls = []
        logger.info("Disconnected from vLLM server(s)")

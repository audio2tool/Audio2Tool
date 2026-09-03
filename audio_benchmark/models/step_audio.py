"""
Step Audio 2 Model Adapter

Step Audio 2 is an audio-language model from StepFun that supports
audio understanding and conversational AI tasks.

HuggingFace: stepfun-ai/Step-Audio-Chat
"""

import time
import logging
from typing import Dict, Any, Optional, List

import torch
import librosa

from .base import BaseAudioModel, ModelOutput, register_model

logger = logging.getLogger(__name__)


@register_model("step-audio-2")
class StepAudio2Model(BaseAudioModel):
    """
    Step Audio 2 model adapter for tool calling benchmark.
    
    This adapter wraps the Step-Audio-Chat model from StepFun.
    The model supports audio understanding and can generate structured
    tool calls from audio input.
    
    Default model: stepfun-ai/Step-Audio-Chat
    """
    
    DEFAULT_MODEL_PATH = "stepfun-ai/Step-Audio-Chat"
    
    def __init__(
        self,
        device: str = "cuda",
        model_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        torch_dtype: str = "auto",
        **kwargs
    ):
        super().__init__(device, model_path, config, **kwargs)
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.torch_dtype = torch_dtype
        self.tokenizer = None
        
    def load_model(self) -> None:
        """Load Step Audio 2 model and processor."""
        if self._is_loaded:
            logger.info("Model already loaded")
            return
            
        logger.info(f"Loading Step Audio 2 from {self.model_path}")
        
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
        
        try:
            # Try loading with AutoProcessor first
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
        except Exception as e:
            logger.warning(f"Could not load processor: {e}")
            # Fallback to tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            self.processor = None
            
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=dtype,
                device_map=self.device if self.device != "cpu" else None,
                trust_remote_code=True,
            )
        except Exception as e:
            logger.warning(f"Could not load with AutoModelForCausalLM: {e}")
            # Try alternative model class
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(
                self.model_path,
                torch_dtype=dtype,
                device_map=self.device if self.device != "cpu" else None,
                trust_remote_code=True,
            )
        
        if self.device == "cpu":
            self.model = self.model.to("cpu")
        
        self._is_loaded = True
        logger.info(f"Step Audio 2 loaded successfully on {self.device}")
        
    def process_audio(self, audio_path: str) -> Any:
        """
        Load and process audio file for Step Audio 2.
        
        Args:
            audio_path: Path to WAV audio file
            
        Returns:
            Audio array at appropriate sample rate
        """
        # Step Audio typically expects 16kHz audio
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
        
        # Build prompt with tools
        tool_prompt = self._build_tool_prompt(tools)
        
        if system_prompt is None:
            system_prompt = (
                "You are an AI assistant that converts voice commands into tool calls. "
                "Listen to the audio and determine which tool should be called. "
                "Output ONLY the tool call in the format: tool_name(param1=value1, param2=value2). "
                "If no parameters are mentioned, output: tool_name()."
            )
        
        user_prompt = (
            f"Available tools:\n{tool_prompt}\n\n"
            "Convert this voice command to a tool call."
        )
        
        # Build inputs based on available processor/tokenizer
        try:
            if self.processor and hasattr(self.processor, 'apply_chat_template'):
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
            elif self.processor:
                # Processor without chat template
                full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
                inputs = self.processor(
                    text=full_prompt,
                    audios=[audio],
                    return_tensors="pt",
                    padding=True
                )
            else:
                # Tokenizer only
                full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
                inputs = self.tokenizer(
                    full_prompt,
                    return_tensors="pt",
                    padding=True
                )
                # Handle audio separately if model supports it
                if hasattr(self.model, 'encode_audio'):
                    audio_tensor = torch.tensor(audio).unsqueeze(0)
                    audio_features = self.model.encode_audio(audio_tensor.to(self.model.device))
                    inputs['audio_features'] = audio_features
                    
        except Exception as e:
            logger.warning(f"Error in input preparation: {e}")
            # Fallback
            full_prompt = f"{system_prompt}\n\n{user_prompt}\n\nResponse:"
            if self.tokenizer:
                inputs = self.tokenizer(full_prompt, return_tensors="pt", padding=True)
            else:
                inputs = self.processor(text=full_prompt, return_tensors="pt", padding=True)
        
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
        input_len = inputs.get('input_ids', torch.tensor([[]])).shape[1]
        
        if self.processor and hasattr(self.processor, 'batch_decode'):
            raw_output = self.processor.batch_decode(
                generated_ids[:, input_len:],
                skip_special_tokens=True
            )[0]
        elif self.tokenizer:
            raw_output = self.tokenizer.batch_decode(
                generated_ids[:, input_len:],
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

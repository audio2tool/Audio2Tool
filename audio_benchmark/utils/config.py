"""
Configuration management utilities.
"""

import csv
import json
import re
import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, Optional, List


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    model_path: Optional[str] = None
    device: str = "cuda"
    torch_dtype: str = "auto"
    extra_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""
    name: str
    data_dir: str
    filter_domain: Optional[str] = None
    filter_category: Optional[str] = None
    filter_tool: Optional[str] = None
    max_samples: Optional[int] = None
    speaker_idx: Optional[int] = None
    speakers_per_query: Optional[int] = None
    extra_args: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class BenchmarkConfig:
    """Complete benchmark configuration."""
    # Models to evaluate
    models: List[ModelConfig] = field(default_factory=list)
    
    # Datasets to evaluate on
    datasets: List[DatasetConfig] = field(default_factory=list)
    
    # Evaluation settings
    results_dir: str = "./results"
    max_speakers_per_query: int = 1
    save_raw_outputs: bool = True
    continue_on_error: bool = True
    
    # Generation settings
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 1.0
    seed: Optional[int] = None
    
    # Parallelism (for API-based models like vLLM)
    num_workers: int = 1
    
    # System prompt
    system_prompt: Optional[str] = None
    
    # Tools taxonomy CSV (overrides auto-generated schemas)
    tools_file: Optional[str] = None
    
    # Logging
    log_level: str = "INFO"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkConfig':
        """Create from dictionary."""
        # Convert model configs, handling extra arguments
        models = []
        for m in data.get('models', []):
            if isinstance(m, dict):
                # Extract known fields
                known_fields = {'name', 'model_path', 'device', 'torch_dtype', 'extra_args'}
                model_data = {}
                extra = {}
                for k, v in m.items():
                    if k in known_fields:
                        model_data[k] = v
                    else:
                        extra[k] = v
                # Merge extra args
                if extra:
                    existing_extra = model_data.get('extra_args', {})
                    existing_extra.update(extra)
                    model_data['extra_args'] = existing_extra
                models.append(ModelConfig(**model_data))
            else:
                models.append(m)
        
        # Convert dataset configs
        datasets = [
            DatasetConfig(**d) if isinstance(d, dict) else d
            for d in data.get('datasets', [])
        ]
        
        return cls(
            models=models,
            datasets=datasets,
            results_dir=data.get('results_dir', './results'),
            max_speakers_per_query=data.get('max_speakers_per_query', 1),
            save_raw_outputs=data.get('save_raw_outputs', True),
            continue_on_error=data.get('continue_on_error', True),
            max_new_tokens=data.get('max_new_tokens', 256),
            do_sample=data.get('do_sample', False),
            temperature=data.get('temperature', 1.0),
            seed=data.get('seed'),
            num_workers=data.get('num_workers', 1),
            system_prompt=data.get('system_prompt'),
            tools_file=data.get('tools_file'),
            log_level=data.get('log_level', 'INFO'),
        )


def load_config(path: str) -> BenchmarkConfig:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        path: Path to configuration file
        
    Returns:
        BenchmarkConfig object
    """
    path = Path(path)
    
    with open(path, 'r') as f:
        if path.suffix in ['.yaml', '.yml']:
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
            
    return BenchmarkConfig.from_dict(data)


def load_tools_from_csv(csv_path: str) -> List[Dict[str, Any]]:
    """
    Load tool definitions from a taxonomy CSV file.

    Skips deprecated tools. Returns a list of tool dicts compatible with
    the evaluator and the OpenAI function-calling schema builder::

        {"name": "...", "description": "...", "parameters": {
            "param": {"type": "string", "description": "..."}}}
    """
    tools: List[Dict[str, Any]] = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("deprecated", "").lower() == "true":
                continue

            name = row.get("tool_name", "").strip()
            if not name:
                continue

            signature = row.get("signature", "")
            description = row.get("description", "")
            constraints = row.get("argument_constraints", "")
            defaults = row.get("argument_defaults", "")

            # Parse parameters from the signature
            params: Dict[str, Dict[str, str]] = {}
            sig_match = re.search(r"\((.+)\)", signature)
            if sig_match:
                for part in sig_match.group(1).split(","):
                    part = part.strip()
                    if ":" in part:
                        p_name, p_type = part.split(":", 1)
                        p_name = p_name.strip()
                        p_type = p_type.strip().rstrip("?")
                        params[p_name] = {
                            "type": "string",
                            "description": f"Type: {p_type}",
                        }

            # Enrich with allowed values from constraints
            if constraints:
                for chunk in constraints.split("],"):
                    chunk = chunk.strip().rstrip("]")
                    if ":" in chunk:
                        cname, cvals = chunk.split(":", 1)
                        cname = cname.strip()
                        if cname in params:
                            params[cname]["description"] += (
                                f". Allowed: [{cvals.strip()}]"
                            )

            # Enrich with default values
            if defaults:
                for chunk in defaults.split(","):
                    chunk = chunk.strip()
                    if "=" in chunk:
                        dname, dval = chunk.split("=", 1)
                        dname = dname.strip()
                        if dname in params:
                            params[dname]["description"] += (
                                f". Default: {dval.strip()}"
                            )

            tools.append({
                "name": name,
                "description": description,
                "parameters": params,
            })

    return tools


def build_tools_prompt_section(tools: List[Dict[str, Any]]) -> str:
    """
    Render tool definitions as a human-readable text block suitable for
    inclusion in a system prompt.
    """
    lines = ["", "Available tools:"]
    for t in tools:
        params = t.get("parameters", {})
        if params:
            sig_parts = ", ".join(f"{k}" for k in params)
            lines.append(f"- {t['name']}({sig_parts}): {t['description']}")
        else:
            lines.append(f"- {t['name']}(): {t['description']}")
    return "\n".join(lines)


def save_config(config: BenchmarkConfig, path: str) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration to save
        path: Output path
    """
    path = Path(path)
    
    data = config.to_dict()
    
    with open(path, 'w') as f:
        if path.suffix in ['.yaml', '.yml']:
            yaml.dump(data, f, default_flow_style=False)
        else:
            json.dump(data, f, indent=2)

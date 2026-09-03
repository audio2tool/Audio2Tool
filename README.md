# Audio2Tool Benchmark

A modular and extensible framework for benchmarking audio-language models on tool calling tasks.

## Features

- **Registry-based model system**: Easy to add new models
- **Multiple dataset formats**: Loaders for all Audio2Tool dataset tiers (1-8)
- **Comprehensive metrics**: Tool accuracy, parameter accuracy, exact match, latency
- **Configuration-driven**: Run benchmarks via YAML/JSON config files
- **Detailed analysis**: Per-domain, per-tool, and error analysis
- **Multi-GPU support**: Parallel evaluation across GPUs, with optional model parallelism

## Installation

```bash
pip install -r requirements.txt
```

## Dataset

The benchmark runs on the Audio2Tool dataset release, which contains tiered
audio tool-calling queries (`tier1_direct`, `tier2_parametric`,
`tier3_multi_intent`, `tier4_implicit`, `tier5_needle`, `tier6_correction`,
`tier7_multiturn`, `tier8_intent_blending`) and a `tools_registry.csv`
describing the tool taxonomy. Point the config entries `data_dir` and
`tools_file` at your local copy of the dataset.

## Quick Start

### Using Configuration File

```bash
# Edit configs/example_config.yaml first (dataset paths, model selection)
python audio_benchmark/run_benchmark.py --config configs/example_config.yaml

# Multi-GPU (for locally hosted models)
python audio_benchmark/run_benchmark_multigpu.py --config configs/example_config.yaml --num-gpus 8
```

### Using Command Line

```bash
# Single model evaluation
python audio_benchmark/run_benchmark.py \
    --model qwen2-audio \
    --dataset tier1 \
    --data-dir /path/to/Audio2ToolDataset/public/tier1_direct \
    --max-samples 100

# Multiple models
python audio_benchmark/run_benchmark.py \
    --models qwen2-audio kimi-audio \
    --dataset tier1 \
    --data-dir /path/to/Audio2ToolDataset/public/tier1_direct

# Filter by domain
python audio_benchmark/run_benchmark.py \
    --model qwen2-audio \
    --dataset tier1 \
    --data-dir /path/to/Audio2ToolDataset/public/tier1_direct \
    --filter-domain smart_car

# List available models and datasets
python audio_benchmark/run_benchmark.py --list
```

### Using Python API

```python
from audio_benchmark import get_model, get_dataset, BenchmarkEvaluator

# Initialize model
model = get_model("qwen2-audio", device="cuda")

# Initialize and load dataset
dataset = get_dataset(
    "tier1",
    data_dir="/path/to/Audio2ToolDataset/public/tier1_direct",
    max_samples=100
)
dataset.load()

# Run evaluation
evaluator = BenchmarkEvaluator(
    model=model,
    dataset=dataset,
    results_dir="./results"
)
metrics = evaluator.run()

# Print results
print(metrics)
```

## Available Models

| Model | Backend | Description |
|-------|---------|-------------|
| `qwen2-audio` | local (HF) | Qwen/Qwen2-Audio-7B-Instruct |
| `qwen3-omni` | local (HF) | Qwen/Qwen3-Omni-30B-A3B-Instruct (MoE, 2+ GPUs) |
| `qwen3-omni-vllm` | vLLM server | Qwen3-Omni via external vLLM server(s) |
| `kimi-audio` | local (HF) | moonshotai/Kimi-Audio-7B-Instruct |
| `kimi-audio-vllm` | server | Kimi-Audio via OpenAI-compatible server(s) |
| `step-audio-2` | local (HF) | stepfun-ai Step-Audio-2 |
| `step-audio-2-vllm` | vLLM server | Step-Audio-2-mini via external vLLM server |
| `audio-flamingo3-vllm` | vLLM server | NVIDIA Audio Flamingo 3 |
| `whisper-qwen3-vllm` | cascaded | Whisper ASR → Qwen3 (text LLM) via vLLM |
| `whisper-gemma3-vllm` | cascaded | Whisper ASR → Gemma3 (text LLM) via vLLM |

Server-backed models auto-discover running servers and distribute requests via
round-robin load balancing. See the corresponding files in `audio_benchmark/models/`
for server launch instructions.

## Available Datasets

| Dataset | Description |
|---------|-------------|
| `tier1` | Direct tool calling queries |
| `tier2` | Parametric queries |
| `tier3` | Multi-intent queries |
| `tier4` | Implicit intent queries |
| `tier5` | Needle-in-haystack (long context) |
| `tier6` | Self-correction queries |
| `tier7` | Multi-turn conversational queries |
| `tier8` | Intent blending queries |

## Adding New Models

Create a new file in `audio_benchmark/models/` and use the `@register_model` decorator:

```python
from .base import BaseAudioModel, ModelOutput, register_model

@register_model("my-new-model")
class MyNewModel(BaseAudioModel):
    def load_model(self):
        # Load your model here
        self.model = ...
        self.processor = ...
        self._is_loaded = True

    def process_audio(self, audio_path: str):
        # Process audio file
        return processed_audio

    def generate(self, audio_path, tools, system_prompt=None, **kwargs):
        # Run inference and return ModelOutput
        return ModelOutput(
            tool_name="predicted_tool",
            tool_call="predicted_tool(param=value)",
            parameters={"param": "value"},
            raw_output="raw model output"
        )
```

## Adding New Datasets

Create a new file in `audio_benchmark/datasets/` and use the `@register_dataset` decorator:

```python
from .base import BaseDataset, QuerySample, register_dataset

@register_dataset("my-dataset")
class MyDataset(BaseDataset):
    def load(self):
        # Load your dataset here
        for item in your_data:
            sample = QuerySample(
                query_idx=item["idx"],
                query_text=item["query"],
                tool_name=item["tool"],
                tool_call=item["tool_call"],
                extracted_params=item["params"],
                audio_paths=item["audio_files"],
            )
            self.samples.append(sample)
        self._loaded = True

    def get_tools_schema(self):
        # Return list of tool definitions
        return [...]
```

## Metrics

The benchmark computes the following metrics:

| Metric | Description |
|--------|-------------|
| **Tool Accuracy** | Fraction of samples where tool name is correctly predicted |
| **Exact Match** | Fraction where both tool and all parameters are correct |
| **Param Precision** | Fraction of predicted parameters that are correct |
| **Param Recall** | Fraction of ground truth parameters that were predicted |
| **Param F1** | Harmonic mean of precision and recall |
| **Latency** | Inference time statistics (mean, p50, p95, p99) |

## Output Structure

Results are saved in the following structure:

```
results/
├── qwen2-audio_20260215_120000/
│   ├── metrics.json      # Aggregated metrics
│   ├── results.json      # Per-sample results
│   └── summary.txt       # Human-readable summary
├── kimi-audio_20260215_130000/
│   └── ...
└── comparison_20260215_140000.json  # Cross-model comparison
```

## Configuration Reference

See `configs/example_config.yaml` for all available options:

```yaml
models:
  - name: model_name          # Registered model name
    model_path: path/or/id    # HuggingFace ID or local path
    device: cuda              # Device (cuda, cpu)
    torch_dtype: auto         # Data type (auto, float16, bfloat16)

datasets:
  - name: dataset_name        # Registered dataset name
    data_dir: /path/to/data   # Data directory
    max_samples: null         # Limit samples (null = all)
    filter_domain: null       # Filter by domain
    speakers_per_query: 1     # Speakers to use per query

results_dir: ./results
max_speakers_per_query: 1
save_raw_outputs: true
continue_on_error: true

max_new_tokens: 256
do_sample: false
temperature: 1.0
system_prompt: null  # Uses default if null
log_level: INFO

tools_file: /path/to/Audio2ToolDataset/tools_registry.csv
```

## License

MIT License

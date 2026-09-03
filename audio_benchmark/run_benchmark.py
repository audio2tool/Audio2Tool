#!/usr/bin/env python3
"""
Main benchmark runner script.

Usage:
    # Run with config file
    python run_benchmark.py --config configs/benchmark_config.yaml
    
    # Run specific model on specific dataset
    python run_benchmark.py --model qwen2-audio --dataset tier1 --data-dir /path/to/tier1
    
    # Run with multiple models
    python run_benchmark.py --models qwen2-audio kimi-audio step-audio-2 --dataset tier1
    
    # Limit samples for testing
    python run_benchmark.py --model qwen2-audio --dataset tier1 --max-samples 100
"""

import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from audio_benchmark.models import get_model, list_models, MODEL_REGISTRY
from audio_benchmark.datasets import get_dataset, list_datasets, DATASET_REGISTRY
from audio_benchmark.evaluation import BenchmarkEvaluator, ToolCallingMetrics
from audio_benchmark.utils import (
    load_config, setup_logging, BenchmarkConfig,
    load_tools_from_csv, build_tools_prompt_section,
)


logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all libraries."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Set random seed: {seed}")


def run_single_benchmark(
    model_name: str,
    dataset_name: str,
    data_dir: str,
    results_dir: str,
    model_kwargs: dict = None,
    dataset_kwargs: dict = None,
    eval_kwargs: dict = None,
    generation_kwargs: dict = None,
    tools_schema: list = None,
) -> ToolCallingMetrics:
    """
    Run benchmark for a single model on a single dataset.
    
    Args:
        model_name: Name of the model to evaluate
        dataset_name: Name of the dataset to use
        data_dir: Path to dataset directory
        results_dir: Path to save results
        model_kwargs: Additional arguments for model initialization
        dataset_kwargs: Additional arguments for dataset initialization
        eval_kwargs: Additional arguments for evaluator
        generation_kwargs: Additional arguments for model generation
        tools_schema: Optional pre-loaded tool schemas (overrides dataset auto-gen)
        
    Returns:
        ToolCallingMetrics with evaluation results
    """
    model_kwargs = model_kwargs or {}
    dataset_kwargs = dataset_kwargs or {}
    eval_kwargs = eval_kwargs or {}
    generation_kwargs = generation_kwargs or {}
    
    logger.info(f"Running benchmark: {model_name} on {dataset_name}")
    logger.info(f"Data directory: {data_dir}")
    
    # Initialize model
    logger.info(f"Initializing model: {model_name}")
    model = get_model(model_name, **model_kwargs)
    
    # Initialize dataset
    logger.info(f"Initializing dataset: {dataset_name}")
    dataset = get_dataset(dataset_name, data_dir=data_dir, **dataset_kwargs)
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset.load()
    logger.info(f"Dataset loaded: {len(dataset)} samples")
    
    # Override tool schemas if provided from taxonomy CSV
    if tools_schema is not None:
        dataset.set_tools_schema(tools_schema)
        logger.info(f"Using {len(tools_schema)} tools from taxonomy CSV")
    
    # Print dataset statistics
    stats = dataset.get_statistics()
    logger.info(f"Dataset statistics:")
    logger.info(f"  Total queries: {stats['total_queries']}")
    logger.info(f"  Total audio files: {stats['total_audio_files']}")
    logger.info(f"  Unique tools: {stats['unique_tools']}")
    logger.info(f"  Domains: {stats['domains']}")
    
    # Initialize evaluator
    evaluator = BenchmarkEvaluator(
        model=model,
        dataset=dataset,
        results_dir=results_dir,
        **eval_kwargs
    )
    
    # Run evaluation
    logger.info("Starting evaluation...")
    metrics = evaluator.run(**generation_kwargs)
    
    # Print results
    logger.info("\n" + str(metrics))
    
    # Cleanup
    model.unload_model()
    
    return metrics


def run_from_config(config_path: str) -> dict:
    """
    Run benchmark from configuration file.
    
    Args:
        config_path: Path to YAML/JSON configuration file
        
    Returns:
        Dictionary mapping (model, dataset) to metrics
    """
    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    
    # Set reproducibility seed
    if config.seed is not None:
        set_seed(config.seed)
    
    # Load tools from taxonomy CSV if provided
    csv_tools = None
    system_prompt = config.system_prompt
    if config.tools_file:
        csv_path = Path(config.tools_file)
        if not csv_path.is_absolute():
            csv_path = Path(config_path).parent / csv_path
        logger.info(f"Loading tool taxonomy from {csv_path}")
        csv_tools = load_tools_from_csv(str(csv_path))
        logger.info(f"Loaded {len(csv_tools)} tools from taxonomy CSV (passed as structured tools)")
    
    all_results = {}
    
    for model_config in config.models:
        for dataset_config in config.datasets:
            model_kwargs = {
                'device': model_config.device,
                'model_path': model_config.model_path,
                'torch_dtype': model_config.torch_dtype,
                **model_config.extra_args
            }
            
            dataset_kwargs = {
                'filter_domain': dataset_config.filter_domain,
                'filter_category': dataset_config.filter_category,
                'filter_tool': dataset_config.filter_tool,
                'max_samples': dataset_config.max_samples,
                'speaker_idx': dataset_config.speaker_idx,
                'speakers_per_query': dataset_config.speakers_per_query,
                **dataset_config.extra_args
            }
            
            eval_kwargs = {
                'system_prompt': system_prompt,
                'max_speakers_per_query': config.max_speakers_per_query,
                'save_raw_outputs': config.save_raw_outputs,
            }
            
            generation_kwargs = {
                'max_new_tokens': config.max_new_tokens,
                'do_sample': config.do_sample,
                'temperature': config.temperature,
                'continue_on_error': config.continue_on_error,
                'num_workers': getattr(config, 'num_workers', 1),
            }
            
            try:
                metrics = run_single_benchmark(
                    model_name=model_config.name,
                    dataset_name=dataset_config.name,
                    data_dir=dataset_config.data_dir,
                    results_dir=config.results_dir,
                    model_kwargs=model_kwargs,
                    dataset_kwargs=dataset_kwargs,
                    eval_kwargs=eval_kwargs,
                    generation_kwargs=generation_kwargs,
                    tools_schema=csv_tools,
                )
                
                key = (model_config.name, dataset_config.name)
                all_results[key] = metrics
                
            except Exception as e:
                logger.error(f"Failed to run {model_config.name} on {dataset_config.name}: {e}")
                if not config.continue_on_error:
                    raise
    
    # Save combined results
    save_combined_results(all_results, config.results_dir)
    
    return all_results


def save_combined_results(results: dict, results_dir: str) -> None:
    """Save combined results from multiple model/dataset combinations."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Build comparison table
    comparison = []
    for (model, dataset), metrics in results.items():
        comparison.append({
            "model": model,
            "dataset": dataset,
            "tool_accuracy": metrics.tool_accuracy,
            "exact_match": metrics.exact_match,
            "param_f1": metrics.param_f1,
            "latency_mean_ms": metrics.latency_mean_ms,
            "total_samples": metrics.total_samples,
        })
    
    # Save comparison
    comparison_path = results_dir / f"comparison_{timestamp}.json"
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    logger.info(f"Saved comparison to {comparison_path}")
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON")
    print("=" * 80)
    print(f"{'Model':<20} {'Dataset':<15} {'Tool Acc':<12} {'Exact Match':<12} {'Param F1':<12}")
    print("-" * 80)
    for row in comparison:
        print(f"{row['model']:<20} {row['dataset']:<15} {row['tool_accuracy']:.2%}       {row['exact_match']:.2%}        {row['param_f1']:.2%}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Audio Tool Calling Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with config file
    python run_benchmark.py --config configs/benchmark_config.yaml
    
    # Run specific model on specific dataset
    python run_benchmark.py --model qwen2-audio --dataset tier1 \\
        --data-dir /path/to/Audio2ToolDataset/tier1
    
    # Run multiple models
    python run_benchmark.py --models qwen2-audio kimi-audio --dataset tier1 \\
        --data-dir /path/to/Audio2ToolDataset/tier1
    
    # Limit samples for testing
    python run_benchmark.py --model qwen2-audio --dataset tier1 \\
        --data-dir /path/to/Audio2ToolDataset/tier1 \\
        --max-samples 100
        
    # List available models and datasets
    python run_benchmark.py --list
        """
    )
    
    # Config file option
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file (YAML or JSON)"
    )
    
    # Model options
    parser.add_argument(
        "--model", "-m",
        type=str,
        help="Single model to evaluate"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="Multiple models to evaluate"
    )
    
    # Dataset options
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        help="Dataset to use"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Path to dataset directory"
    )
    
    # Filter options
    parser.add_argument(
        "--filter-domain",
        type=str,
        help="Filter by domain (smart_car, smart_home, wearables)"
    )
    parser.add_argument(
        "--filter-category",
        type=str,
        help="Filter by category"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to evaluate"
    )
    parser.add_argument(
        "--speakers-per-query",
        type=int,
        default=1,
        help="Number of speakers to evaluate per query (default: 1)"
    )
    
    # Output options
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results",
        help="Directory to save results (default: ./results)"
    )
    
    # Generation options
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate (default: 256)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (default: cuda)"
    )
    
    # Parallelism options
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for API-based models (default: 1). "
             "Use 8 for 8 vLLM servers."
    )

    # Other options
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and datasets"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    
    # List available models/datasets
    if args.list:
        print("\nAvailable Models:")
        for name in list_models():
            print(f"  - {name}")
        print("\nAvailable Datasets:")
        for name in list_datasets():
            print(f"  - {name}")
        return
    
    # Run from config file
    if args.config:
        run_from_config(args.config)
        return
    
    # Run from command line arguments
    if not args.dataset or not args.data_dir:
        parser.error("--dataset and --data-dir are required (or use --config)")
    
    # Get models to evaluate
    models = args.models or ([args.model] if args.model else None)
    if not models:
        parser.error("--model or --models is required (or use --config)")
    
    # Run benchmarks
    all_results = {}
    for model_name in models:
        try:
            dataset_kwargs = {
                'filter_domain': args.filter_domain,
                'filter_category': args.filter_category,
                'max_samples': args.max_samples,
                'speakers_per_query': args.speakers_per_query,
            }
            
            model_kwargs = {
                'device': args.device,
            }
            
            generation_kwargs = {
                'max_new_tokens': args.max_tokens,
            }
            
            eval_kwargs = {
                'max_speakers_per_query': args.speakers_per_query,
            }
            
            generation_kwargs['num_workers'] = args.workers
            
            metrics = run_single_benchmark(
                model_name=model_name,
                dataset_name=args.dataset,
                data_dir=args.data_dir,
                results_dir=args.results_dir,
                model_kwargs=model_kwargs,
                dataset_kwargs=dataset_kwargs,
                eval_kwargs=eval_kwargs,
                generation_kwargs=generation_kwargs,
            )
            
            all_results[(model_name, args.dataset)] = metrics
            
        except Exception as e:
            logger.error(f"Failed to run {model_name}: {e}")
            raise
    
    # Save combined results if multiple models
    if len(all_results) > 1:
        save_combined_results(all_results, args.results_dir)


if __name__ == "__main__":
    main()

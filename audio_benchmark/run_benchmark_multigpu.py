#!/usr/bin/env python3
"""
Multi-GPU Benchmark Runner

Distributes benchmark workloads across multiple GPUs for faster evaluation.

Strategies:
1. model_parallel: Run different models on different GPUs simultaneously
2. data_parallel: Split dataset across GPUs for single model evaluation
3. combo_parallel: Run different (model, dataset) combinations on different GPUs

Usage:
    # Auto-detect GPUs and run all models in parallel
    python run_benchmark_multigpu.py --config configs/all_tiers_benchmark.yaml
    
    # Specify number of GPUs
    python run_benchmark_multigpu.py --config configs/all_tiers_benchmark.yaml --num-gpus 8
    
    # Use specific strategy
    python run_benchmark_multigpu.py --config configs/all_tiers_benchmark.yaml --strategy combo_parallel
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import traceback

# CRITICAL: Set spawn method before any CUDA imports
# This must happen before importing torch
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from audio_benchmark.utils import load_config, setup_logging, BenchmarkConfig

logger = logging.getLogger(__name__)


@dataclass
class GPUTask:
    """A single task to run on one or more GPUs."""
    task_id: int
    gpu_id: int  # Primary GPU (for logging/results naming)
    model_name: str
    model_path: Optional[str]
    dataset_name: str
    data_dir: str
    results_dir: str
    model_kwargs: Dict[str, Any]
    dataset_kwargs: Dict[str, Any]
    eval_kwargs: Dict[str, Any]
    generation_kwargs: Dict[str, Any]
    gpu_ids: Optional[List[int]] = None  # All GPUs for model parallelism (None = single GPU)


def get_available_gpus() -> List[int]:
    """Get list of available GPU IDs."""
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def run_single_task(task: GPUTask) -> Dict[str, Any]:
    """
    Run a single benchmark task on a specific GPU.
    This function runs in a separate process.
    """
    # Set GPU visibility for this process
    if task.gpu_ids and len(task.gpu_ids) > 1:
        # Model parallelism: make multiple GPUs visible
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in task.gpu_ids)
        num_gpus = len(task.gpu_ids)
    else:
        # Single GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(task.gpu_id)
        num_gpus = 1
    
    # Re-import after setting CUDA_VISIBLE_DEVICES
    import torch
    torch.cuda.set_device(0)  # Device 0 is always the first visible GPU
    
    # Setup logging for this process
    setup_logging(level="INFO")
    gpu_label = ",".join(str(g) for g in task.gpu_ids) if task.gpu_ids else str(task.gpu_id)
    proc_logger = logging.getLogger(f"GPU-{gpu_label}")
    
    proc_logger.info(f"Starting task {task.task_id}: {task.model_name} on {task.dataset_name} (GPU(s) {gpu_label})")
    
    try:
        # Import here to ensure CUDA device is set
        import random
        import numpy as np
        from audio_benchmark.models import get_model
        from audio_benchmark.datasets import get_dataset
        from audio_benchmark.evaluation import BenchmarkEvaluator
        
        # Set reproducibility seeds before any inference
        seed = task.generation_kwargs.get('seed')
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            proc_logger.info(f"Set random seed: {seed}")
        
        # Override device and GPU settings
        model_kwargs = task.model_kwargs.copy()
        model_kwargs['device'] = 'cuda:0'
        model_kwargs['num_gpus'] = num_gpus
        if num_gpus > 1:
            # GPU IDs are re-indexed from 0 due to CUDA_VISIBLE_DEVICES
            model_kwargs['gpu_ids'] = list(range(num_gpus))
            proc_logger.info(f"Model parallelism enabled: {num_gpus} GPUs")
        
        # Initialize model
        proc_logger.info(f"Loading model: {task.model_name}")
        model = get_model(task.model_name, **model_kwargs)
        
        # Initialize dataset
        proc_logger.info(f"Loading dataset: {task.dataset_name}")
        dataset = get_dataset(task.dataset_name, data_dir=task.data_dir, **task.dataset_kwargs)
        dataset.load()
        proc_logger.info(f"Dataset loaded: {len(dataset)} samples")
        
        # Create task-specific results directory
        task_results_dir = Path(task.results_dir) / f"{task.model_name}_{task.dataset_name}_gpu{gpu_label}"
        
        # Initialize evaluator
        evaluator = BenchmarkEvaluator(
            model=model,
            dataset=dataset,
            results_dir=str(task_results_dir),
            **task.eval_kwargs
        )
        
        # Run evaluation — strip seed from generation_kwargs (already applied above)
        run_kwargs = {k: v for k, v in task.generation_kwargs.items() if k != 'seed'}
        start_time = time.time()
        metrics = evaluator.run(
            tqdm_position=task.gpu_id,
            tqdm_desc=f"GPU {task.gpu_id} {task.dataset_name}",
            **run_kwargs
        )
        elapsed = time.time() - start_time
        
        proc_logger.info(f"Task {task.task_id} completed in {elapsed:.1f}s")
        proc_logger.info(f"Tool Accuracy: {metrics.tool_accuracy:.2%}, Exact Match: {metrics.exact_match:.2%}")
        
        # Cleanup
        model.unload_model()
        torch.cuda.empty_cache()
        
        return {
            "task_id": task.task_id,
            "gpu_id": task.gpu_id,
            "model": task.model_name,
            "dataset": task.dataset_name,
            "success": True,
            "metrics": metrics.to_dict(),
            "elapsed_seconds": elapsed,
        }
        
    except Exception as e:
        proc_logger.error(f"Task {task.task_id} failed: {e}")
        traceback.print_exc()
        return {
            "task_id": task.task_id,
            "gpu_id": task.gpu_id,
            "model": task.model_name,
            "dataset": task.dataset_name,
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def create_tasks_combo_parallel(
    config: BenchmarkConfig, gpus: List[int], num_gpus_per_model: int = 1
) -> List[GPUTask]:
    """
    Create tasks for combo_parallel strategy.
    Each (model, dataset) combination runs on a separate GPU (or group of GPUs).
    
    Args:
        config: Benchmark configuration.
        gpus: List of available GPU IDs.
        num_gpus_per_model: Number of GPUs to allocate per model instance
                           for model parallelism. Default 1 (no parallelism).
    """
    tasks = []
    task_id = 0
    
    # Create all (model, dataset) combinations
    combinations = []
    for model_config in config.models:
        for dataset_config in config.datasets:
            combinations.append((model_config, dataset_config))
    
    # Build GPU groups for model parallelism
    if num_gpus_per_model > 1:
        # Chunk GPUs into groups of num_gpus_per_model
        gpu_groups = []
        for i in range(0, len(gpus), num_gpus_per_model):
            group = gpus[i:i + num_gpus_per_model]
            if len(group) == num_gpus_per_model:
                gpu_groups.append(group)
        if not gpu_groups:
            raise ValueError(
                f"Not enough GPUs ({len(gpus)}) for num_gpus_per_model={num_gpus_per_model}. "
                f"Need at least {num_gpus_per_model} GPUs."
            )
        logger.info(f"Model parallelism: {len(gpu_groups)} GPU groups of {num_gpus_per_model}: {gpu_groups}")
    else:
        gpu_groups = [[g] for g in gpus]
    
    # Assign to GPU groups round-robin
    for i, (model_config, dataset_config) in enumerate(combinations):
        group = gpu_groups[i % len(gpu_groups)]
        primary_gpu = group[0]
        
        model_kwargs = {
            'model_path': model_config.model_path,
            'torch_dtype': model_config.torch_dtype,
            **model_config.extra_args
        }
        
        dataset_kwargs = {
            'filter_domain': dataset_config.filter_domain,
            'filter_category': dataset_config.filter_category,
            'max_samples': dataset_config.max_samples,
            'speaker_idx': dataset_config.speaker_idx,
            'speakers_per_query': dataset_config.speakers_per_query,
            **dataset_config.extra_args
        }
        
        eval_kwargs = {
            'system_prompt': config.system_prompt,
            'max_speakers_per_query': config.max_speakers_per_query,
            'save_raw_outputs': config.save_raw_outputs,
        }
        
        generation_kwargs = {
            'max_new_tokens': config.max_new_tokens,
            'do_sample': config.do_sample,
            'temperature': config.temperature,
            'continue_on_error': config.continue_on_error,
            'seed': config.seed,
        }
        
        task = GPUTask(
            task_id=task_id,
            gpu_id=primary_gpu,
            model_name=model_config.name,
            model_path=model_config.model_path,
            dataset_name=dataset_config.name,
            data_dir=dataset_config.data_dir,
            results_dir=config.results_dir,
            model_kwargs=model_kwargs,
            dataset_kwargs=dataset_kwargs,
            eval_kwargs=eval_kwargs,
            generation_kwargs=generation_kwargs,
            gpu_ids=group if num_gpus_per_model > 1 else None,
        )
        tasks.append(task)
        task_id += 1
        
    return tasks


def create_tasks_model_parallel(
    config: BenchmarkConfig, gpus: List[int], num_gpus_per_model: int = 1
) -> List[GPUTask]:
    """
    Create tasks for model_parallel strategy.
    Each model runs on a dedicated GPU (or group of GPUs), processing all datasets sequentially.
    """
    tasks = []
    task_id = 0
    
    # Build GPU groups for model parallelism
    if num_gpus_per_model > 1:
        gpu_groups = []
        for i in range(0, len(gpus), num_gpus_per_model):
            group = gpus[i:i + num_gpus_per_model]
            if len(group) == num_gpus_per_model:
                gpu_groups.append(group)
        if not gpu_groups:
            raise ValueError(
                f"Not enough GPUs ({len(gpus)}) for num_gpus_per_model={num_gpus_per_model}"
            )
    else:
        gpu_groups = [[g] for g in gpus]
    
    # Assign models to GPU groups
    for i, model_config in enumerate(config.models):
        group = gpu_groups[i % len(gpu_groups)]
        primary_gpu = group[0]
        
        for dataset_config in config.datasets:
            model_kwargs = {
                'model_path': model_config.model_path,
                'torch_dtype': model_config.torch_dtype,
                **model_config.extra_args
            }
            
            dataset_kwargs = {
                'filter_domain': dataset_config.filter_domain,
                'filter_category': dataset_config.filter_category,
                'max_samples': dataset_config.max_samples,
                'speaker_idx': dataset_config.speaker_idx,
                'speakers_per_query': dataset_config.speakers_per_query,
                **dataset_config.extra_args
            }
            
            eval_kwargs = {
                'system_prompt': config.system_prompt,
                'max_speakers_per_query': config.max_speakers_per_query,
                'save_raw_outputs': config.save_raw_outputs,
            }
            
            generation_kwargs = {
                'max_new_tokens': config.max_new_tokens,
                'do_sample': config.do_sample,
                'temperature': config.temperature,
                'continue_on_error': config.continue_on_error,
                'seed': config.seed,
            }
            
            task = GPUTask(
                task_id=task_id,
                gpu_id=primary_gpu,
                model_name=model_config.name,
                model_path=model_config.model_path,
                dataset_name=dataset_config.name,
                data_dir=dataset_config.data_dir,
                results_dir=config.results_dir,
                model_kwargs=model_kwargs,
                dataset_kwargs=dataset_kwargs,
                eval_kwargs=eval_kwargs,
                generation_kwargs=generation_kwargs,
                gpu_ids=group if num_gpus_per_model > 1 else None,
            )
            tasks.append(task)
            task_id += 1
            
    return tasks


def run_tasks_parallel(tasks: List[GPUTask], max_concurrent: int) -> List[Dict[str, Any]]:
    """Run tasks in parallel using multiprocessing."""
    results = []
    
    # Group tasks by GPU group to avoid conflicts
    # Use a hashable key from the GPU group
    gpu_tasks: Dict[str, List[GPUTask]] = {}
    for task in tasks:
        key = ",".join(str(g) for g in task.gpu_ids) if task.gpu_ids else str(task.gpu_id)
        if key not in gpu_tasks:
            gpu_tasks[key] = []
        gpu_tasks[key].append(task)
    
    logger.info(f"Running {len(tasks)} tasks across {len(gpu_tasks)} GPU group(s)")
    for gpu_key, gpu_task_list in gpu_tasks.items():
        logger.info(f"  GPU(s) {gpu_key}: {len(gpu_task_list)} tasks")
    
    # Use ProcessPoolExecutor with spawn context for CUDA compatibility
    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=min(max_concurrent, len(gpu_tasks)), mp_context=ctx) as executor:
        # Submit first task for each GPU group
        running = {}
        pending = {gpu_key: list(task_list) for gpu_key, task_list in gpu_tasks.items()}
        
        # Track which GPU group each task belongs to
        task_gpu_key = {}
        
        # Initial submission - one task per GPU group
        for gpu_key, task_list in pending.items():
            if task_list:
                task = task_list.pop(0)
                future = executor.submit(run_single_task, task)
                running[future] = task
                task_gpu_key[id(task)] = gpu_key
                logger.info(f"Submitted task {task.task_id}: {task.model_name}/{task.dataset_name} on GPU(s) {gpu_key}")
        
        # Process completions and submit new tasks
        while running:
            # Wait for any task to complete
            done_futures = []
            for future in list(running.keys()):
                if future.done():
                    done_futures.append(future)
            
            if not done_futures:
                time.sleep(1)
                continue
            
            for future in done_futures:
                task = running.pop(future)
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result["success"]:
                        logger.info(f"Task {task.task_id} completed: {task.model_name}/{task.dataset_name}")
                    else:
                        logger.error(f"Task {task.task_id} failed: {result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    logger.error(f"Task {task.task_id} exception: {e}")
                    results.append({
                        "task_id": task.task_id,
                        "gpu_id": task.gpu_id,
                        "model": task.model_name,
                        "dataset": task.dataset_name,
                        "success": False,
                        "error": str(e),
                    })
                
                # Submit next task for this GPU group if available
                gpu_key = task_gpu_key.get(id(task), str(task.gpu_id))
                if pending.get(gpu_key):
                    next_task = pending[gpu_key].pop(0)
                    next_future = executor.submit(run_single_task, next_task)
                    running[next_future] = next_task
                    task_gpu_key[id(next_task)] = gpu_key
                    logger.info(f"Submitted task {next_task.task_id}: {next_task.model_name}/{next_task.dataset_name} on GPU(s) {gpu_key}")
    
    return results


def save_combined_results(results: List[Dict[str, Any]], results_dir: str) -> None:
    """Save combined results from all tasks."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save full results
    full_results_path = results_dir / f"multigpu_results_{timestamp}.json"
    with open(full_results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved full results to {full_results_path}")
    
    # Build comparison table
    successful = [r for r in results if r.get("success")]
    comparison = []
    for r in successful:
        metrics = r.get("metrics", {})
        comparison.append({
            "model": r["model"],
            "dataset": r["dataset"],
            "gpu": r["gpu_id"],
            "tool_accuracy": metrics.get("tool_accuracy", 0),
            "exact_match": metrics.get("exact_match", 0),
            "param_f1": metrics.get("param_f1", 0),
            "latency_mean_ms": metrics.get("latency_mean_ms", 0),
            "total_samples": metrics.get("total_samples", 0),
            "elapsed_seconds": r.get("elapsed_seconds", 0),
        })
    
    # Save comparison
    comparison_path = results_dir / f"comparison_{timestamp}.json"
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    logger.info(f"Saved comparison to {comparison_path}")
    
    # Print comparison table
    print("\n" + "=" * 100)
    print("MULTI-GPU BENCHMARK RESULTS")
    print("=" * 100)
    print(f"{'Model':<20} {'Dataset':<10} {'GPU':<5} {'Tool Acc':<10} {'Exact':<10} {'Param F1':<10} {'Time(s)':<10}")
    print("-" * 100)
    for row in sorted(comparison, key=lambda x: (x['model'], x['dataset'])):
        print(f"{row['model']:<20} {row['dataset']:<10} {row['gpu']:<5} "
              f"{row['tool_accuracy']:.2%}     {row['exact_match']:.2%}     "
              f"{row['param_f1']:.2%}     {row['elapsed_seconds']:.0f}")
    print("=" * 100)
    
    # Summary by model
    print("\nSUMMARY BY MODEL (averaged across datasets):")
    print("-" * 60)
    models = set(r['model'] for r in comparison)
    for model in sorted(models):
        model_results = [r for r in comparison if r['model'] == model]
        avg_tool = sum(r['tool_accuracy'] for r in model_results) / len(model_results)
        avg_exact = sum(r['exact_match'] for r in model_results) / len(model_results)
        print(f"{model:<20} Tool Acc: {avg_tool:.2%}  Exact Match: {avg_exact:.2%}")
    
    # Summary by dataset
    print("\nSUMMARY BY DATASET (averaged across models):")
    print("-" * 60)
    datasets = set(r['dataset'] for r in comparison)
    for dataset in sorted(datasets):
        ds_results = [r for r in comparison if r['dataset'] == dataset]
        avg_tool = sum(r['tool_accuracy'] for r in ds_results) / len(ds_results)
        avg_exact = sum(r['exact_match'] for r in ds_results) / len(ds_results)
        print(f"{dataset:<10} Tool Acc: {avg_tool:.2%}  Exact Match: {avg_exact:.2%}")
    
    print("=" * 100)
    
    # Report failures
    failed = [r for r in results if not r.get("success")]
    if failed:
        print(f"\nFAILED TASKS: {len(failed)}")
        for r in failed:
            print(f"  - {r['model']}/{r['dataset']} on GPU {r['gpu_id']}: {r.get('error', 'Unknown')}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU Audio Tool Calling Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to configuration file (YAML or JSON)"
    )
    
    parser.add_argument(
        "--num-gpus", "-g",
        type=int,
        default=None,
        help="Number of GPUs to use (default: all available)"
    )
    
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated list of GPU IDs to use (e.g., '0,1,2,3')"
    )
    
    parser.add_argument(
        "--strategy",
        type=str,
        default="combo_parallel",
        choices=["combo_parallel", "model_parallel"],
        help="Parallelization strategy (default: combo_parallel)"
    )
    
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="Number of GPUs per model instance for model parallelism (default: 1). "
             "E.g., --num-gpus-per-model 2 spreads each model across 2 GPUs to reduce OOM."
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
    
    # Determine GPUs to use
    available_gpus = get_available_gpus()
    if not available_gpus:
        logger.error("No GPUs available!")
        sys.exit(1)
    
    if args.gpus:
        gpus = [int(g.strip()) for g in args.gpus.split(",")]
        # Validate GPU IDs
        invalid = [g for g in gpus if g not in available_gpus]
        if invalid:
            logger.error(f"Invalid GPU IDs: {invalid}. Available: {available_gpus}")
            sys.exit(1)
    elif args.num_gpus:
        gpus = available_gpus[:args.num_gpus]
    else:
        gpus = available_gpus
    
    logger.info(f"Using GPUs: {gpus}")
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Log model parallelism setting
    if args.num_gpus_per_model > 1:
        logger.info(f"Model parallelism enabled: {args.num_gpus_per_model} GPUs per model instance")
        logger.info(f"Effective concurrent slots: {len(gpus) // args.num_gpus_per_model}")
    
    # Create tasks based on strategy
    logger.info(f"Creating tasks with strategy: {args.strategy}")
    if args.strategy == "combo_parallel":
        tasks = create_tasks_combo_parallel(config, gpus, num_gpus_per_model=args.num_gpus_per_model)
    else:
        tasks = create_tasks_model_parallel(config, gpus, num_gpus_per_model=args.num_gpus_per_model)
    
    logger.info(f"Created {len(tasks)} tasks")
    
    # Print task summary
    print("\n" + "=" * 80)
    print("TASK SUMMARY")
    print("=" * 80)
    for task in tasks:
        gpu_label = ",".join(str(g) for g in task.gpu_ids) if task.gpu_ids else str(task.gpu_id)
        print(f"  Task {task.task_id}: {task.model_name} on {task.dataset_name} -> GPU(s) {gpu_label}")
    print("=" * 80 + "\n")
    
    # Run tasks in parallel
    start_time = time.time()
    results = run_tasks_parallel(tasks, max_concurrent=len(gpus))
    total_time = time.time() - start_time
    
    logger.info(f"All tasks completed in {total_time:.1f}s")
    
    # Save and display results
    save_combined_results(results, config.results_dir)
    
    print(f"\nTotal execution time: {total_time:.1f}s")


if __name__ == "__main__":
    main()

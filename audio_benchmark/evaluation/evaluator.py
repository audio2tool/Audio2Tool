"""
Benchmark Evaluator

This module provides the main evaluation orchestration for running
audio models on benchmark datasets and computing metrics.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
from tqdm import tqdm

from ..models.base import BaseAudioModel, ModelOutput
from ..datasets.base import BaseDataset, QuerySample
from .metrics import (
    ToolCallingMetrics,
    compute_tool_accuracy,
    compute_parameter_accuracy,
    compute_exact_match,
    compute_multi_tool_metrics,
    compute_additional_tool_recall,
    compute_ranking_metrics,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """
    Result of a single evaluation (one audio sample).
    
    Attributes:
        query_idx: Query index
        speaker_idx: Speaker index within query
        audio_path: Path to audio file
        ground_truth_tool: Expected tool name (primary)
        ground_truth_params: Expected parameters
        predicted_tool: Predicted tool name (primary)
        predicted_params: Predicted parameters
        tool_correct: Whether tool prediction is correct
        params_exact_match: Whether parameters match exactly
        param_metrics: Detailed parameter metrics
        latency_ms: Inference latency
        raw_output: Raw model output
        error: Error message if inference failed
        # Multi-tool fields
        all_expected_tools: All expected tool names (primary + additional)
        all_predicted_tools: All predicted tool names
        multi_tool_metrics: Multi-tool evaluation metrics
    """
    query_idx: int
    speaker_idx: int
    audio_path: str
    ground_truth_tool: str
    ground_truth_params: Dict[str, Any]
    predicted_tool: str = ""
    predicted_params: Dict[str, Any] = field(default_factory=dict)
    tool_correct: bool = False
    params_exact_match: bool = False
    param_metrics: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    raw_output: str = ""
    error: Optional[str] = None
    domain: str = ""
    category: str = ""
    # Multi-tool fields
    all_expected_tools: List[str] = field(default_factory=list)
    all_predicted_tools: List[str] = field(default_factory=list)
    multi_tool_metrics: Dict[str, Any] = field(default_factory=dict)
    # Ranking metrics
    ranking_metrics: Dict[str, float] = field(default_factory=dict)
    
    @property
    def is_multi_tool(self) -> bool:
        """Check if this is a multi-tool query."""
        return len(self.all_expected_tools) > 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query_idx": self.query_idx,
            "speaker_idx": self.speaker_idx,
            "audio_path": self.audio_path,
            "ground_truth_tool": self.ground_truth_tool,
            "ground_truth_params": self.ground_truth_params,
            "predicted_tool": self.predicted_tool,
            "predicted_params": self.predicted_params,
            "tool_correct": self.tool_correct,
            "params_exact_match": self.params_exact_match,
            "param_metrics": self.param_metrics,
            "latency_ms": self.latency_ms,
            "raw_output": self.raw_output,
            "error": self.error,
            "domain": self.domain,
            "category": self.category,
            # Multi-tool fields
            "all_expected_tools": self.all_expected_tools,
            "all_predicted_tools": self.all_predicted_tools,
            "multi_tool_metrics": self.multi_tool_metrics,
            # Ranking metrics
            "ranking_metrics": self.ranking_metrics,
        }


class BenchmarkEvaluator:
    """
    Main benchmark evaluator that orchestrates model inference and metric computation.
    
    Attributes:
        model: Audio model to evaluate
        dataset: Dataset to evaluate on
        results_dir: Directory to save results
        results: List of evaluation results
        metrics: Aggregated metrics
    """
    
    def __init__(
        self,
        model: BaseAudioModel,
        dataset: BaseDataset,
        results_dir: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_speakers_per_query: int = 1,
        save_raw_outputs: bool = True,
    ):
        """
        Initialize the evaluator.
        
        Args:
            model: Audio model to evaluate
            dataset: Dataset to evaluate on
            results_dir: Directory to save results (optional)
            system_prompt: Custom system prompt for the model
            max_speakers_per_query: Number of speakers to evaluate per query
            save_raw_outputs: Whether to save raw model outputs
        """
        self.model = model
        self.dataset = dataset
        self.results_dir = Path(results_dir) if results_dir else None
        self.system_prompt = system_prompt
        self.max_speakers_per_query = max_speakers_per_query
        self.save_raw_outputs = save_raw_outputs
        
        self.results: List[EvaluationResult] = []
        self.metrics: Optional[ToolCallingMetrics] = None
        self._run_timestamp: Optional[str] = None
        
    def run(
        self,
        progress_bar: bool = True,
        continue_on_error: bool = True,
        tqdm_position: int = 0,
        tqdm_desc: str = None,
        num_workers: int = 1,
        **generation_kwargs
    ) -> ToolCallingMetrics:
        """
        Run the benchmark evaluation.
        
        Args:
            progress_bar: Whether to show progress bar
            continue_on_error: Whether to continue on inference errors
            num_workers: Number of parallel workers for API-based models (default: 1)
            **generation_kwargs: Additional arguments for model.generate()
            
        Returns:
            Aggregated metrics
        """
        self._run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure model is loaded
        if not self.model.is_loaded:
            logger.info("Loading model...")
            self.model.load_model()
            
        # Ensure dataset is loaded
        if not self.dataset.is_loaded:
            logger.info("Loading dataset...")
            self.dataset.load()
            
        # Get tools schema
        tools = self.dataset.get_tools_schema()
        logger.info(f"Evaluating on {len(self.dataset)} queries with {len(tools)} tools")
        
        # Clear previous results
        self.results = []
        
        if num_workers > 1:
            self._run_parallel(tools, continue_on_error, progress_bar,
                               tqdm_position, tqdm_desc, num_workers,
                               **generation_kwargs)
        else:
            self._run_sequential(tools, continue_on_error, progress_bar,
                                 tqdm_position, tqdm_desc,
                                 **generation_kwargs)
            
        # Compute aggregated metrics
        self.metrics = self._compute_metrics()
        
        # Save results
        if self.results_dir:
            self._save_results()
            
        return self.metrics

    def _run_sequential(self, tools, continue_on_error, progress_bar,
                        tqdm_position, tqdm_desc, **generation_kwargs):
        """Run evaluation sequentially (original behavior)."""
        samples = self.dataset.samples
        
        # Calculate total evaluations (queries × speakers per query)
        total_evaluations = sum(
            min(self.max_speakers_per_query, sample.num_speakers)
            for sample in samples
        )
        
        desc = tqdm_desc or f"Evaluating {self.model.model_name}"
        pbar = tqdm(total=total_evaluations, desc=desc,
                    position=tqdm_position, leave=True) if progress_bar else None

        for sample in samples:
            num_speakers = min(self.max_speakers_per_query, sample.num_speakers)
            self._evaluate_sample(sample, tools, continue_on_error,
                                  **generation_kwargs)
            if pbar:
                pbar.update(num_speakers)
        
        if pbar:
            pbar.close()

    def _run_parallel(self, tools, continue_on_error, progress_bar,
                      tqdm_position, tqdm_desc, num_workers,
                      **generation_kwargs):
        """Run evaluation with parallel workers using ThreadPoolExecutor."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        lock = threading.Lock()
        desc = tqdm_desc or f"Evaluating {self.model.model_name}"
        
        # Calculate total evaluations (queries × speakers per query)
        total_evaluations = sum(
            min(self.max_speakers_per_query, sample.num_speakers)
            for sample in self.dataset.samples
        )
        pbar = tqdm(total=total_evaluations, desc=desc,
                    position=tqdm_position, leave=True) if progress_bar else None

        def eval_one(sample):
            results = []
            num_speakers = min(self.max_speakers_per_query, sample.num_speakers)
            all_expected = sample.all_expected_tools
            
            for speaker_idx in range(num_speakers):
                audio_path = sample.audio_paths[speaker_idx]
                result = EvaluationResult(
                    query_idx=sample.query_idx,
                    speaker_idx=speaker_idx,
                    audio_path=audio_path,
                    ground_truth_tool=sample.tool_name,
                    ground_truth_params=sample.extracted_params,
                    domain=sample.domain,
                    category=sample.category,
                    all_expected_tools=all_expected,
                )
                try:
                    output = self.model.generate(
                        audio_path=audio_path,
                        tools=tools,
                        system_prompt=self.system_prompt,
                        gt_query=sample.query_text,
                        **generation_kwargs
                    )
                    result.predicted_tool = output.tool_name
                    result.predicted_params = output.parameters
                    result.latency_ms = output.latency_ms or 0.0
                    if self.save_raw_outputs:
                        result.raw_output = output.raw_output
                    
                    # Store all predicted tools
                    result.all_predicted_tools = output.all_tool_names
                    
                    result.tool_correct = compute_tool_accuracy(
                        output.tool_name, sample.tool_name
                    )
                    result.param_metrics = compute_parameter_accuracy(
                        output.parameters, sample.extracted_params
                    )
                    result.params_exact_match = result.param_metrics["exact_match"]
                    
                    # Compute multi-tool metrics if applicable
                    if sample.is_multi_tool or len(output.all_tool_names) > 1:
                        result.multi_tool_metrics = compute_multi_tool_metrics(
                            output.all_tool_names,
                            all_expected
                        )
                        if sample.additional_tool_calls:
                            add_recall = compute_additional_tool_recall(
                                output.all_tool_names,
                                sample.additional_tool_calls
                            )
                            result.multi_tool_metrics["additional_tool_recall"] = add_recall
                    
                    # Compute ranking metrics (Recall@k and NDCG)
                    result.ranking_metrics = compute_ranking_metrics(
                        output.all_tool_names,
                        all_expected,
                        k_values=[1, 2, 3, 5]
                    )
                    
                except Exception as e:
                    logger.error(f"Error evaluating query {sample.query_idx}, speaker {speaker_idx}: {e}")
                    result.error = str(e)
                    if not continue_on_error:
                        raise
                results.append(result)
            return results

        logger.info(f"Running parallel evaluation with {num_workers} workers")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(eval_one, sample): sample
                for sample in self.dataset.samples
            }
            for future in as_completed(futures):
                try:
                    results = future.result()
                    with lock:
                        self.results.extend(results)
                        if pbar:
                            pbar.update(len(results))  # Update by number of speakers evaluated
                except Exception as e:
                    logger.error(f"Worker exception: {e}")
                    sample = futures[future]
                    num_speakers = min(self.max_speakers_per_query, sample.num_speakers)
                    if pbar:
                        pbar.update(num_speakers)

        if pbar:
            pbar.close()
    
    def _evaluate_sample(
        self,
        sample: QuerySample,
        tools: List[Dict[str, Any]],
        continue_on_error: bool,
        **generation_kwargs
    ) -> None:
        """Evaluate a single query sample with its audio files."""
        # Determine how many speakers to evaluate
        num_speakers = min(self.max_speakers_per_query, sample.num_speakers)
        
        for speaker_idx in range(num_speakers):
            audio_path = sample.audio_paths[speaker_idx]
            
            # Get all expected tools (primary + additional)
            all_expected = sample.all_expected_tools
            
            result = EvaluationResult(
                query_idx=sample.query_idx,
                speaker_idx=speaker_idx,
                audio_path=audio_path,
                ground_truth_tool=sample.tool_name,
                ground_truth_params=sample.extracted_params,
                domain=sample.domain,
                category=sample.category,
                all_expected_tools=all_expected,
            )
            
            try:
                # Run inference
                output = self.model.generate(
                    audio_path=audio_path,
                    tools=tools,
                    system_prompt=self.system_prompt,
                    gt_query=sample.query_text,
                    **generation_kwargs
                )
                
                # Store predictions (primary tool)
                result.predicted_tool = output.tool_name
                result.predicted_params = output.parameters
                result.latency_ms = output.latency_ms or 0.0
                if self.save_raw_outputs:
                    result.raw_output = output.raw_output
                
                # Store all predicted tools
                result.all_predicted_tools = output.all_tool_names
                    
                # Compute primary tool metrics
                result.tool_correct = compute_tool_accuracy(
                    output.tool_name, sample.tool_name
                )
                result.param_metrics = compute_parameter_accuracy(
                    output.parameters, sample.extracted_params
                )
                result.params_exact_match = result.param_metrics["exact_match"]
                
                # Compute multi-tool metrics if applicable
                if sample.is_multi_tool or len(output.all_tool_names) > 1:
                    result.multi_tool_metrics = compute_multi_tool_metrics(
                        output.all_tool_names,
                        all_expected
                    )
                    # Add additional tool recall
                    if sample.additional_tool_calls:
                        add_recall = compute_additional_tool_recall(
                            output.all_tool_names,
                            sample.additional_tool_calls
                        )
                        result.multi_tool_metrics["additional_tool_recall"] = add_recall
                
                # Compute ranking metrics (Recall@k and NDCG)
                result.ranking_metrics = compute_ranking_metrics(
                    output.all_tool_names,
                    all_expected,
                    k_values=[1, 2, 3, 5]
                )
                
            except Exception as e:
                logger.error(f"Error evaluating query {sample.query_idx}, speaker {speaker_idx}: {e}")
                result.error = str(e)
                if not continue_on_error:
                    raise
                    
            self.results.append(result)
    
    def _compute_metrics(self) -> ToolCallingMetrics:
        """Compute aggregated metrics from results."""
        if not self.results:
            return ToolCallingMetrics()
            
        # Filter out errors
        valid_results = [r for r in self.results if r.error is None]
        
        if not valid_results:
            return ToolCallingMetrics(total_samples=len(self.results))
        
        # Overall metrics
        tool_correct = [r.tool_correct for r in valid_results]
        param_exact = [r.params_exact_match for r in valid_results]
        param_precision = [r.param_metrics.get("precision", 0) for r in valid_results]
        param_recall = [r.param_metrics.get("recall", 0) for r in valid_results]
        param_f1 = [r.param_metrics.get("f1", 0) for r in valid_results]
        exact_match = [r.tool_correct and r.params_exact_match for r in valid_results]
        latencies = [r.latency_ms for r in valid_results if r.latency_ms > 0]
        
        metrics = ToolCallingMetrics(
            total_samples=len(valid_results),
            tool_accuracy=np.mean(tool_correct),
            param_exact_match=np.mean(param_exact),
            param_precision=np.mean(param_precision),
            param_recall=np.mean(param_recall),
            param_f1=np.mean(param_f1),
            exact_match=np.mean(exact_match),
        )
        
        # Multi-tool metrics (for samples with additional_tool_calls)
        multi_tool_results = [r for r in valid_results if r.is_multi_tool]
        if multi_tool_results:
            metrics.multi_tool_samples = len(multi_tool_results)
            
            # All tools match
            all_match = [r.multi_tool_metrics.get("all_tools_match", False) for r in multi_tool_results]
            metrics.multi_tool_all_match = np.mean(all_match)
            
            # Tool recall/precision/f1
            recalls = [r.multi_tool_metrics.get("tool_recall", 0) for r in multi_tool_results]
            precisions = [r.multi_tool_metrics.get("tool_precision", 0) for r in multi_tool_results]
            f1s = [r.multi_tool_metrics.get("tool_f1", 0) for r in multi_tool_results]
            metrics.multi_tool_recall = np.mean(recalls)
            metrics.multi_tool_precision = np.mean(precisions)
            metrics.multi_tool_f1 = np.mean(f1s)
            
            # Additional tool recall (only for samples that have additional tools)
            add_recalls = []
            for r in multi_tool_results:
                add_info = r.multi_tool_metrics.get("additional_tool_recall", {})
                if isinstance(add_info, dict) and add_info.get("count_expected", 0) > 0:
                    add_recalls.append(add_info.get("recall", 0))
            if add_recalls:
                metrics.additional_tool_recall = np.mean(add_recalls)
            
            # Multi-tool Recall@k (only for multi-tool samples)
            multi_with_ranking = [r for r in multi_tool_results if r.ranking_metrics]
            if multi_with_ranking:
                metrics.multi_tool_recall_at_1 = np.mean([r.ranking_metrics.get("recall@1", 0) for r in multi_with_ranking])
                metrics.multi_tool_recall_at_2 = np.mean([r.ranking_metrics.get("recall@2", 0) for r in multi_with_ranking])
                metrics.multi_tool_recall_at_3 = np.mean([r.ranking_metrics.get("recall@3", 0) for r in multi_with_ranking])
                metrics.multi_tool_recall_at_5 = np.mean([r.ranking_metrics.get("recall@5", 0) for r in multi_with_ranking])
                metrics.multi_tool_ndcg = np.mean([r.ranking_metrics.get("ndcg", 0) for r in multi_with_ranking])
        
        # Ranking metrics (Recall@k and NDCG)
        results_with_ranking = [r for r in valid_results if r.ranking_metrics]
        if results_with_ranking:
            metrics.recall_at_1 = np.mean([r.ranking_metrics.get("recall@1", 0) for r in results_with_ranking])
            metrics.recall_at_2 = np.mean([r.ranking_metrics.get("recall@2", 0) for r in results_with_ranking])
            metrics.recall_at_3 = np.mean([r.ranking_metrics.get("recall@3", 0) for r in results_with_ranking])
            metrics.recall_at_5 = np.mean([r.ranking_metrics.get("recall@5", 0) for r in results_with_ranking])
            metrics.ndcg = np.mean([r.ranking_metrics.get("ndcg", 0) for r in results_with_ranking])
            metrics.ndcg_at_1 = np.mean([r.ranking_metrics.get("ndcg@1", 0) for r in results_with_ranking])
            metrics.ndcg_at_2 = np.mean([r.ranking_metrics.get("ndcg@2", 0) for r in results_with_ranking])
            metrics.ndcg_at_3 = np.mean([r.ranking_metrics.get("ndcg@3", 0) for r in results_with_ranking])
            metrics.ndcg_at_5 = np.mean([r.ranking_metrics.get("ndcg@5", 0) for r in results_with_ranking])
        
        # Latency stats
        if latencies:
            metrics.latency_mean_ms = np.mean(latencies)
            metrics.latency_std_ms = np.std(latencies)
            metrics.latency_p50_ms = np.percentile(latencies, 50)
            metrics.latency_p95_ms = np.percentile(latencies, 95)
            metrics.latency_p99_ms = np.percentile(latencies, 99)
        
        # Domain breakdown
        domains = set(r.domain for r in valid_results if r.domain)
        for domain in domains:
            domain_results = [r for r in valid_results if r.domain == domain]
            domain_multi = [r for r in domain_results if r.is_multi_tool]
            metrics.domain_metrics[domain] = {
                "samples": len(domain_results),
                "tool_accuracy": np.mean([r.tool_correct for r in domain_results]),
                "exact_match": np.mean([r.tool_correct and r.params_exact_match for r in domain_results]),
                "param_f1": np.mean([r.param_metrics.get("f1", 0) for r in domain_results]),
                "multi_tool_samples": len(domain_multi),
                "multi_tool_recall": np.mean([r.multi_tool_metrics.get("tool_recall", 0) for r in domain_multi]) if domain_multi else 0.0,
            }
        
        # Category breakdown
        categories = set(r.category for r in valid_results if r.category)
        for category in categories:
            cat_results = [r for r in valid_results if r.category == category]
            cat_multi = [r for r in cat_results if r.is_multi_tool]
            metrics.category_metrics[category] = {
                "samples": len(cat_results),
                "tool_accuracy": np.mean([r.tool_correct for r in cat_results]),
                "exact_match": np.mean([r.tool_correct and r.params_exact_match for r in cat_results]),
                "multi_tool_samples": len(cat_multi),
                "multi_tool_recall": np.mean([r.multi_tool_metrics.get("tool_recall", 0) for r in cat_multi]) if cat_multi else 0.0,
            }
        
        # Tool breakdown (top 20 most common)
        tool_counts = {}
        for r in valid_results:
            tool = r.ground_truth_tool
            if tool not in tool_counts:
                tool_counts[tool] = []
            tool_counts[tool].append(r)
        
        sorted_tools = sorted(tool_counts.items(), key=lambda x: -len(x[1]))[:20]
        for tool_name, tool_results in sorted_tools:
            metrics.tool_metrics[tool_name] = {
                "samples": len(tool_results),
                "tool_accuracy": np.mean([r.tool_correct for r in tool_results]),
                "exact_match": np.mean([r.tool_correct and r.params_exact_match for r in tool_results]),
            }
        
        # Error analysis (most common mistakes)
        errors = []
        for r in valid_results:
            if not r.tool_correct:
                errors.append({
                    "type": "wrong_tool",
                    "expected": r.ground_truth_tool,
                    "predicted": r.predicted_tool,
                })
            elif not r.params_exact_match:
                errors.append({
                    "type": "wrong_params",
                    "tool": r.ground_truth_tool,
                    "expected": r.ground_truth_params,
                    "predicted": r.predicted_params,
                    "missing": r.param_metrics.get("missing_params", []),
                    "wrong": r.param_metrics.get("wrong_value_params", []),
                })
            # Add multi-tool errors
            if r.is_multi_tool and not r.multi_tool_metrics.get("all_tools_match", False):
                missing = r.multi_tool_metrics.get("missing_tools", [])
                if missing:
                    errors.append({
                        "type": "missing_additional_tools",
                        "expected": r.all_expected_tools,
                        "predicted": r.all_predicted_tools,
                        "missing": missing,
                    })
        
        # Count error types
        error_counts = {}
        for e in errors:
            if e["type"] == "wrong_tool":
                key = f"wrong_tool: {e['expected']} -> {e['predicted']}"
            elif e["type"] == "missing_additional_tools":
                key = f"missing_tools: {e['missing']}"
            else:
                key = f"wrong_params: {e['tool']}"
            error_counts[key] = error_counts.get(key, 0) + 1
        
        metrics.common_errors = [
            {"error": k, "count": v} 
            for k, v in sorted(error_counts.items(), key=lambda x: -x[1])[:10]
        ]
        
        return metrics
    
    def _save_results(self) -> None:
        """Save results to disk."""
        if self.results_dir is None:
            return
            
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create run directory
        run_dir = self.results_dir / f"{self.model.model_name}_{self._run_timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics_path = run_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics.to_dict(), f, indent=2, default=str)
        logger.info(f"Saved metrics to {metrics_path}")
        
        # Save detailed results
        results_path = run_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2, default=str)
        logger.info(f"Saved detailed results to {results_path}")
        
        # Save summary
        summary_path = run_dir / "summary.txt"
        with open(summary_path, 'w') as f:
            f.write(str(self.metrics))
        logger.info(f"Saved summary to {summary_path}")
        
    def get_results_df(self):
        """Get results as a pandas DataFrame."""
        try:
            import pandas as pd
            return pd.DataFrame([r.to_dict() for r in self.results])
        except ImportError:
            logger.warning("pandas not available. Install with: pip install pandas")
            return None

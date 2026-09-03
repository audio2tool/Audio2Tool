"""
Evaluation Metrics for Tool Calling Benchmark

This module provides various metrics for evaluating tool calling
accuracy and parameter filling accuracy.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Set
import re
import logging

logger = logging.getLogger(__name__)


def normalize_tool_name(name: str) -> str:
    """
    Normalize tool name for comparison.
    
    - Converts to lowercase
    - Removes whitespace
    - Removes parentheses and arguments
    
    Args:
        name: Raw tool name
        
    Returns:
        Normalized tool name
    """
    if not name:
        return ""
    # Remove function call syntax
    name = re.sub(r'\(.*\)', '', name)
    # Remove whitespace and convert to lowercase
    name = name.strip().lower()
    return name


def normalize_parameters(params: Dict[str, Any]) -> Dict[str, str]:
    """
    Normalize parameters for comparison.
    
    - Converts all values to lowercase strings
    - Strips whitespace
    - Handles common variations (true/True/TRUE)
    
    Args:
        params: Parameter dictionary
        
    Returns:
        Normalized parameter dictionary
    """
    if not params:
        return {}
        
    normalized = {}
    for key, value in params.items():
        # Normalize key
        norm_key = key.strip().lower()
        
        # Normalize value
        if value is None:
            norm_value = ""
        elif isinstance(value, bool):
            norm_value = "true" if value else "false"
        elif isinstance(value, (int, float)):
            norm_value = str(value)
        else:
            norm_value = str(value).strip().lower()
            
        normalized[norm_key] = norm_value
        
    return normalized


def compute_tool_accuracy(
    predicted: str, 
    ground_truth: str,
    case_sensitive: bool = False
) -> bool:
    """
    Compute whether the predicted tool name matches ground truth.
    
    Args:
        predicted: Predicted tool name
        ground_truth: Ground truth tool name
        case_sensitive: Whether comparison should be case-sensitive
        
    Returns:
        True if tool names match
    """
    if case_sensitive:
        pred_norm = predicted.strip()
        gt_norm = ground_truth.strip()
    else:
        pred_norm = normalize_tool_name(predicted)
        gt_norm = normalize_tool_name(ground_truth)
        
    return pred_norm == gt_norm


def compute_parameter_accuracy(
    predicted: Dict[str, Any],
    ground_truth: Dict[str, Any],
    case_sensitive: bool = False
) -> Dict[str, Any]:
    """
    Compute parameter accuracy metrics.
    
    Args:
        predicted: Predicted parameters
        ground_truth: Ground truth parameters
        case_sensitive: Whether comparison should be case-sensitive
        
    Returns:
        Dictionary with:
        - exact_match: Whether all parameters match exactly
        - precision: Fraction of predicted params that are correct
        - recall: Fraction of ground truth params that were predicted
        - f1: F1 score combining precision and recall
        - correct_params: List of correctly predicted parameters
        - missing_params: List of missing parameters
        - extra_params: List of extra predicted parameters
        - wrong_value_params: Parameters with wrong values
    """
    if case_sensitive:
        pred_norm = {k: str(v) for k, v in predicted.items()}
        gt_norm = {k: str(v) for k, v in ground_truth.items()}
    else:
        pred_norm = normalize_parameters(predicted)
        gt_norm = normalize_parameters(ground_truth)
    
    # Handle empty cases
    if not gt_norm and not pred_norm:
        return {
            "exact_match": True,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "correct_params": [],
            "missing_params": [],
            "extra_params": [],
            "wrong_value_params": [],
        }
    
    if not gt_norm and pred_norm:
        return {
            "exact_match": False,
            "precision": 0.0,
            "recall": 1.0,  # Nothing to recall
            "f1": 0.0,
            "correct_params": [],
            "missing_params": [],
            "extra_params": list(pred_norm.keys()),
            "wrong_value_params": [],
        }
    
    if gt_norm and not pred_norm:
        return {
            "exact_match": False,
            "precision": 0.0,  # Nothing predicted
            "recall": 0.0,
            "f1": 0.0,
            "correct_params": [],
            "missing_params": list(gt_norm.keys()),
            "extra_params": [],
            "wrong_value_params": [],
        }
    
    # Compute detailed metrics
    correct_params = []
    wrong_value_params = []
    missing_params = []
    extra_params = []
    
    pred_keys = set(pred_norm.keys())
    gt_keys = set(gt_norm.keys())
    
    # Parameters in ground truth
    for key in gt_keys:
        if key in pred_keys:
            if pred_norm[key] == gt_norm[key]:
                correct_params.append(key)
            else:
                wrong_value_params.append(key)
        else:
            missing_params.append(key)
    
    # Extra parameters in prediction
    extra_params = list(pred_keys - gt_keys)
    
    # Compute metrics
    num_correct = len(correct_params)
    num_predicted = len(pred_keys)
    num_ground_truth = len(gt_keys)
    
    precision = num_correct / num_predicted if num_predicted > 0 else 0.0
    recall = num_correct / num_ground_truth if num_ground_truth > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    exact_match = (
        len(missing_params) == 0 and 
        len(extra_params) == 0 and 
        len(wrong_value_params) == 0
    )
    
    return {
        "exact_match": exact_match,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "correct_params": correct_params,
        "missing_params": missing_params,
        "extra_params": extra_params,
        "wrong_value_params": wrong_value_params,
    }


def compute_exact_match(
    predicted_tool: str,
    predicted_params: Dict[str, Any],
    ground_truth_tool: str,
    ground_truth_params: Dict[str, Any],
    case_sensitive: bool = False
) -> bool:
    """
    Compute exact match (both tool name and all parameters must match).
    
    Args:
        predicted_tool: Predicted tool name
        predicted_params: Predicted parameters
        ground_truth_tool: Ground truth tool name
        ground_truth_params: Ground truth parameters
        case_sensitive: Whether comparison should be case-sensitive
        
    Returns:
        True if both tool and parameters match exactly
    """
    tool_match = compute_tool_accuracy(
        predicted_tool, ground_truth_tool, case_sensitive
    )
    
    if not tool_match:
        return False
        
    param_result = compute_parameter_accuracy(
        predicted_params, ground_truth_params, case_sensitive
    )
    
    return param_result["exact_match"]


# ============================================================================
# Multi-Tool Metrics (for tier2, tier3 additional_tool_calls)
# ============================================================================

def compute_multi_tool_metrics(
    predicted_tools: List[str],
    expected_tools: List[str],
    case_sensitive: bool = False
) -> Dict[str, Any]:
    """
    Compute metrics for multi-tool queries.
    
    Args:
        predicted_tools: List of predicted tool names
        expected_tools: List of expected tool names (primary + additional)
        case_sensitive: Whether comparison should be case-sensitive
        
    Returns:
        Dictionary with multi-tool metrics:
        - primary_tool_match: Whether the primary (first) tool matches
        - all_tools_match: Whether all expected tools were predicted (exact set match)
        - tool_recall: Fraction of expected tools that were predicted
        - tool_precision: Fraction of predicted tools that were expected
        - tool_f1: F1 score for tool set
        - num_expected: Number of expected tools
        - num_predicted: Number of predicted tools
        - missing_tools: Tools in expected but not predicted
        - extra_tools: Tools predicted but not expected
    """
    # Normalize tool names for comparison
    if case_sensitive:
        pred_norm = [t.strip() for t in predicted_tools if t]
        exp_norm = [t.strip() for t in expected_tools if t]
    else:
        pred_norm = [normalize_tool_name(t) for t in predicted_tools if t]
        exp_norm = [normalize_tool_name(t) for t in expected_tools if t]
    
    # Handle empty cases
    if not exp_norm and not pred_norm:
        return {
            "primary_tool_match": True,
            "all_tools_match": True,
            "tool_recall": 1.0,
            "tool_precision": 1.0,
            "tool_f1": 1.0,
            "num_expected": 0,
            "num_predicted": 0,
            "missing_tools": [],
            "extra_tools": [],
        }
    
    if not exp_norm:
        return {
            "primary_tool_match": False,
            "all_tools_match": False,
            "tool_recall": 1.0,
            "tool_precision": 0.0,
            "tool_f1": 0.0,
            "num_expected": 0,
            "num_predicted": len(pred_norm),
            "missing_tools": [],
            "extra_tools": pred_norm,
        }
    
    if not pred_norm:
        return {
            "primary_tool_match": False,
            "all_tools_match": False,
            "tool_recall": 0.0,
            "tool_precision": 0.0,
            "tool_f1": 0.0,
            "num_expected": len(exp_norm),
            "num_predicted": 0,
            "missing_tools": exp_norm,
            "extra_tools": [],
        }
    
    # Check primary tool match
    primary_match = pred_norm[0] == exp_norm[0] if pred_norm and exp_norm else False
    
    # Convert to sets for recall/precision computation
    pred_set = set(pred_norm)
    exp_set = set(exp_norm)
    
    # Compute overlap
    correct_tools = pred_set & exp_set
    missing_tools = exp_set - pred_set
    extra_tools = pred_set - exp_set
    
    # Compute metrics
    num_correct = len(correct_tools)
    num_predicted = len(pred_set)
    num_expected = len(exp_set)
    
    recall = num_correct / num_expected if num_expected > 0 else 0.0
    precision = num_correct / num_predicted if num_predicted > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    all_match = (len(missing_tools) == 0 and len(extra_tools) == 0)
    
    return {
        "primary_tool_match": primary_match,
        "all_tools_match": all_match,
        "tool_recall": recall,
        "tool_precision": precision,
        "tool_f1": f1,
        "num_expected": num_expected,
        "num_predicted": num_predicted,
        "missing_tools": list(missing_tools),
        "extra_tools": list(extra_tools),
    }


def compute_additional_tool_recall(
    predicted_tools: List[str],
    additional_tools: List[str],
    case_sensitive: bool = False
) -> Dict[str, Any]:
    """
    Compute recall specifically for additional tools (excluding primary).
    
    Args:
        predicted_tools: All predicted tool names
        additional_tools: List of additional expected tool names (excluding primary)
        case_sensitive: Whether comparison should be case-sensitive
        
    Returns:
        Dictionary with:
        - recall: Fraction of additional tools predicted
        - count_predicted: Number of additional tools predicted
        - count_expected: Number of additional tools expected
        - found: List of additional tools that were predicted
        - missing: List of additional tools not predicted
    """
    if not additional_tools:
        return {
            "recall": 1.0,  # No additional tools to find
            "count_predicted": 0,
            "count_expected": 0,
            "found": [],
            "missing": [],
        }
    
    # Normalize
    if case_sensitive:
        pred_norm = set(t.strip() for t in predicted_tools if t)
        add_norm = [t.strip() for t in additional_tools if t]
    else:
        pred_norm = set(normalize_tool_name(t) for t in predicted_tools if t)
        add_norm = [normalize_tool_name(t) for t in additional_tools if t]
    
    found = [t for t in add_norm if t in pred_norm]
    missing = [t for t in add_norm if t not in pred_norm]
    
    recall = len(found) / len(add_norm) if add_norm else 1.0
    
    return {
        "recall": recall,
        "count_predicted": len(found),
        "count_expected": len(add_norm),
        "found": found,
        "missing": missing,
    }


# ============================================================================
# Ranking Metrics: Recall@k and NDCG
# ============================================================================

def compute_recall_at_k(
    predicted_tools: List[str],
    expected_tools: List[str],
    k_values: List[int] = [1, 2, 3, 5],
    case_sensitive: bool = False
) -> Dict[str, float]:
    """
    Compute Recall@k for multiple k values.
    
    Recall@k = (# of expected tools in top-k predictions) / (# of expected tools)
    
    Args:
        predicted_tools: Ranked list of predicted tool names (order matters)
        expected_tools: List of expected tool names
        k_values: List of k values to compute recall for
        case_sensitive: Whether comparison should be case-sensitive
        
    Returns:
        Dictionary with recall@k for each k value, e.g.:
        {"recall@1": 0.5, "recall@2": 0.75, "recall@3": 1.0, "recall@5": 1.0}
    """
    # Normalize tool names
    if case_sensitive:
        pred_norm = [t.strip() for t in predicted_tools if t]
        exp_norm = set(t.strip() for t in expected_tools if t)
    else:
        pred_norm = [normalize_tool_name(t) for t in predicted_tools if t]
        exp_norm = set(normalize_tool_name(t) for t in expected_tools if t)
    
    # Handle empty expected case
    if not exp_norm:
        return {f"recall@{k}": 1.0 for k in k_values}
    
    # Handle empty predictions
    if not pred_norm:
        return {f"recall@{k}": 0.0 for k in k_values}
    
    num_expected = len(exp_norm)
    results = {}
    
    for k in k_values:
        top_k_predictions = set(pred_norm[:k])
        hits = len(top_k_predictions & exp_norm)
        results[f"recall@{k}"] = hits / num_expected
    
    return results


def compute_dcg(relevances: List[float], k: Optional[int] = None) -> float:
    """
    Compute Discounted Cumulative Gain.
    
    DCG = sum(rel_i / log2(i + 2)) for i in range(k)
    
    Args:
        relevances: List of relevance scores (1 for relevant, 0 for not)
        k: Number of positions to consider (None = all)
        
    Returns:
        DCG score
    """
    import math
    
    if k is not None:
        relevances = relevances[:k]
    
    dcg = 0.0
    for i, rel in enumerate(relevances):
        # Using log2(i + 2) so position 0 has discount log2(2) = 1
        dcg += rel / math.log2(i + 2)
    
    return dcg


def compute_ndcg(
    predicted_tools: List[str],
    expected_tools: List[str],
    k: Optional[int] = None,
    case_sensitive: bool = False
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG).
    
    NDCG = DCG / IDCG, where IDCG is the ideal DCG (all relevant items first)
    
    Args:
        predicted_tools: Ranked list of predicted tool names (order matters)
        expected_tools: List of expected tool names
        k: Number of positions to consider (None = use max of predictions and expected)
        case_sensitive: Whether comparison should be case-sensitive
        
    Returns:
        NDCG score between 0 and 1
    """
    # Normalize tool names
    if case_sensitive:
        pred_norm = [t.strip() for t in predicted_tools if t]
        exp_norm = set(t.strip() for t in expected_tools if t)
    else:
        pred_norm = [normalize_tool_name(t) for t in predicted_tools if t]
        exp_norm = set(normalize_tool_name(t) for t in expected_tools if t)
    
    # Handle empty cases
    if not exp_norm:
        return 1.0  # Nothing to find, perfect score
    
    if not pred_norm:
        return 0.0  # Nothing predicted
    
    # Compute relevance scores for predictions
    relevances = [1.0 if tool in exp_norm else 0.0 for tool in pred_norm]
    
    # Compute DCG
    dcg = compute_dcg(relevances, k)
    
    # Compute ideal DCG (all relevant items at the top)
    num_relevant = len(exp_norm)
    ideal_relevances = [1.0] * num_relevant + [0.0] * (len(pred_norm) - num_relevant)
    idcg = compute_dcg(ideal_relevances, k)
    
    # Handle edge case where IDCG is 0
    if idcg == 0:
        return 1.0 if dcg == 0 else 0.0
    
    return dcg / idcg


def compute_ranking_metrics(
    predicted_tools: List[str],
    expected_tools: List[str],
    k_values: List[int] = [1, 2, 3, 5],
    case_sensitive: bool = False
) -> Dict[str, Any]:
    """
    Compute all ranking metrics: Recall@k and NDCG.
    
    Args:
        predicted_tools: Ranked list of predicted tool names (order matters)
        expected_tools: List of expected tool names
        k_values: List of k values for Recall@k
        case_sensitive: Whether comparison should be case-sensitive
        
    Returns:
        Dictionary with all ranking metrics:
        - recall@1, recall@2, recall@3, recall@5
        - ndcg (overall)
        - ndcg@1, ndcg@2, ndcg@3, ndcg@5 (truncated)
    """
    results = {}
    
    # Compute Recall@k
    recall_metrics = compute_recall_at_k(
        predicted_tools, expected_tools, k_values, case_sensitive
    )
    results.update(recall_metrics)
    
    # Compute overall NDCG
    results["ndcg"] = compute_ndcg(
        predicted_tools, expected_tools, k=None, case_sensitive=case_sensitive
    )
    
    # Compute NDCG@k for each k
    for k in k_values:
        results[f"ndcg@{k}"] = compute_ndcg(
            predicted_tools, expected_tools, k=k, case_sensitive=case_sensitive
        )
    
    return results


@dataclass
class ToolCallingMetrics:
    """
    Aggregated metrics for tool calling benchmark.
    
    Attributes:
        total_samples: Total number of samples evaluated
        tool_accuracy: Fraction of samples with correct tool name
        param_exact_match: Fraction of samples with exact parameter match
        param_precision: Average parameter precision
        param_recall: Average parameter recall
        param_f1: Average parameter F1 score
        exact_match: Fraction of samples with both correct tool and params
        domain_metrics: Metrics broken down by domain
        tool_metrics: Metrics broken down by tool
        latency_stats: Latency statistics
    """
    total_samples: int = 0
    tool_accuracy: float = 0.0
    param_exact_match: float = 0.0
    param_precision: float = 0.0
    param_recall: float = 0.0
    param_f1: float = 0.0
    exact_match: float = 0.0
    
    # Multi-tool metrics (for tier2, tier3 with additional_tool_calls)
    multi_tool_samples: int = 0  # Samples with additional_tool_calls
    multi_tool_all_match: float = 0.0  # All tools matched exactly
    multi_tool_recall: float = 0.0  # Average recall across multi-tool samples
    multi_tool_precision: float = 0.0  # Average precision across multi-tool samples
    multi_tool_f1: float = 0.0  # Average F1 across multi-tool samples
    additional_tool_recall: float = 0.0  # Recall specifically for additional tools
    # Multi-tool Recall@k (only for multi-tool samples)
    multi_tool_recall_at_1: float = 0.0
    multi_tool_recall_at_2: float = 0.0
    multi_tool_recall_at_3: float = 0.0
    multi_tool_recall_at_5: float = 0.0
    multi_tool_ndcg: float = 0.0
    
    # Ranking metrics: Recall@k and NDCG (all samples)
    recall_at_1: float = 0.0
    recall_at_2: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    ndcg: float = 0.0
    ndcg_at_1: float = 0.0
    ndcg_at_2: float = 0.0
    ndcg_at_3: float = 0.0
    ndcg_at_5: float = 0.0
    
    # Detailed breakdowns
    domain_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    tool_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    category_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Latency statistics
    latency_mean_ms: float = 0.0
    latency_std_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    
    # Error analysis
    common_errors: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_samples": self.total_samples,
            "tool_accuracy": self.tool_accuracy,
            "param_exact_match": self.param_exact_match,
            "param_precision": self.param_precision,
            "param_recall": self.param_recall,
            "param_f1": self.param_f1,
            "exact_match": self.exact_match,
            # Multi-tool metrics
            "multi_tool_samples": self.multi_tool_samples,
            "multi_tool_all_match": self.multi_tool_all_match,
            "multi_tool_recall": self.multi_tool_recall,
            "multi_tool_precision": self.multi_tool_precision,
            "multi_tool_f1": self.multi_tool_f1,
            "additional_tool_recall": self.additional_tool_recall,
            "multi_tool_recall@1": self.multi_tool_recall_at_1,
            "multi_tool_recall@2": self.multi_tool_recall_at_2,
            "multi_tool_recall@3": self.multi_tool_recall_at_3,
            "multi_tool_recall@5": self.multi_tool_recall_at_5,
            "multi_tool_ndcg": self.multi_tool_ndcg,
            # Ranking metrics
            "recall@1": self.recall_at_1,
            "recall@2": self.recall_at_2,
            "recall@3": self.recall_at_3,
            "recall@5": self.recall_at_5,
            "ndcg": self.ndcg,
            "ndcg@1": self.ndcg_at_1,
            "ndcg@2": self.ndcg_at_2,
            "ndcg@3": self.ndcg_at_3,
            "ndcg@5": self.ndcg_at_5,
            # Breakdowns
            "domain_metrics": self.domain_metrics,
            "tool_metrics": self.tool_metrics,
            "category_metrics": self.category_metrics,
            "latency_mean_ms": self.latency_mean_ms,
            "latency_std_ms": self.latency_std_ms,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "common_errors": self.common_errors,
        }
    
    def __str__(self) -> str:
        """Pretty print metrics."""
        lines = [
            "=" * 50,
            "TOOL CALLING BENCHMARK RESULTS",
            "=" * 50,
            f"Total Samples: {self.total_samples}",
            "",
            "Overall Metrics:",
            f"  Tool Accuracy:      {self.tool_accuracy:.2%}",
            f"  Exact Match:        {self.exact_match:.2%}",
            "",
            "Parameter Metrics:",
            f"  Param Exact Match:  {self.param_exact_match:.2%}",
            f"  Param Precision:    {self.param_precision:.2%}",
            f"  Param Recall:       {self.param_recall:.2%}",
            f"  Param F1:           {self.param_f1:.2%}",
        ]
        
        # Add multi-tool metrics if there are multi-tool samples
        if self.multi_tool_samples > 0:
            lines.extend([
                "",
                f"Multi-Tool Metrics ({self.multi_tool_samples} samples):",
                f"  All Tools Match:    {self.multi_tool_all_match:.2%}",
                f"  Tool Recall:        {self.multi_tool_recall:.2%}",
                f"  Tool Precision:     {self.multi_tool_precision:.2%}",
                f"  Tool F1:            {self.multi_tool_f1:.2%}",
                f"  Additional Recall:  {self.additional_tool_recall:.2%}",
                "",
                "Multi-Tool Recall@k:",
                f"  Recall@1:           {self.multi_tool_recall_at_1:.2%}",
                f"  Recall@2:           {self.multi_tool_recall_at_2:.2%}",
                f"  Recall@3:           {self.multi_tool_recall_at_3:.2%}",
                f"  Recall@5:           {self.multi_tool_recall_at_5:.2%}",
                f"  NDCG:               {self.multi_tool_ndcg:.4f}",
            ])
        
        # Add ranking metrics
        lines.extend([
            "",
            "Ranking Metrics (Recall@k):",
            f"  Recall@1:           {self.recall_at_1:.2%}",
            f"  Recall@2:           {self.recall_at_2:.2%}",
            f"  Recall@3:           {self.recall_at_3:.2%}",
            f"  Recall@5:           {self.recall_at_5:.2%}",
            "",
            "Ranking Metrics (NDCG):",
            f"  NDCG:               {self.ndcg:.4f}",
            f"  NDCG@1:             {self.ndcg_at_1:.4f}",
            f"  NDCG@2:             {self.ndcg_at_2:.4f}",
            f"  NDCG@3:             {self.ndcg_at_3:.4f}",
            f"  NDCG@5:             {self.ndcg_at_5:.4f}",
        ])
        
        lines.extend([
            "",
            "Latency:",
            f"  Mean:  {self.latency_mean_ms:.1f} ms",
            f"  P50:   {self.latency_p50_ms:.1f} ms",
            f"  P95:   {self.latency_p95_ms:.1f} ms",
            f"  P99:   {self.latency_p99_ms:.1f} ms",
            "=" * 50,
        ])
        
        if self.domain_metrics:
            lines.append("\nMetrics by Domain:")
            for domain, metrics in self.domain_metrics.items():
                lines.append(f"  {domain}:")
                lines.append(f"    Tool Accuracy: {metrics.get('tool_accuracy', 0):.2%}")
                lines.append(f"    Exact Match:   {metrics.get('exact_match', 0):.2%}")
                
        return "\n".join(lines)

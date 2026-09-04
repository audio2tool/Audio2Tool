"""
Release Dataset Loader

Loads any tier of the public Audio2Tool dataset release
(https://huggingface.co/datasets/RVtech/Audio2Tool).

Each tier directory has a uniform layout:
    <tier_dir>/
    ├── metadata.jsonl   # one row per (query, speaker) audio recording
    └── audio/           # wav files referenced by metadata.jsonl

Usage:
    dataset = get_dataset(
        "release",
        data_dir="/path/to/Audio2Tool/public/tier1_direct",
    )
    dataset.load()
"""

import ast
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

from .base import BaseDataset, QuerySample, register_dataset

logger = logging.getLogger(__name__)


@register_dataset("release")
class ReleaseDataset(BaseDataset):
    """
    Dataset loader for the public Audio2Tool release format.

    Works with every tier of the release (tier1_direct, tier2_parametric,
    tier3_multi_intent, tier4_implicit, tier5_needle, tier6_correction,
    tier7_multiturn, tier8_intent_blending): point ``data_dir`` at the
    tier directory containing ``metadata.jsonl``.
    """

    def __init__(
        self,
        data_dir: str,
        filter_domain: Optional[str] = None,
        filter_category: Optional[str] = None,
        filter_tool: Optional[str] = None,
        max_samples: Optional[int] = None,
        speaker_idx: Optional[int] = None,
        speakers_per_query: Optional[int] = None,
        only_successful: bool = True,
        **kwargs,
    ):
        """
        Initialize the release dataset loader.

        Args:
            data_dir: Path to a tier directory containing metadata.jsonl
            filter_domain: Only include a specific domain (smart_car, smart_home, wearables)
            filter_category: Only include a specific category
            filter_tool: Only include a specific tool
            max_samples: Maximum number of query samples to load
            speaker_idx: Use only this speaker index per query
            speakers_per_query: Number of speakers to use per query (default: all)
            only_successful: Only include successfully synthesized audio
        """
        super().__init__(data_dir, filter_domain, filter_category, max_samples, speaker_idx, **kwargs)
        self.filter_tool = filter_tool
        self.speakers_per_query = speakers_per_query
        self.only_successful = only_successful
        self._tools_schema = None

    def load(self) -> None:
        """Load the dataset from metadata.jsonl."""
        if self._loaded:
            logger.info("Dataset already loaded")
            return

        metadata_path = self.data_dir / "metadata.jsonl"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"metadata.jsonl not found in {self.data_dir}. "
                f"Point data_dir at a tier directory of the Audio2Tool release, "
                f"e.g. /path/to/Audio2Tool/public/tier1_direct"
            )

        logger.info(f"Loading release dataset from {self.data_dir}")

        # Group rows by query_idx; each row is one (query, speaker) recording.
        queries: Dict[int, Dict[str, Any]] = {}
        audio_by_query: Dict[int, List[str]] = {}

        with open(metadata_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)

                query_idx = row["query_idx"]
                if query_idx not in queries:
                    queries[query_idx] = row
                    audio_by_query[query_idx] = []

                if self.speaker_idx is not None and row.get("speaker_idx") != self.speaker_idx:
                    continue
                # Skip only explicit failures; null means not applicable
                # (e.g. tier8 mixed audio has no per-speaker synthesis flag).
                if self.only_successful and row.get("speech_synthesis_success") is False:
                    continue

                audio_rel = row.get("audio", "")
                if not audio_rel:
                    continue
                audio_path = self.data_dir / audio_rel
                if not audio_path.exists():
                    logger.warning(f"Audio file missing: {audio_path}")
                    continue

                if (
                    self.speakers_per_query
                    and len(audio_by_query[query_idx]) >= self.speakers_per_query
                ):
                    continue
                audio_by_query[query_idx].append(str(audio_path))

        loaded_count = 0
        for query_idx in sorted(queries):
            if self.max_samples and loaded_count >= self.max_samples:
                break
            sample = self._build_sample(queries[query_idx], audio_by_query[query_idx])
            if sample is not None:
                self.samples.append(sample)
                loaded_count += 1

        self._loaded = True
        logger.info(f"Loaded {len(self.samples)} samples")

    def _build_sample(self, row: Dict[str, Any], audio_paths: List[str]) -> Optional[QuerySample]:
        """Build a QuerySample from a metadata row and its audio paths."""
        if self.filter_domain and row.get("domain") != self.filter_domain:
            return None
        if self.filter_category and row.get("category") != self.filter_category:
            return None
        if self.filter_tool and row.get("tool_name") != self.filter_tool:
            return None
        if not audio_paths:
            logger.warning(f"No audio files found for query {row.get('query_idx')}")
            return None

        extracted_params = row.get("extracted_params") or {}
        if isinstance(extracted_params, str):
            try:
                extracted_params = ast.literal_eval(extracted_params)
            except (ValueError, SyntaxError):
                extracted_params = {}

        additional_tools = row.get("additional_tool_calls") or []
        if isinstance(additional_tools, str):
            try:
                additional_tools = ast.literal_eval(additional_tools)
            except (ValueError, SyntaxError):
                additional_tools = []

        return QuerySample(
            query_idx=row.get("query_idx", 0),
            query_text=row.get("query", ""),
            tool_name=row.get("tool_name", ""),
            tool_call=row.get("tool_call", ""),
            extracted_params=extracted_params,
            audio_paths=audio_paths,
            domain=row.get("domain", ""),
            category=row.get("category", ""),
            tier=row.get("tier", ""),
            additional_tool_calls=additional_tools,
            metadata={
                "tool_id": row.get("tool_id"),
                "source_endpoint": row.get("source_endpoint"),
                "is_tier7": row.get("is_tier7", False),
                "reasoning": row.get("reasoning"),
                "original_tool_call": row.get("original_tool_call"),
                "correction_type": row.get("correction_type"),
            },
        )

    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """
        Get a simplified schema of the tools appearing in the dataset.

        For the complete schema, use the tools_registry.csv shipped with
        the dataset release (see the `tools_file` config option).
        """
        if self._tools_schema is not None:
            return self._tools_schema

        self._tools_schema = []
        for tool_name in self.get_unique_tools():
            sample_params = {}
            for sample in self.samples:
                if sample.tool_name == tool_name and sample.extracted_params:
                    sample_params = sample.extracted_params
                    break

            self._tools_schema.append({
                "name": tool_name,
                "description": f"Tool: {tool_name}",
                "parameters": {
                    param: {"type": "string", "description": f"Parameter: {param}"}
                    for param in sample_params.keys()
                },
            })

        return self._tools_schema

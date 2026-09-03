"""
Tier11 Dataset Loader - Qwen3 Noise Ablation

Loads the tier11 dataset (qwen3 noise ablation data).
Same structure as tier2 (noise-ablated audio variants).
"""

import json
import csv
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import ast

from .base import BaseDataset, QuerySample, register_dataset

logger = logging.getLogger(__name__)


@register_dataset("tier11")
class Tier11Dataset(BaseDataset):
    """Dataset loader for tier11 (qwen3 noise ablation)."""
    
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
        **kwargs
    ):
        super().__init__(data_dir, filter_domain, filter_category, max_samples, speaker_idx, **kwargs)
        self.filter_tool = filter_tool
        self.speakers_per_query = speakers_per_query
        self.only_successful = only_successful
        self._tools_schema = None
        
    def load(self) -> None:
        if self._loaded:
            return
            
        logger.info(f"Loading Tier11 dataset from {self.data_dir}")
        
        query_dirs = sorted(self.data_dir.glob("query_*"))
        logger.info(f"Found {len(query_dirs)} query directories")
        
        loaded_count = 0
        for query_dir in query_dirs:
            if self.max_samples and loaded_count >= self.max_samples:
                break
                
            sample = self._load_query(query_dir)
            if sample is not None:
                self.samples.append(sample)
                loaded_count += 1
                
        self._loaded = True
        logger.info(f"Loaded {len(self.samples)} samples")
        
    def _load_query(self, query_dir: Path) -> Optional[QuerySample]:
        metadata_path = query_dir / "query_metadata.json"
        results_path = query_dir / "generation_results.csv"
        
        if not metadata_path.exists():
            return None
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        if self.filter_domain and metadata.get('domain') != self.filter_domain:
            return None
        if self.filter_category and metadata.get('category') != self.filter_category:
            return None
        if self.filter_tool and metadata.get('tool_name') != self.filter_tool:
            return None
            
        params_str = metadata.get('extracted_params', '{}')
        try:
            if params_str and isinstance(params_str, str) and params_str.startswith('{'):
                extracted_params = ast.literal_eval(params_str)
            else:
                extracted_params = {}
        except (ValueError, SyntaxError):
            extracted_params = {}
            
        audio_paths = self._get_audio_paths(query_dir, results_path)
        
        if not audio_paths:
            return None
            
        additional_tools = []
        additional_str = metadata.get('additional_tool_calls', '')
        if additional_str and additional_str != 'nan':
            try:
                additional_tools = ast.literal_eval(additional_str)
            except (ValueError, SyntaxError):
                pass
            
        return QuerySample(
            query_idx=metadata.get('query_idx', 0),
            query_text=metadata.get('query', ''),
            tool_name=metadata.get('tool_name', ''),
            tool_call=metadata.get('tool_call', ''),
            extracted_params=extracted_params,
            audio_paths=audio_paths,
            domain=metadata.get('domain', ''),
            category=metadata.get('category', ''),
            tier=metadata.get('tier', 'tier11'),
            additional_tool_calls=additional_tools,
            metadata={
                'tool_id': metadata.get('tool_id'),
                'source_endpoint': metadata.get('source_endpoint'),
            }
        )
        
    def _get_audio_paths(self, query_dir: Path, results_path: Path) -> List[str]:
        audio_paths = []
        
        if results_path.exists():
            with open(results_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if self.speaker_idx is not None:
                        if int(row.get('speaker_idx', -1)) != self.speaker_idx:
                            continue
                    if self.only_successful:
                        if row.get('success', 'True').lower() != 'true':
                            continue
                    audio_path = row.get('audio_path', '')
                    if audio_path:
                        p = Path(audio_path)
                        if p.exists():
                            audio_paths.append(audio_path)
                        else:
                            local = query_dir / p.name
                            if local.exists():
                                audio_paths.append(str(local))
                    if self.speakers_per_query and len(audio_paths) >= self.speakers_per_query:
                        break
        else:
            wav_files = sorted(query_dir.glob("speaker_*.wav"))
            for wav_file in wav_files:
                audio_paths.append(str(wav_file))
                if self.speakers_per_query and len(audio_paths) >= self.speakers_per_query:
                    break
                    
        return audio_paths
    
    def get_tools_schema(self) -> List[Dict[str, Any]]:
        if self._tools_schema is not None:
            return self._tools_schema
        unique_tools = self.get_unique_tools()
        self._tools_schema = []
        for tool_name in unique_tools:
            sample_params = {}
            for sample in self.samples:
                if sample.tool_name == tool_name and sample.extracted_params:
                    sample_params = sample.extracted_params
                    break
            tool_def = {
                "name": tool_name,
                "description": f"Tool: {tool_name}",
                "parameters": {
                    param: {"type": "string", "description": f"Parameter: {param}"}
                    for param in sample_params.keys()
                }
            }
            self._tools_schema.append(tool_def)
        return self._tools_schema
    
    def set_tools_schema(self, schema: List[Dict[str, Any]]) -> None:
        self._tools_schema = schema

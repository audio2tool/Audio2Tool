"""
Tier8 Dataset Loader - Intent Blending

Loads the tier8_intent_blending dataset which contains mixed audio
with foreground and background intents blended together.

Each query has:
- mixed.wav: Audio with foreground and background intents mixed
- query_metadata.json: Contains foreground and background tool info
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import ast

from .base import BaseDataset, QuerySample, register_dataset

logger = logging.getLogger(__name__)


@register_dataset("tier8")
class Tier8Dataset(BaseDataset):
    """
    Dataset loader for tier8_intent_blending.
    
    This dataset contains mixed audio with foreground and background
    intents. The model should identify the foreground tool call.
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
        **kwargs
    ):
        super().__init__(data_dir, filter_domain, filter_category, max_samples, speaker_idx, **kwargs)
        self.filter_tool = filter_tool
        self.speakers_per_query = speakers_per_query
        self._tools_schema = None
        
    def load(self) -> None:
        """Load the tier8 dataset from disk."""
        if self._loaded:
            logger.info("Dataset already loaded")
            return
            
        logger.info(f"Loading Tier8 dataset from {self.data_dir}")
        
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
        """Load a single query sample from a directory."""
        metadata_path = query_dir / "query_metadata.json"
        audio_path = query_dir / "mixed.wav"
        
        if not metadata_path.exists():
            return None
        if not audio_path.exists():
            return None
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Get foreground info (the primary tool to identify)
        foreground = metadata.get('foreground', {})
        background = metadata.get('background', {})
        
        # Apply filters based on foreground
        if self.filter_domain and foreground.get('domain') != self.filter_domain:
            return None
        if self.filter_category and foreground.get('category') != self.filter_category:
            return None
        if self.filter_tool and foreground.get('tool_name') != self.filter_tool:
            return None
            
        # Parse extracted_params from foreground
        params_str = foreground.get('extracted_params', '{}')
        try:
            if params_str and isinstance(params_str, str) and params_str.startswith('{'):
                extracted_params = ast.literal_eval(params_str)
            else:
                extracted_params = {}
        except (ValueError, SyntaxError):
            extracted_params = {}
            
        # Parse additional tool calls from foreground
        additional_tools = []
        additional_str = foreground.get('additional_tool_calls', '')
        if additional_str and additional_str != 'nan' and additional_str != '[]':
            try:
                additional_tools = ast.literal_eval(additional_str)
            except (ValueError, SyntaxError):
                pass
            
        return QuerySample(
            query_idx=metadata.get('query_idx', 0),
            query_text=foreground.get('query', ''),
            tool_name=foreground.get('tool_name', ''),
            tool_call=foreground.get('tool_call', ''),
            extracted_params=extracted_params,
            audio_paths=[str(audio_path)],
            domain=foreground.get('domain', ''),
            category=foreground.get('category', ''),
            tier='tier8_intent_blending',
            additional_tool_calls=additional_tools,
            metadata={
                'tool_id': foreground.get('tool_id'),
                'source_endpoint': foreground.get('source_endpoint'),
                'foreground_tier': foreground.get('tier'),
                'background_tool': background.get('tool_name'),
                'background_domain': background.get('domain'),
                'background_query': background.get('query'),
            }
        )
    
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

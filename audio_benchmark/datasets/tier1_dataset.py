"""
Tier1 Dataset Loader

Loads the tier1_queries_cleaned_v2 dataset which contains direct
tool calling queries with audio files from multiple speakers.
"""

import json
import csv
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import ast

from .base import BaseDataset, QuerySample, register_dataset

logger = logging.getLogger(__name__)


@register_dataset("tier1")
class Tier1Dataset(BaseDataset):
    """
    Dataset loader for tier1_queries_cleaned_v2.
    
    This dataset contains direct tool calling queries where each
    query maps to a single tool call. Each query has audio files
    from 25 different speakers.
    
    Directory structure:
        tier1_queries_cleaned_v2/
        ├── query_00000/
        │   ├── query_metadata.json
        │   ├── generation_results.csv
        │   └── speaker_*.wav
        ├── query_00001/
        │   └── ...
        └── ...
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
        **kwargs
    ):
        """
        Initialize Tier1 dataset loader.
        
        Args:
            data_dir: Path to tier1_queries_cleaned_v2 directory
            filter_domain: Only include specific domain (smart_car, smart_home, wearables)
            filter_category: Only include specific category
            filter_tool: Only include specific tool
            max_samples: Maximum number of query samples to load
            speaker_idx: Use only this speaker index (0-24)
            speakers_per_query: Number of speakers to use per query (default: all)
            only_successful: Only include successfully generated audio files
            **kwargs: Additional arguments
        """
        super().__init__(data_dir, filter_domain, filter_category, max_samples, speaker_idx, **kwargs)
        self.filter_tool = filter_tool
        self.speakers_per_query = speakers_per_query
        self.only_successful = only_successful
        self._tools_schema = None
        
    def load(self) -> None:
        """Load the tier1 dataset from disk."""
        if self._loaded:
            logger.info("Dataset already loaded")
            return
            
        logger.info(f"Loading Tier1 dataset from {self.data_dir}")
        
        # Find all query directories
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
        results_path = query_dir / "generation_results.csv"
        
        if not metadata_path.exists():
            logger.warning(f"Missing metadata: {metadata_path}")
            return None
            
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        # Apply filters
        if self.filter_domain and metadata.get('domain') != self.filter_domain:
            return None
        if self.filter_category and metadata.get('category') != self.filter_category:
            return None
        if self.filter_tool and metadata.get('tool_name') != self.filter_tool:
            return None
            
        # Parse extracted_params
        params_str = metadata.get('extracted_params', '{}')
        try:
            if params_str.startswith('{'):
                # Handle Python dict format {'key': 'value'}
                extracted_params = ast.literal_eval(params_str)
            else:
                extracted_params = json.loads(params_str)
        except (json.JSONDecodeError, ValueError, SyntaxError):
            extracted_params = {}
            
        # Get audio paths
        audio_paths = self._get_audio_paths(query_dir, results_path)
        
        if not audio_paths:
            logger.warning(f"No audio files found for {query_dir}")
            return None
        
        # Parse additional tool calls if present (usually empty for tier1)
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
            tier=metadata.get('tier', 'tier1_direct'),
            additional_tool_calls=additional_tools,
            metadata={
                'tool_id': metadata.get('tool_id'),
                'source_endpoint': metadata.get('source_endpoint'),
                'is_tier7': metadata.get('is_tier7', False),
            }
        )
        
    def _get_audio_paths(self, query_dir: Path, results_path: Path) -> List[str]:
        """Get audio file paths from generation results."""
        audio_paths = []
        
        if results_path.exists():
            with open(results_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Apply speaker index filter
                    if self.speaker_idx is not None:
                        if int(row.get('speaker_idx', -1)) != self.speaker_idx:
                            continue
                            
                    # Apply success filter
                    if self.only_successful:
                        if row.get('success', 'True').lower() != 'true':
                            continue
                            
                    audio_path = row.get('audio_path', '')
                    if audio_path:
                        p = Path(audio_path)
                        if p.exists():
                            audio_paths.append(audio_path)
                        else:
                            # CSV may contain stale absolute paths; try the
                            # filename inside the current query directory.
                            local = query_dir / p.name
                            if local.exists():
                                audio_paths.append(str(local))
                        
                    # Apply speakers per query limit
                    if self.speakers_per_query and len(audio_paths) >= self.speakers_per_query:
                        break
        else:
            # Fallback: find WAV files directly
            wav_files = sorted(query_dir.glob("speaker_*.wav"))
            for wav_file in wav_files:
                audio_paths.append(str(wav_file))
                if self.speakers_per_query and len(audio_paths) >= self.speakers_per_query:
                    break
                    
        return audio_paths
    
    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """
        Get the schema of available tools.
        
        Returns a simplified schema based on the unique tools in the dataset.
        For a complete schema, you should provide a separate tool definition file.
        """
        if self._tools_schema is not None:
            return self._tools_schema
            
        # Build schema from unique tools
        unique_tools = self.get_unique_tools()
        
        self._tools_schema = []
        for tool_name in unique_tools:
            # Get example parameters from dataset samples
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
        """Set a custom tools schema."""
        self._tools_schema = schema

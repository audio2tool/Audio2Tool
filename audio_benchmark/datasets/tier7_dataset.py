"""
Tier7 Dataset Loader

Loads the tier7_multiturn_queries dataset which contains multi-turn
conversational queries with audio files.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import ast

from .base import BaseDataset, QuerySample, register_dataset

logger = logging.getLogger(__name__)


@register_dataset("tier7")
class Tier7Dataset(BaseDataset):
    """
    Dataset loader for tier7_multiturn_queries.
    
    This dataset contains multi-turn conversational queries where
    each query may involve multiple turns of conversation.
    
    Directory structure:
        tier7_multiturn_queries/
        ├── query_00000/
        │   ├── query_metadata.json (if exists)
        │   ├── turn_00_user_*.wav
        │   ├── turn_01_agent_*.wav
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
        include_agent_turns: bool = False,
        **kwargs
    ):
        """
        Initialize Tier7 dataset loader.
        
        Args:
            data_dir: Path to tier7_multiturn_queries directory
            filter_domain: Only include specific domain
            filter_category: Only include specific category
            filter_tool: Only include specific tool
            max_samples: Maximum number of query samples to load
            speaker_idx: Not used for tier7 (each query has unique speakers)
            include_agent_turns: Include agent audio turns (for full conversation)
            **kwargs: Additional arguments
        """
        super().__init__(data_dir, filter_domain, filter_category, max_samples, speaker_idx, **kwargs)
        self.filter_tool = filter_tool
        self.include_agent_turns = include_agent_turns
        self._tools_schema = None
        
    def load(self) -> None:
        """Load the tier7 dataset from disk."""
        if self._loaded:
            logger.info("Dataset already loaded")
            return
            
        logger.info(f"Loading Tier7 dataset from {self.data_dir}")
        
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
        
        # Load metadata if available
        metadata = {}
        if metadata_path.exists():
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
                extracted_params = ast.literal_eval(params_str)
            else:
                extracted_params = json.loads(params_str)
        except (json.JSONDecodeError, ValueError, SyntaxError):
            extracted_params = {}
            
        # Get audio paths (user turns only by default)
        audio_paths = self._get_audio_paths(query_dir)
        
        if not audio_paths:
            logger.warning(f"No audio files found for {query_dir}")
            return None
            
        # Extract query index from directory name
        query_idx = int(query_dir.name.split('_')[1]) if '_' in query_dir.name else 0
        
        # Parse additional tool calls if present (usually empty for tier7)
        additional_tools = []
        additional_str = metadata.get('additional_tool_calls', '')
        if additional_str and additional_str != 'nan':
            try:
                additional_tools = ast.literal_eval(additional_str)
            except (ValueError, SyntaxError):
                pass
            
        return QuerySample(
            query_idx=metadata.get('query_idx', query_idx),
            query_text=metadata.get('query', ''),
            tool_name=metadata.get('tool_name', ''),
            tool_call=metadata.get('tool_call', ''),
            extracted_params=extracted_params,
            audio_paths=audio_paths,
            domain=metadata.get('domain', ''),
            category=metadata.get('category', ''),
            tier=metadata.get('tier', 'tier7_multiturn'),
            additional_tool_calls=additional_tools,
            metadata={
                'tool_id': metadata.get('tool_id'),
                'source_endpoint': metadata.get('source_endpoint'),
                'is_tier7': metadata.get('is_tier7', True),
                'turns': self._get_turn_info(query_dir),
            }
        )
        
    def _get_audio_paths(self, query_dir: Path) -> List[str]:
        """Get audio file paths from query directory."""
        audio_paths = []
        
        # Get user turn audio files
        user_turns = sorted(query_dir.glob("turn_*_user_*.wav"))
        for audio_file in user_turns:
            audio_paths.append(str(audio_file))
            
        # Optionally include agent turns
        if self.include_agent_turns:
            agent_turns = sorted(query_dir.glob("turn_*_agent_*.wav"))
            for audio_file in agent_turns:
                audio_paths.append(str(audio_file))
                
        return audio_paths
    
    def _get_turn_info(self, query_dir: Path) -> Dict[str, Any]:
        """Get information about conversation turns."""
        user_turns = list(query_dir.glob("turn_*_user_*.wav"))
        agent_turns = list(query_dir.glob("turn_*_agent_*.wav"))
        
        return {
            "num_user_turns": len(user_turns),
            "num_agent_turns": len(agent_turns),
            "total_turns": len(user_turns) + len(agent_turns),
        }
    
    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """
        Get the schema of available tools.
        
        Returns a simplified schema based on the unique tools in the dataset.
        """
        if self._tools_schema is not None:
            return self._tools_schema
            
        # Build schema from unique tools
        unique_tools = self.get_unique_tools()
        
        self._tools_schema = []
        for tool_name in unique_tools:
            if not tool_name:
                continue
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

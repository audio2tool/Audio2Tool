"""
Tool Schema Loader

Loads the taxonomy_tools CSV and formats it for system prompts.
"""

import csv
from pathlib import Path
from typing import Dict, List, Any

# Default path to tools taxonomy
DEFAULT_TOOLS_PATH = Path(__file__).parent.parent / "taxonomy_tools_5.backtracked.csv"


def load_tools_from_csv(csv_path: str = None) -> List[Dict[str, Any]]:
    """
    Load tools from the taxonomy CSV file.
    
    Args:
        csv_path: Path to CSV file (uses default if None)
        
    Returns:
        List of tool dictionaries with name, signature, description, etc.
    """
    csv_path = Path(csv_path) if csv_path else DEFAULT_TOOLS_PATH
    
    tools = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip deprecated tools
            if row.get('deprecated', '').lower() == 'true':
                continue
                
            tool = {
                'tool_id': int(row.get('tool_id', 0)),
                'domain': row.get('domain', ''),
                'category': row.get('category', ''),
                'name': row.get('tool_name', ''),
                'signature': row.get('signature', ''),
                'description': row.get('description', ''),
                'argument_defaults': row.get('argument_defaults', ''),
                'argument_constraints': row.get('argument_constraints', ''),
            }
            tools.append(tool)
    
    return tools


def format_tools_for_prompt(tools: List[Dict[str, Any]], include_constraints: bool = True) -> str:
    """
    Format tools into a string for system prompts.
    
    Args:
        tools: List of tool dictionaries
        include_constraints: Whether to include argument constraints
        
    Returns:
        Formatted string of tool definitions
    """
    lines = []
    
    # Group by domain
    domains = {}
    for tool in tools:
        domain = tool['domain']
        if domain not in domains:
            domains[domain] = []
        domains[domain].append(tool)
    
    for domain, domain_tools in sorted(domains.items()):
        lines.append(f"\n## {domain.upper().replace('_', ' ')}")
        
        for tool in sorted(domain_tools, key=lambda x: x['name']):
            lines.append(f"\n### {tool['signature']}")
            lines.append(f"{tool['description']}")
            if include_constraints and tool['argument_constraints']:
                lines.append(f"Constraints: {tool['argument_constraints']}")
    
    return "\n".join(lines)


def format_tools_compact(tools: List[Dict[str, Any]]) -> str:
    """
    Format tools in a compact format (just signatures and descriptions).
    
    Args:
        tools: List of tool dictionaries
        
    Returns:
        Compact formatted string
    """
    lines = []
    for tool in sorted(tools, key=lambda x: x['name']):
        lines.append(f"- {tool['signature']}: {tool['description']}")
    return "\n".join(lines)


def get_tool_names(tools: List[Dict[str, Any]]) -> List[str]:
    """Get just the tool names."""
    return [t['name'] for t in tools]


# Singleton cached tools
_CACHED_TOOLS = None

def get_tools(csv_path: str = None) -> List[Dict[str, Any]]:
    """Get tools (cached after first load)."""
    global _CACHED_TOOLS
    if _CACHED_TOOLS is None:
        _CACHED_TOOLS = load_tools_from_csv(csv_path)
    return _CACHED_TOOLS

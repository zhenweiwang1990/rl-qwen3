"""Profile search and reading tools for the people search agent.

This module only contains the return_final_answer function.
The search and read operations are handled directly by MCP tools via mcp_tools.py.
"""

from typing import List


def return_final_answer(profiles: List[str]) -> str:
    """Return the final list of matching profiles. 
    
    This function should be called when the agent has found the matching profiles. 
    If no matching profiles can be found, return an empty list.
    
    Args:
        profiles: List of linkedin_handle strings that match the user's query
    
    Returns:
        Confirmation message
    """
    ...

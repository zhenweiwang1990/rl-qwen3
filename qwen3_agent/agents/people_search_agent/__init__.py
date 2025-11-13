"""People search agent - find LinkedIn profiles matching user queries.

NOTE: This agent uses MCP tools directly for search and read operations.
The search_profiles and read_profile functions are no longer exported as they
are handled by the MCP server. Use PeopleSearchAgent to interact with profiles.
"""

from .agent import PeopleSearchAgent
from .tasks import PeopleSearchTask
from .evaluator import PeopleSearchEvaluator, PeopleSearchRubric
from .tools import return_final_answer
from .mcp_tools import get_search_and_read_tools, call_mcp_tool

__all__ = [
    "PeopleSearchAgent",
    "PeopleSearchTask",
    "PeopleSearchEvaluator",
    "PeopleSearchRubric",
    "return_final_answer",
    # MCP tools
    "get_search_and_read_tools",
    "call_mcp_tool",
]


"""People search agent implementation."""

from typing import Any, Dict, List
from dataclasses import asdict
import textwrap
import logging

from qwen3_agent.core.framework.agent import BaseAgent, ActionResult
from qwen3_agent.core.framework.task import BaseTask
from langchain_core.utils.function_calling import convert_to_openai_tool

from .tools import return_final_answer
from .mcp_tools import get_search_and_read_tools, call_mcp_tool
from .tasks import PeopleSearchTask
from .evaluator import PeopleSearchEvaluator, PeopleSearchRubric

logger = logging.getLogger(__name__)


class PeopleSearchAgent(BaseAgent):
    """People search agent.
    
    This agent can search and read profiles to find people matching user queries.
    
    Available tools:
    - search_profiles: Search for profiles using keywords
    - read_profile: Read full content of a specific profile
    - return_final_answer: Provide final list of matching profiles
    
    Example:
        ```python
        agent = PeopleSearchAgent()
        evaluator = PeopleSearchEvaluator()
        
        # Use with generic_rollout
        trajectory = await generic_rollout(
            llm=llm_inference,
            task=people_search_task,
            agent=agent,
            evaluator=evaluator,
        )
        ```
    """
    
    def __init__(self, evaluator: PeopleSearchEvaluator | None = None):
        """Initialize people search agent.
        
        Args:
            evaluator: Optional evaluator (needed for evaluate_final_answer)
        """
        self.evaluator = evaluator
        self._prepare_tools()
    
    def _prepare_tools(self):
        """Prepare tool schemas by fetching directly from MCP server."""
        # Get tool schemas directly from MCP server
        try:
            mcp_tools = get_search_and_read_tools()
            if len(mcp_tools) < 2:
                raise RuntimeError(f"Expected 2 MCP tools, got {len(mcp_tools)}")
            
            self.search_tool = mcp_tools[0]  # searchProfileTool
            self.read_tool = mcp_tools[1]    # readProfileTool
            logger.info("Loaded MCP tool schemas successfully")
            
        except Exception as e:
            logger.error(f"Failed to get MCP tools: {e}")
            raise RuntimeError(
                f"Cannot initialize agent without MCP tools. "
                f"Ensure MCP server is running at the configured URL. Error: {e}"
            )
        
        # return_final_answer is local
        self.answer_tool = convert_to_openai_tool(return_final_answer)
    
    def get_system_prompt(self, task: BaseTask) -> str:
        """Generate system prompt for people search task.
        
        Args:
            task: People search task
            
        Returns:
            System prompt string
        """
        assert isinstance(task, PeopleSearchTask), f"Expected PeopleSearchTask, got {type(task)}"
        
        # Get max_turns from evaluator if available
        max_turns = self.evaluator.max_turns if self.evaluator else 10
        
        prompt = textwrap.dedent(f"""\
            You are a people search agent specialized in finding LinkedIn profiles. \
You are given a user query describing the type of people they want to find, and a set of tools \
to search and read LinkedIn profiles. Use the tools to search profiles and find people \
that match the user's query. You may take up to {max_turns} turns to find the answer, \
so if your first search doesn't find enough matching profiles, you can try with different keywords.

            When searching:
            - Use relevant keywords from the query (job titles, industries, locations, skills, etc.)
            - Review search results snippets to identify promising candidates
            - Read full profiles to verify they match the query requirements
            - Return the linkedin_handles of profiles that match the user's criteria

            Important:
            - **Return AT MOST 10 profiles** - select the most relevant ones if there are more matches
            - Be thorough - try multiple search queries with different keywords
            - Verify profiles by reading them before including in final answer
            - Only include profiles that genuinely match the query requirements
            - Prioritize quality over quantity - it's better to return fewer highly relevant profiles
        """)
        
        return prompt
    
    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """Return OpenAI-format tool schemas.
        
        Returns:
            List of tool definitions
        """
        return [
            self.search_tool,
            self.read_tool,
            self.answer_tool,
        ]
    
    def execute_action(
        self, 
        tool_name: str, 
        tool_args: Dict[str, Any], 
        task: BaseTask
    ) -> ActionResult:
        """Execute a tool action.
        
        Args:
            tool_name: Name of tool to execute
            tool_args: Tool arguments
            task: Current task
            
        Returns:
            ActionResult with execution results
        """
        assert isinstance(task, PeopleSearchTask)
        
        try:
            # Map tool names (handle both MCP and local names)
            if tool_name in ("search_profiles", "searchProfileTool"):
                return self._execute_search(tool_args, task)
            
            elif tool_name in ("read_profile", "readProfileTool"):
                return self._execute_read(tool_args, task)
            
            elif tool_name == "return_final_answer":
                return self._execute_answer(tool_args, task)
            
            else:
                return ActionResult(
                    success=False,
                    data=None,
                    error=f"Unknown tool: {tool_name}"
                )
        
        except Exception as e:
            return ActionResult(
                success=False,
                data=None,
                error=f"Execution error: {str(e)}"
            )
    
    def _execute_search(self, tool_args: Dict[str, Any], task: PeopleSearchTask) -> ActionResult:
        """Execute searchProfileTool via MCP.
        
        Args:
            tool_args: Search arguments
            task: Current task
            
        Returns:
            ActionResult with search results
        """
        try:
            result = call_mcp_tool("searchProfileTool", tool_args)
            return ActionResult(success=True, data=result)
        except Exception as e:
            return ActionResult(
                success=False,
                data=None,
                error=f"MCP search failed: {str(e)}"
            )
    
    def _execute_read(self, tool_args: Dict[str, Any], task: PeopleSearchTask) -> ActionResult:
        """Execute readProfileTool via MCP.
        
        Args:
            tool_args: Read arguments
            task: Current task
            
        Returns:
            ActionResult with profile content or error
        """
        try:
            result = call_mcp_tool("readProfileTool", tool_args)
            return ActionResult(success=True, data=result)
        except Exception as e:
            return ActionResult(
                success=False,
                data=None,
                error=f"MCP read failed: {str(e)}"
            )
    
    def _execute_answer(self, tool_args: Dict[str, Any], task: PeopleSearchTask) -> ActionResult:
        """Execute return_final_answer tool.
        
        Args:
            tool_args: Answer arguments
            task: Current task
            
        Returns:
            ActionResult with answer status
        """
        profiles = tool_args.get("profiles")
        
        if profiles is None:
            return ActionResult(
                success=False,
                data=None,
                error="Missing required argument: profiles"
            )
        
        if not isinstance(profiles, list):
            return ActionResult(
                success=False,
                data=None,
                error=f"profiles must be a list, got {type(profiles)}"
            )
        
        # Store the answer for evaluation
        result_data = {
            "profiles": profiles,
            "status": "answer_provided",
            "count": len(profiles)
        }
        
        return ActionResult(success=True, data=result_data)
    
    def is_terminal_action(self, tool_name: str) -> bool:
        """Check if action terminates episode.
        
        Args:
            tool_name: Tool name
            
        Returns:
            True if this is return_final_answer
        """
        return tool_name == "return_final_answer"
    
    def get_tool_name_mapping(self) -> Dict[str, str]:
        """Get mapping from MCP tool names to internal names.
        
        Returns:
            Dict mapping MCP names to internal names
        """
        return {
            "searchProfileTool": "search_profiles",
            "readProfileTool": "read_profile",
        }


"""Email search agent implementation."""

from typing import Any, Dict, List
from dataclasses import asdict
import textwrap

from qwen3_agent.core.framework.agent import BaseAgent, ActionResult
from qwen3_agent.core.framework.task import BaseTask
from langchain_core.utils.function_calling import convert_to_openai_tool

from .tools import search_emails, read_email, return_final_answer
from .tasks import EmailTask
from .evaluator import EmailEvaluator, EmailRubric


class EmailAgent(BaseAgent):
    """Email search agent.
    
    This agent can search and read emails to answer user queries.
    
    Available tools:
    - search_emails: Search for emails using keywords and filters
    - read_email: Read full content of a specific email
    - return_final_answer: Provide final answer with sources
    
    Example:
        ```python
        agent = EmailAgent()
        evaluator = EmailEvaluator()
        
        # Use with generic_rollout
        trajectory = await generic_rollout(
            llm=llm_inference,
            task=email_task,
            agent=agent,
            evaluator=evaluator,
        )
        ```
    """
    
    def __init__(self, evaluator: EmailEvaluator | None = None):
        """Initialize email agent.
        
        Args:
            evaluator: Optional evaluator (needed for evaluate_final_answer)
        """
        self.evaluator = evaluator
        self._prepare_tools()
    
    def _prepare_tools(self):
        """Prepare tool schemas."""
        # Convert to OpenAI tool format
        self.search_tool = convert_to_openai_tool(search_emails)
        self.read_tool = convert_to_openai_tool(read_email)
        self.answer_tool = convert_to_openai_tool(return_final_answer)
        
        # Remove inbox parameter from search tool (will be injected from task)
        if "inbox" in self.search_tool["function"]["parameters"]["properties"]:
            del self.search_tool["function"]["parameters"]["properties"]["inbox"]
        if "inbox" in self.search_tool["function"]["parameters"].get("required", []):
            self.search_tool["function"]["parameters"]["required"].remove("inbox")
    
    def get_system_prompt(self, task: BaseTask) -> str:
        """Generate system prompt for email search task.
        
        Args:
            task: Email task
            
        Returns:
            System prompt string
        """
        assert isinstance(task, EmailTask), f"Expected EmailTask, got {type(task)}"
        
        # Get max_turns from evaluator if available
        max_turns = self.evaluator.max_turns if self.evaluator else 10
        
        prompt = textwrap.dedent(f"""\
            You are an email search agent. You are given a user query and a list of tools \
you can use to search the user's email. Use the tools to search the user's emails \
and find the answer to the user's query. You may take up to {max_turns} turns \
to find the answer, so if your first search doesn't find the answer, you can try \
with different keywords.

            User's email address is {task.inbox_address}
            Today's date is {task.query_date}
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
        assert isinstance(task, EmailTask)
        
        try:
            if tool_name == "search_emails":
                return self._execute_search(tool_args, task)
            
            elif tool_name == "read_email":
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
    
    def _execute_search(self, tool_args: Dict[str, Any], task: EmailTask) -> ActionResult:
        """Execute search_emails tool.
        
        Args:
            tool_args: Search arguments
            task: Current task
            
        Returns:
            ActionResult with search results
        """
        # Validate required args
        if "keywords" not in tool_args:
            return ActionResult(
                success=False,
                data=None,
                error="Missing required argument: keywords"
            )
        
        if not isinstance(tool_args["keywords"], list):
            return ActionResult(
                success=False,
                data=None,
                error="keywords must be a list"
            )
        
        # Execute search (inject inbox from task)
        results = search_emails(
            inbox=task.inbox_address,
            **tool_args
        )
        
        # Convert to dict format
        results_data = [asdict(r) for r in results]
        
        return ActionResult(success=True, data=results_data)
    
    def _execute_read(self, tool_args: Dict[str, Any], task: EmailTask) -> ActionResult:
        """Execute read_email tool.
        
        Args:
            tool_args: Read arguments
            task: Current task
            
        Returns:
            ActionResult with email content or error
        """
        message_id = tool_args.get("message_id")
        
        if not message_id:
            return ActionResult(
                success=False,
                data=None,
                error="Missing required argument: message_id"
            )
        
        if not isinstance(message_id, str):
            return ActionResult(
                success=False,
                data=None,
                error=f"message_id must be a string, got {type(message_id)}"
            )
        
        # Execute read
        email = read_email(message_id)
        
        if email is None:
            # Return success but with error in data (this is not an execution failure)
            return ActionResult(
                success=True,
                data={"error": "Email not found"}
            )
        
        return ActionResult(success=True, data=email.model_dump())
    
    def _execute_answer(self, tool_args: Dict[str, Any], task: EmailTask) -> ActionResult:
        """Execute return_final_answer tool.
        
        Args:
            tool_args: Answer arguments
            task: Current task
            
        Returns:
            ActionResult with answer status
        """
        answer = tool_args.get("answer")
        sources = tool_args.get("sources")
        
        if answer is None:
            return ActionResult(
                success=False,
                data=None,
                error="Missing required argument: answer"
            )
        
        if sources is None:
            return ActionResult(
                success=False,
                data=None,
                error="Missing required argument: sources"
            )
        
        if not isinstance(sources, list):
            return ActionResult(
                success=False,
                data=None,
                error=f"sources must be a list, got {type(sources)}"
            )
        
        # If evaluator is available, evaluate the answer
        # Note: This is an async operation, but we'll handle it in a special way
        # Store the answer/sources in the result so evaluator can process them
        result_data = {
            "answer": answer,
            "sources": sources,
            "status": "answer_provided"
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


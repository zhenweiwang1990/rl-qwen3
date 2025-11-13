"""Base agent abstraction for RL training."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import json


@dataclass
class ActionResult:
    """Result from executing an agent action/tool.
    
    Attributes:
        success: Whether the action executed successfully
        data: The result data (can be any type)
        error: Optional error message if action failed
    """
    success: bool
    data: Any
    error: Optional[str] = None


class BaseAgent(ABC):
    """Base class for all RL agents.
    
    An agent defines:
    - System prompt generation
    - Available tools/actions
    - Action execution logic
    - Action parsing from LLM responses
    
    Example:
        ```python
        class MyAgent(BaseAgent):
            def get_system_prompt(self, task: BaseTask) -> str:
                return f"You are an agent. Solve: {task.get_query()}"
            
            def get_tools_schema(self) -> List[Dict[str, Any]]:
                return [{"type": "function", "function": {...}}]
            
            def execute_action(self, tool_name, tool_args, task):
                if tool_name == "my_tool":
                    result = my_tool(**tool_args)
                    return ActionResult(success=True, data=result)
                return ActionResult(success=False, error="Unknown tool")
            
            def is_terminal_action(self, tool_name: str) -> bool:
                return tool_name == "return_answer"
        ```
    """
    
    @abstractmethod
    def get_system_prompt(self, task: "BaseTask") -> str:  # type: ignore
        """Generate system prompt for the task.
        
        This prompt will be used to instruct the LLM about its role,
        available tools, and the task context.
        
        Args:
            task: The task to solve
            
        Returns:
            System prompt string
        """
        pass
    
    @abstractmethod
    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """Return OpenAI-format tool schemas.
        
        Returns list of tool definitions in OpenAI function calling format.
        Return empty list if not using native tool calling.
        
        Returns:
            List of tool definitions in OpenAI format
        """
        pass
    
    @abstractmethod
    def execute_action(
        self, 
        tool_name: str, 
        tool_args: Dict[str, Any], 
        task: "BaseTask"  # type: ignore
    ) -> ActionResult:
        """Execute a tool/action.
        
        This is where the agent actually performs actions in the environment
        or uses tools to gather information.
        
        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool
            task: Current task context
            
        Returns:
            ActionResult with execution results
        """
        pass
    
    @abstractmethod
    def is_terminal_action(self, tool_name: str) -> bool:
        """Check if an action terminates the episode.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            True if this action ends the episode
        """
        pass
    
    def parse_action(
        self, 
        message: Any,
        use_native_tools: bool = True
    ) -> tuple[str, Dict[str, Any]]:
        """Parse LLM response into action.
        
        This method provides default parsing for both native tool calling
        and JSON-based tool calling. Override if you need custom parsing logic.
        
        Args:
            message: LLM message object (from litellm response.choices[0].message)
            use_native_tools: Whether native tool calling is used
            
        Returns:
            (tool_name, tool_args) tuple
            
        Raises:
            ValueError: If parsing fails
        """
        if use_native_tools:
            # Parse native tool calling
            if not hasattr(message, 'tool_calls') or not message.tool_calls:
                raise ValueError("No tool calls found in message")
            
            tool_call = message.tool_calls[0]
            if hasattr(tool_call, 'function'):
                # OpenAI format
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
            else:
                # Dict format
                tool_name = tool_call.get("function", {}).get("name")
                tool_args = json.loads(tool_call.get("function", {}).get("arguments", "{}"))
            
            if not tool_name:
                raise ValueError("Tool name not found in tool call")
            
            return tool_name, tool_args
        else:
            # Parse JSON-based tool calling
            content = message.content if hasattr(message, 'content') else str(message)
            
            if not content:
                raise ValueError("No content in message")
            
            # Find JSON object in content
            start_index = content.find("{")
            end_index = content.rfind("}")
            
            if start_index == -1 or end_index == -1 or start_index >= end_index:
                raise ValueError(f"No valid JSON found in response: {content[:200]}")
            
            json_str = content[start_index:end_index + 1]
            
            try:
                tool_call = json.loads(json_str)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON: {e}")
            
            tool_name = tool_call.get("tool_name")
            tool_args = tool_call.get("tool_args")
            
            if not tool_name:
                raise ValueError("tool_name not found in parsed JSON")
            if tool_args is None:
                raise ValueError("tool_args not found in parsed JSON")
            if not isinstance(tool_args, dict):
                raise ValueError(f"tool_args must be a dict, got {type(tool_args)}")
            
            return tool_name, tool_args


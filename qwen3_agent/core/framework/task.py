"""Base task abstraction for agent training and evaluation."""

from abc import ABC, abstractmethod
from typing import Any, Dict
from pydantic import BaseModel, Field


class BaseTask(BaseModel, ABC):
    """Base class for all agent tasks.
    
    A task represents a problem or query that an agent needs to solve.
    Subclasses should define task-specific fields and implement the abstract methods.
    
    Example:
        ```python
        class MyTask(BaseTask):
            question: str
            answer: str
            
            def get_query(self) -> str:
                return self.question
            
            def get_ground_truth(self) -> Any:
                return self.answer
        ```
    """
    
    # Common fields for all tasks
    id: str = Field(description="Unique identifier for this task")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the task"
    )
    
    @abstractmethod
    def get_query(self) -> str:
        """Get the user query/instruction for this task.
        
        This is the text that will be presented to the agent as the user's question
        or instruction.
        
        Returns:
            Query string to present to the agent
        """
        pass
    
    @abstractmethod
    def get_ground_truth(self) -> Any:
        """Get ground truth answer/solution for evaluation.
        
        This is the correct answer or expected output that will be used
        to evaluate the agent's performance.
        
        Returns:
            Ground truth data (can be any type depending on task)
        """
        pass
    
    def get_context(self) -> Dict[str, Any]:
        """Get additional context for the task.
        
        Override this to provide task-specific context that may be needed
        during rollout or evaluation (e.g., environment state, constraints).
        
        Returns:
            Dictionary of contextual information
        """
        return self.metadata
    
    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True


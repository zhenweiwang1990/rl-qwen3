"""Base evaluator abstraction for agent evaluation and reward calculation."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, TypeVar
from dataclasses import dataclass, asdict


@dataclass
class BaseRubric:
    """Base rubric with common metrics.
    
    Subclasses should add task-specific metrics as fields.
    
    Attributes:
        num_turns: Number of turns taken
        prompt_tokens: Total prompt tokens used
        completion_tokens: Total completion tokens used
    """
    num_turns: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    
    def to_metrics(self) -> Dict[str, float]:
        """Convert rubric to metrics dictionary.
        
        Returns:
            Dictionary mapping metric names to values
        """
        return {k: float(v) for k, v in asdict(self).items()}


# Type variable for rubric type
RubricType = TypeVar('RubricType', bound=BaseRubric)


class BaseEvaluator(ABC, Generic[RubricType]):
    """Base class for task evaluators.
    
    An evaluator is responsible for:
    - Creating rubric instances to track metrics
    - Updating rubric during action execution
    - Calculating final reward based on trajectory
    
    Example:
        ```python
        @dataclass
        class MyRubric(BaseRubric):
            answer_correct: bool = False
            
        class MyEvaluator(BaseEvaluator[MyRubric]):
            def create_rubric(self) -> MyRubric:
                return MyRubric()
            
            async def evaluate_trajectory(self, traj, task, rubric):
                return 1.0 if rubric.answer_correct else 0.0
            
            def on_action_executed(self, rubric, tool_name, tool_args, result, task):
                if tool_name == "answer" and result == task.get_ground_truth():
                    rubric.answer_correct = True
        ```
    """
    
    @abstractmethod
    def create_rubric(self) -> RubricType:
        """Create a new rubric instance.
        
        Returns:
            Fresh rubric object for tracking metrics
        """
        pass
    
    @abstractmethod
    async def evaluate_trajectory(
        self,
        trajectory: Any,  # art.Trajectory
        task: "BaseTask",  # type: ignore
        rubric: RubricType,
    ) -> float:
        """Evaluate a complete trajectory and calculate reward.
        
        This is called after the trajectory is complete to compute the final reward.
        
        Args:
            trajectory: The completed trajectory (art.Trajectory)
            task: The task that was attempted
            rubric: The rubric tracking metrics
            
        Returns:
            Final reward value
        """
        pass
    
    @abstractmethod
    def on_action_executed(
        self,
        rubric: RubricType,
        tool_name: str,
        tool_args: Dict[str, Any],
        result: Any,
        task: "BaseTask",  # type: ignore
    ) -> None:
        """Update rubric after action execution.
        
        This is called after each action to update metrics based on
        what the agent did and what result it got.
        
        Args:
            rubric: Current rubric
            tool_name: Tool that was executed
            tool_args: Arguments passed to tool
            result: Execution result (data from ActionResult)
            task: Current task
        """
        pass
    
    def on_parsing_error(
        self,
        rubric: RubricType,
        error_type: str,
        error_message: str,
    ) -> None:
        """Handle parsing or execution errors.
        
        Override this to track error-related metrics in your rubric.
        
        Args:
            rubric: Current rubric
            error_type: Type of error (e.g., "parse_error", "llm_error")
            error_message: Error details
        """
        pass


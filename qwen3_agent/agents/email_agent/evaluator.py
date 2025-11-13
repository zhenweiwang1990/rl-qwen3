"""Email agent evaluator and reward calculation."""

from dataclasses import dataclass
from typing import Any, Dict
import logging

from qwen3_agent.core.framework.evaluator import BaseEvaluator, BaseRubric
from qwen3_agent.core.framework.task import BaseTask
from qwen3_agent.evaluation import determine_if_answer_is_correct
from .tasks import EmailTask

logger = logging.getLogger(__name__)


@dataclass
class EmailRubric(BaseRubric):
    """Rubric for evaluating email agent performance.
    
    Tracks various metrics specific to the email search task.
    """
    
    # Correctness metrics
    answer_correct: bool = False
    sources_correct: bool = False
    attempted_answer: bool = False
    
    # Progress metrics
    ever_found_right_email: bool = False
    ever_read_right_email: bool = False
    
    # Error metrics
    cant_parse_tool_call: bool = False
    bad_tool_call_name: bool = False
    bad_tool_call_args: bool = False
    ever_tried_to_read_invalid_email: bool = False
    
    # Termination metrics
    ran_out_of_turns: bool = False
    returned_i_dont_know: bool = False
    
    # Additional metrics
    num_sources: int = 0


class EmailEvaluator(BaseEvaluator[EmailRubric]):
    """Evaluator for email search tasks.
    
    Calculates rewards based on:
    - Answer correctness (verified by GPT-4o judge)
    - Source correctness (whether correct email was cited)
    - Partial credit for progress (finding/reading right email)
    - Penalties for errors and inefficiency
    """
    
    def __init__(
        self,
        simple_reward: bool = False,
        verbose: bool = False,
        max_turns: int = 10,
    ):
        """Initialize evaluator.
        
        Args:
            simple_reward: Use simple 0/1 reward (correct/incorrect)
            verbose: Print detailed evaluation logs
            max_turns: Maximum turns (for efficiency penalty)
        """
        self.simple_reward = simple_reward
        self.verbose = verbose
        self.max_turns = max_turns
    
    def create_rubric(self) -> EmailRubric:
        """Create a new rubric instance.
        
        Returns:
            Fresh EmailRubric
        """
        return EmailRubric()
    
    async def evaluate_trajectory(
        self,
        trajectory: Any,  # art.Trajectory
        task: BaseTask,
        rubric: EmailRubric,
    ) -> float:
        """Evaluate trajectory and calculate reward.
        
        This method performs async evaluation of the answer using GPT-4o judge.
        
        Args:
            trajectory: Completed trajectory
            task: Email task
            rubric: Rubric with tracked metrics
            
        Returns:
            Final reward value between -2 and 2
        """
        assert isinstance(task, EmailTask), f"Expected EmailTask, got {type(task)}"
        
        # Handle max turns
        if rubric.num_turns >= self.max_turns:
            rubric.ran_out_of_turns = True
        
        # If agent attempted an answer, evaluate it with GPT-4o
        if rubric.attempted_answer:
            # Extract answer from trajectory messages
            answer = self._extract_answer_from_trajectory(trajectory)
            
            if answer:
                # Use GPT-4o to judge correctness
                try:
                    rubric.answer_correct = await determine_if_answer_is_correct(
                        answer,
                        task,
                        verbose=self.verbose
                    )
                except Exception as e:
                    logger.error(f"Failed to evaluate answer: {e}")
                    rubric.answer_correct = False
        
        # Calculate reward
        reward = self._calculate_reward(rubric)
        
        if self.verbose:
            logger.info(
                f"Evaluation - Task: {task.id}, "
                f"Reward: {reward:.2f}, "
                f"Answer correct: {rubric.answer_correct}, "
                f"Sources correct: {rubric.sources_correct}"
            )
        
        return reward
    
    def _extract_answer_from_trajectory(self, trajectory: Any) -> str | None:
        """Extract the answer string from trajectory messages.
        
        Args:
            trajectory: Trajectory object
            
        Returns:
            Answer string or None
        """
        # Look for the tool response containing the answer
        messages = trajectory.messages()
        
        for i, msg in enumerate(messages):
            if msg.get("role") == "assistant":
                # Check if this is a tool call to return_final_answer
                if "tool_calls" in msg:
                    tool_calls = msg["tool_calls"]
                    if tool_calls and len(tool_calls) > 0:
                        tool_call = tool_calls[0]
                        if isinstance(tool_call, dict):
                            func = tool_call.get("function", {})
                            if func.get("name") == "return_final_answer":
                                import json
                                try:
                                    args = json.loads(func.get("arguments", "{}"))
                                    return args.get("answer")
                                except:
                                    pass
        
        # Fallback: look in tool responses
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                try:
                    import json
                    data = json.loads(content)
                    if "answer" in data:
                        return data["answer"]
                except:
                    pass
        
        return None
    
    def on_action_executed(
        self,
        rubric: EmailRubric,
        tool_name: str,
        tool_args: Dict[str, Any],
        result: Any,
        task: BaseTask,
    ) -> None:
        """Update rubric based on action execution.
        
        Args:
            rubric: Current rubric
            tool_name: Name of executed tool
            tool_args: Tool arguments
            result: Tool execution result
            task: Current task
        """
        assert isinstance(task, EmailTask)
        ground_truth = task.get_ground_truth()
        correct_message_id = ground_truth["message_ids"][0]
        
        if tool_name == "search_emails":
            # Check if correct email was found
            if isinstance(result, list):
                for item in result:
                    if isinstance(item, dict) and item.get("message_id") == correct_message_id:
                        rubric.ever_found_right_email = True
                        break
        
        elif tool_name == "read_email":
            # Check if correct email was read
            message_id = tool_args.get("message_id")
            if message_id == correct_message_id:
                rubric.ever_read_right_email = True
            
            # Check if invalid email was attempted
            if isinstance(result, dict) and result.get("error") == "Email not found":
                rubric.ever_tried_to_read_invalid_email = True
        
        elif tool_name == "return_final_answer":
            # Store answer and sources for later evaluation
            # This will be evaluated async in evaluate_trajectory
            if isinstance(result, dict):
                answer = result.get("answer")
                sources = result.get("sources", [])
                
                # Store for later async evaluation
                rubric.num_sources = len(sources)
                
                # Track I don't know responses
                if answer == "I don't know":
                    rubric.returned_i_dont_know = True
                else:
                    rubric.attempted_answer = True
                
                # Check sources
                rubric.sources_correct = correct_message_id in sources
    
    def on_parsing_error(
        self,
        rubric: EmailRubric,
        error_type: str,
        error_message: str,
    ) -> None:
        """Track parsing/execution errors.
        
        Args:
            rubric: Current rubric
            error_type: Type of error
            error_message: Error details
        """
        if error_type == "parse_error":
            rubric.cant_parse_tool_call = True
        elif error_type == "action_error":
            if "Unknown tool" in error_message:
                rubric.bad_tool_call_name = True
            else:
                rubric.bad_tool_call_args = True
        elif error_type in ("llm_error", "execution_error"):
            rubric.bad_tool_call_args = True
    
    def _calculate_reward(self, rubric: EmailRubric) -> float:
        """Calculate reward from rubric.
        
        Args:
            rubric: Evaluation rubric
            
        Returns:
            Reward value between -2 and 2
        """
        # Simple reward: 1 for correct, 0 otherwise
        if self.simple_reward:
            return float(rubric.answer_correct)
        
        # Complex reward with partial credit
        partial_rewards = 0.0
        partial_rewards += 0.1 if rubric.ever_found_right_email else 0.0
        partial_rewards += 0.1 if rubric.ever_read_right_email else 0.0
        partial_rewards += 0.1 if not rubric.ever_tried_to_read_invalid_email else 0.0
        partial_rewards += 0.1 if rubric.sources_correct else 0.0
        
        # Formatting errors: -2 to -1
        if rubric.cant_parse_tool_call:
            return -2.0 + partial_rewards
        
        if rubric.bad_tool_call_name:
            return -1.9 + partial_rewards
        
        if rubric.bad_tool_call_args:
            return -1.8 + partial_rewards
        
        # Wrong answer: -1 to 0
        if rubric.attempted_answer and not rubric.answer_correct:
            return -1.0 + partial_rewards
        
        # No answer: 0 to 1
        if rubric.returned_i_dont_know or rubric.ran_out_of_turns:
            return 0.0 + partial_rewards
        
        # Correct answer: 1 to 2
        if rubric.answer_correct:
            reward = 1.0
            reward += 0.3 if rubric.sources_correct else 0.0
            reward += 0.1 / max(rubric.num_sources, 1)
            reward += 0.1 * (1 - rubric.num_turns / self.max_turns)
            return reward
        
        # Shouldn't reach here
        logger.warning(f"Rubric not handled properly: {rubric}")
        return 0.0


"""People search agent evaluator and reward calculation."""

from dataclasses import dataclass
from typing import Any, Dict, Set, List
import logging

from qwen3_agent.core.framework.evaluator import BaseEvaluator, BaseRubric
from qwen3_agent.core.framework.task import BaseTask
from .tasks import PeopleSearchTask

logger = logging.getLogger(__name__)


@dataclass
class PeopleSearchRubric(BaseRubric):
    """Rubric for evaluating people search agent performance.
    
    Tracks various metrics specific to the people search task.
    """
    
    # Correctness metrics
    answer_correct: bool = False
    answer_perfect_match: bool = False  # All profiles match exactly
    attempted_answer: bool = False
    
    # Progress metrics
    profiles_found_in_search: Set[str] = None  # Profiles from expected list found in search results
    profiles_read_correct: Set[str] = None  # Correct profiles that were read
    
    # Error metrics
    cant_parse_tool_call: bool = False
    bad_tool_call_name: bool = False
    bad_tool_call_args: bool = False
    ever_tried_to_read_invalid_profile: bool = False
    
    # Termination metrics
    ran_out_of_turns: bool = False
    returned_empty_list: bool = False
    
    # Answer metrics
    answer_profiles: Set[str] = None  # Profiles in the final answer
    expected_profiles: Set[str] = None  # Ground truth profiles
    
    def __post_init__(self):
        """Initialize sets."""
        if self.profiles_found_in_search is None:
            self.profiles_found_in_search = set()
        if self.profiles_read_correct is None:
            self.profiles_read_correct = set()
        if self.answer_profiles is None:
            self.answer_profiles = set()
        if self.expected_profiles is None:
            self.expected_profiles = set()


class PeopleSearchEvaluator(BaseEvaluator[PeopleSearchRubric]):
    """Evaluator for people search tasks.
    
    Reward calculation based on rule-based scoring, range approximately [-3, 4]:
    
    Error penalties (highest priority):
    - Parse errors: -2.8 + partial
    - Bad tool name: -2.5 + partial
    - Bad tool args: -2.0 + partial
    - Completely wrong answer (intersection = 0): -2.0 + partial
    
    Perfect match (highest reward): 3.0 + partial
    - When len(expected) > 10: Agent returns 10 profiles, all in expected
    - When len(expected) <= 10: Agent answer fully covers expected profiles
    
    Tiered rewards based on coverage (拉开差距的阶梯式评分):
    - coverage_ratio = len(intersection) / min(10, len(expected))
    - coverage_ratio == 1.0: 3.0 (perfect coverage)
    - coverage_ratio >= 0.8: 2.3
    - coverage_ratio >= 0.6: 1.8
    - coverage_ratio >= 0.4: 1.2
    - coverage_ratio >= 0.2: 0.6
    - coverage_ratio > 0: 0.2
    - coverage_ratio == 0: -2.0
    
    Partial rewards (max ~1.1):
    - +0.1 per correct profile found in search results (max 0.5)
    - +0.1 per correct profile read (max 0.5)
    - +0.1 if didn't try to read invalid profiles
    
    No answer: 0.0 + partial
    
    Key changes:
    - NO penalty for profiles outside expected list (用户不反感额外结果)
    - Coverage based on intersection size, not ratio to total expected
    - Flexible perfect match definition based on expected profile count
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
    
    def create_rubric(self) -> PeopleSearchRubric:
        """Create a new rubric instance.
        
        Returns:
            Fresh PeopleSearchRubric
        """
        return PeopleSearchRubric()
    
    async def evaluate_trajectory(
        self,
        trajectory: Any,
        task: BaseTask,
        rubric: PeopleSearchRubric,
    ) -> float:
        """Evaluate trajectory and calculate reward.
        
        Args:
            trajectory: Completed trajectory
            task: People search task
            rubric: Rubric with tracked metrics
            
        Returns:
            Final reward value between -3 and 4
        """
        assert isinstance(task, PeopleSearchTask), f"Expected PeopleSearchTask, got {type(task)}"
        
        # Store expected profiles
        ground_truth = task.get_ground_truth()
        rubric.expected_profiles = set(ground_truth["expected_profiles"])
        
        # Handle max turns
        if rubric.num_turns >= self.max_turns:
            rubric.ran_out_of_turns = True
        
        # If agent attempted an answer, evaluate it
        if rubric.attempted_answer and len(rubric.answer_profiles) > 0:
            # Calculate intersection
            intersection = rubric.answer_profiles & rubric.expected_profiles
            
            # Check for perfect match based on expected profile count
            if len(rubric.expected_profiles) > 10:
                # Expected > 10: Perfect if agent returns 10 profiles all in expected
                if (len(rubric.answer_profiles) == 10 and 
                    rubric.answer_profiles.issubset(rubric.expected_profiles)):
                    rubric.answer_perfect_match = True
                    rubric.answer_correct = True
            else:
                # Expected <= 10: Perfect if agent fully covers expected
                if rubric.expected_profiles.issubset(rubric.answer_profiles):
                    rubric.answer_perfect_match = True
                    rubric.answer_correct = True
            
            # Check if answer has any correct profiles
            if len(intersection) > 0:
                rubric.answer_correct = True
            else:
                # No profiles match - wrong answer
                rubric.answer_correct = False
        
        # Calculate reward
        reward = self._calculate_reward(rubric)
        
        if self.verbose:
            logger.info(
                f"Evaluation - Task: {task.id}, "
                f"Reward: {reward:.2f}, "
                f"Answer correct: {rubric.answer_correct}, "
                f"Perfect match: {rubric.answer_perfect_match}, "
                f"Found: {len(rubric.profiles_found_in_search)}/{len(rubric.expected_profiles)}, "
                f"Read: {len(rubric.profiles_read_correct)}/{len(rubric.expected_profiles)}, "
                f"Returned: {len(rubric.answer_profiles)}"
            )
        
        return reward
    
    def on_action_executed(
        self,
        rubric: PeopleSearchRubric,
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
        assert isinstance(task, PeopleSearchTask)
        ground_truth = task.get_ground_truth()
        expected_profiles_set = set(ground_truth["expected_profiles"])
        
        if tool_name == "search_profiles":
            # Check if correct profiles were found
            if isinstance(result, list):
                for item in result:
                    if isinstance(item, dict):
                        linkedin_handle = item.get("linkedin_handle")
                        if linkedin_handle and linkedin_handle in expected_profiles_set:
                            rubric.profiles_found_in_search.add(linkedin_handle)
        
        elif tool_name == "read_profile":
            # Check if correct profile was read
            linkedin_handle = tool_args.get("linkedin_handle")
            
            # Check if profile exists (result is not None and not error)
            if isinstance(result, dict) and result.get("error") == "Profile not found":
                rubric.ever_tried_to_read_invalid_profile = True
            elif linkedin_handle and linkedin_handle in expected_profiles_set:
                # Correct profile was read
                rubric.profiles_read_correct.add(linkedin_handle)
        
        elif tool_name == "return_final_answer":
            # Store answer profiles for evaluation
            if isinstance(result, dict):
                profiles = result.get("profiles", [])
                
                if not profiles or len(profiles) == 0:
                    rubric.returned_empty_list = True
                else:
                    rubric.attempted_answer = True
                    rubric.answer_profiles = set(profiles)
    
    def on_parsing_error(
        self,
        rubric: PeopleSearchRubric,
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
    
    def _calculate_reward(self, rubric: PeopleSearchRubric) -> float:
        """Calculate reward from rubric based on rule-based scoring.
        
        Args:
            rubric: Evaluation rubric
            
        Returns:
            Reward value between -3 and 4
        """
        # Simple reward: 1 for correct, 0 otherwise
        if self.simple_reward:
            return float(rubric.answer_correct)
        
        # Calculate partial rewards (max ~1.1)
        partial_rewards = 0.0
        
        # +0.1 per correct profile found in search (max 0.5)
        found_bonus = min(len(rubric.profiles_found_in_search) * 0.1, 0.5)
        partial_rewards += found_bonus
        
        # +0.1 per correct profile read (max 0.5)
        read_bonus = min(len(rubric.profiles_read_correct) * 0.1, 0.5)
        partial_rewards += read_bonus
        
        # +0.1 if didn't try to read invalid profiles
        if not rubric.ever_tried_to_read_invalid_profile:
            partial_rewards += 0.1
        
        # Parse/tool call format errors: -2.0 to -2.8 + partial
        if rubric.cant_parse_tool_call:
            return -2.8 + partial_rewards
        
        if rubric.bad_tool_call_name:
            return -2.5 + partial_rewards
        
        if rubric.bad_tool_call_args:
            return -2.0 + partial_rewards
        
        # No answer or ran out of turns: 0.0 + partial
        if rubric.returned_empty_list or rubric.ran_out_of_turns:
            return 0.0 + partial_rewards
        
        # If agent attempted an answer, calculate tiered reward
        if rubric.attempted_answer and len(rubric.expected_profiles) > 0:
            # Calculate intersection and coverage
            intersection = rubric.answer_profiles & rubric.expected_profiles
            intersection_size = len(intersection)
            target_max = min(10, len(rubric.expected_profiles))
            coverage_ratio = intersection_size / target_max if target_max > 0 else 0.0
            
            # Perfect match: highest reward
            if rubric.answer_perfect_match:
                return 3.0 + partial_rewards
            
            # Tiered rewards based on coverage ratio
            if coverage_ratio >= 0.8:
                base_reward = 2.3
            elif coverage_ratio >= 0.6:
                base_reward = 1.8
            elif coverage_ratio >= 0.4:
                base_reward = 1.2
            elif coverage_ratio >= 0.2:
                base_reward = 0.6
            elif coverage_ratio > 0:
                base_reward = 0.2
            else:  # coverage_ratio == 0, completely wrong
                base_reward = -2.0
            
            return base_reward + partial_rewards
        
        # Edge case: no answer attempted and didn't run out of turns
        logger.warning(f"Rubric not handled properly: {rubric}")
        return 0.0


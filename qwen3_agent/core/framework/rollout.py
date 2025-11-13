"""Generic rollout function for agent evaluation and training."""

from typing import Optional
import art
from art import Trajectory
from art.utils import limit_concurrency
from art.utils.litellm import convert_litellm_choice_to_openai
from litellm.types.utils import Choices, ModelResponse
from datetime import datetime
import os
import json
import logging

from .agent import BaseAgent
from .task import BaseTask
from .evaluator import BaseEvaluator
from .llm_inference import LLMInference

logger = logging.getLogger(__name__)


@limit_concurrency(
    int(os.environ.get("ROLLOUT_CONCURRENCY", "10")),
    derive_key=lambda llm, task, agent, evaluator, **kw: (
        llm.model.name if hasattr(llm.model, 'name') else str(llm.model)
    )
)
async def generic_rollout(
    llm: LLMInference,
    task: BaseTask,
    agent: BaseAgent,
    evaluator: BaseEvaluator,
    max_turns: int = 10,
    use_native_tools: bool = True,
    verbose: bool = False,
) -> Trajectory:
    """Generic rollout function for any agent/task combination.
    
    This function executes a complete interaction between an agent and a task,
    tracking metrics and calculating rewards. It's designed to work with any
    agent/task implementation that follows the framework interfaces.
    
    The rollout process:
    1. Initialize conversation with system prompt and user query
    2. Loop up to max_turns:
       - Get LLM response
       - Parse action from response
       - Execute action via agent
       - Update evaluator metrics
       - Check if terminal action
    3. Calculate final reward
    4. Return trajectory with messages, reward, and metrics
    
    Args:
        llm: LLM inference interface (wraps ART model or external model)
        task: Task to solve (implements BaseTask)
        agent: Agent implementation (implements BaseAgent)
        evaluator: Evaluator for rewards (implements BaseEvaluator)
        max_turns: Maximum turns allowed
        use_native_tools: Use native function calling (vs JSON-based)
        verbose: Print detailed logs
        
    Returns:
        Trajectory with messages, reward, and metrics
    """
    rollout_start_time = datetime.now()
    rubric = evaluator.create_rubric()
    traj = Trajectory(
        messages_and_choices=[],
        reward=0,
        metadata={"task_id": task.id, **task.get_context()},
    )
    
    # Setup system prompt and tools
    system_prompt = agent.get_system_prompt(task)
    tools = agent.get_tools_schema() if use_native_tools else None
    
    if tools:
        traj.tools = tools
    
    # Initialize conversation
    traj.messages_and_choices = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task.get_query()},
    ]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Starting rollout for task: {task.id}")
        print(f"{'='*60}")
    
    # Main interaction loop
    while rubric.num_turns < max_turns:
        rubric.num_turns += 1
        
        if verbose:
            print(f"\n--- Turn {rubric.num_turns}/{max_turns} ---")
        
        # Get LLM response
        try:
            response = await llm.complete(
                messages=traj.messages(),
                tools=tools,
            )
        except Exception as e:
            # Log error and break
            error_msg = f"LLM error: {getattr(e, 'message', str(e))}"
            traj.logs.append(error_msg)
            if verbose:
                print(f"ERROR: {error_msg}")
                logger.error(f"LLM call failed: {e}", exc_info=True)
            evaluator.on_parsing_error(rubric, "llm_error", str(e))
            break
        
        # Track token usage
        assert isinstance(response, ModelResponse)
        if response.usage:
            rubric.prompt_tokens += response.usage.prompt_tokens  # type: ignore
            rubric.completion_tokens += response.usage.completion_tokens  # type: ignore
        
        choice = response.choices[0]  # type: ignore
        assert isinstance(choice, Choices)
        
        # Handle multiple tool calls (only use the first one)
        if choice.message.tool_calls is not None and len(choice.message.tool_calls) > 1:
            choice.message.tool_calls = choice.message.tool_calls[:1]
        
        # Add to trajectory (format depends on whether model is trainable)
        if llm.is_trainable:
            traj.messages_and_choices.append(convert_litellm_choice_to_openai(choice))
        else:
            traj.messages_and_choices.append(choice.message.to_dict())  # type: ignore
        
        # Parse action from LLM response
        try:
            tool_name, tool_args = agent.parse_action(choice.message, use_native_tools)
        except Exception as e:
            error_msg = f"Parse error: {str(e)}"
            traj.logs.append(error_msg)
            if verbose:
                print(f"ERROR: {error_msg}")
                logger.warning(f"Failed to parse action: {e}")
            evaluator.on_parsing_error(rubric, "parse_error", str(e))
            break
        
        if verbose:
            print(f"Tool: {tool_name}")
            print(f"Args: {json.dumps(tool_args, indent=2)}")
        
        # Execute action
        try:
            result = agent.execute_action(tool_name, tool_args, task)
            
            if not result.success:
                error_msg = f"Action failed: {result.error}"
                traj.logs.append(error_msg)
                if verbose:
                    print(f"ERROR: {error_msg}")
                evaluator.on_parsing_error(rubric, "action_error", result.error or "")
                break
            
            # Update evaluator with action result
            evaluator.on_action_executed(rubric, tool_name, tool_args, result.data, task)
            
            # Add tool response to conversation
            # Format as tool message for native tools, user message otherwise
            if use_native_tools and choice.message.tool_calls:
                tool_response = {
                    "role": "tool",
                    "tool_call_id": choice.message.tool_calls[0].id,
                    "content": json.dumps(result.data),
                }
            else:
                tool_response = {
                    "role": "user",
                    "content": json.dumps(result.data),
                }
            
            traj.messages_and_choices.append(tool_response)
            
            if verbose:
                # Show truncated result
                result_str = json.dumps(result.data)
                if len(result_str) > 200:
                    print(f"Result: {result_str[:200]}...")
                else:
                    print(f"Result: {result_str}")
            
        except Exception as e:
            error_msg = f"Execution error: {str(e)}"
            traj.logs.append(error_msg)
            if verbose:
                print(f"ERROR: {error_msg}")
                logger.error(f"Action execution failed: {e}", exc_info=True)
            evaluator.on_parsing_error(rubric, "execution_error", str(e))
            break
        
        # Check if terminal action
        if agent.is_terminal_action(tool_name):
            if verbose:
                print(f"\nTerminal action reached: {tool_name}")
            break
    
    # Check if ran out of turns
    if rubric.num_turns >= max_turns and verbose:
        print(f"\nReached maximum turns ({max_turns})")
    
    # Calculate final reward
    reward = await evaluator.evaluate_trajectory(traj, task, rubric)
    traj.reward = reward
    traj.metrics = rubric.to_metrics()
    
    # Calculate duration
    rollout_end_time = datetime.now()
    duration_seconds = (rollout_end_time - rollout_start_time).total_seconds()
    traj.metrics["duration"] = duration_seconds
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Rollout completed in {duration_seconds:.2f}s")
        print(f"Reward: {reward:.2f}")
        print(f"Turns: {rubric.num_turns}")
        print(f"{'='*60}\n")
    
    # Log summary
    logger.info(
        f"Rollout completed - Task: {task.id}, "
        f"Duration: {duration_seconds:.2f}s, "
        f"Reward: {reward:.2f}, "
        f"Turns: {rubric.num_turns}, "
        f"Prompt tokens: {rubric.prompt_tokens}, "
        f"Completion tokens: {rubric.completion_tokens}"
    )
    
    return traj


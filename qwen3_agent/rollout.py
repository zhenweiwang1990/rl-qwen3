"""Rollout logic for email agent evaluation."""

import art
from typing import List, Any, Optional, Union
from qwen3_agent.data.types import SyntheticQuery
from art import Trajectory
from art.utils import limit_concurrency
from art.utils.litellm import convert_litellm_choice_to_openai
from litellm import acompletion
import litellm
from qwen3_agent.tools import search_emails, read_email
from langchain_core.utils.function_calling import convert_to_openai_tool
from litellm.caching.caching import LiteLLMCacheType, Cache
import json
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from litellm.types.utils import Choices, ModelResponse, Message
from dataclasses import asdict, dataclass
import os
from datetime import datetime
from qwen3_agent.config import PolicyConfig
import textwrap
from tenacity import retry, stop_after_attempt
import logging

# Type alias for models - supports both ART models and local models
Model = Union[art.Model, art.TrainableModel]

litellm.cache = Cache(type=LiteLLMCacheType.DISK)

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Prepare tools for OpenAI function calling
search_tool = convert_to_openai_tool(search_emails)
del search_tool["function"]["parameters"]["properties"]["inbox"]
search_tool["function"]["parameters"]["required"].remove("inbox")


@dataclass
class EvaluationRubric:
    """Rubric for evaluating agent performance."""
    
    answer_correct: bool = False
    sources_correct: bool = False
    num_turns: int = 0
    attempted_answer: bool = False
    ever_found_right_email: bool = False
    ever_read_right_email: bool = False
    cant_parse_tool_call: bool = False
    bad_tool_call_name: bool = False
    bad_tool_call_args: bool = False
    ran_out_of_turns: bool = False
    returned_i_dont_know: bool = False
    num_sources: int = 0
    ever_tried_to_read_invalid_email: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def to_metrics(self) -> dict[str, float | int]:
        """Convert rubric to metrics dictionary."""
        return {k: int(v) for k, v in asdict(self).items()}


def calculate_reward(
    policy_config: PolicyConfig, rubric: EvaluationRubric, traj: Trajectory
) -> float:
    """Calculate reward based on rubric. Returns reward value between -2 and 2.
    
    Args:
        policy_config: Policy configuration
        rubric: Evaluation rubric with performance metrics
        traj: Trajectory object
    """
    # Simple reward function: 1 for correct, 0 otherwise
    if policy_config.stupid_simple_reward_fn:
        return float(rubric.answer_correct)

    # Complex reward function with partial credit
    partial_rewards = 0
    partial_rewards += 0.1 if rubric.ever_found_right_email else 0
    partial_rewards += 0.1 if rubric.ever_read_right_email else 0
    partial_rewards += 0.1 if not rubric.ever_tried_to_read_invalid_email else 0
    partial_rewards += 0.1 if rubric.sources_correct else 0

    # Formatting errors: -2 to -1
    if rubric.cant_parse_tool_call:
        return -2 + partial_rewards

    if rubric.bad_tool_call_name:
        return -1.9 + partial_rewards

    if rubric.bad_tool_call_args:
        return -1.8 + partial_rewards

    # Wrong answer: -1 to 0
    if rubric.attempted_answer and not rubric.answer_correct:
        return -1 + partial_rewards

    # No answer: 0 to 1
    if rubric.returned_i_dont_know or rubric.ran_out_of_turns:
        return 0 + partial_rewards

    # Correct answer: 1 to 2
    if rubric.answer_correct:
        reward = 1
        reward += 0.3 if rubric.sources_correct else 0
        reward += 0.1 / rubric.num_sources if rubric.num_sources > 0 else 0
        reward += 0.1 * (1 - rubric.num_turns / policy_config.max_turns)
        return reward

    traj.logs.append(f"Rubric: {rubric}")
    traj.logs.append("Rubric not handled properly")
    raise ValueError("Rubric is not handled properly")


def tool_response(response: Any, message: Message) -> ChatCompletionMessageParam:
    """Generate a response for a tool call. Returns a message that can be added to the conversation.
    
    Args:
        response: The response from the tool
        message: The message being responded to
    """
    if message.tool_calls:
        return {
            "role": "tool",
            "tool_call_id": message.tool_calls[0].id,
            "content": json.dumps(response),
        }
    else:
        return {
            "role": "user",
            "content": json.dumps(response),
        }


def return_final_answer(answer: str, sources: List[str] | None) -> str:
    """Return final answer to user's query. Returns the answer string.
    
    Args:
        answer: The answer text
        sources: List of relevant message IDs
    """
    ...


tools: list[ChatCompletionToolParam] = [
    search_tool,
    convert_to_openai_tool(read_email),
    convert_to_openai_tool(return_final_answer),
]  # type: ignore


@retry(stop=stop_after_attempt(3))
async def determine_if_answer_is_correct(answer: str, query: SyntheticQuery, verbose: bool = False) -> bool:
    """Use GPT-4o to determine if the answer is correct. Returns True if answer is semantically correct, False otherwise.
    
    Args:
        answer: The answer provided by the agent
        query: The synthetic query with ground truth
        verbose: Whether to print detailed judge logs
    """
    system_prompt = (
        "You will be given a question and two different answers to the question: "
        "the correct answer and the answer given by an AI. Your job is to determine "
        "if the answer given by the AI is correct. Return True if the answer is "
        "semantically similar to the correct answer, and False otherwise. "
        "Return only the word True or False, no other text."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Question: {query.question}\n"
                f"Correct answer: {query.answer}\n"
                f"AI answer: {answer}"
            ),
        },
    ]

    if verbose:
        print("\n" + "="*60)
        print("JUDGE EVALUATION")
        print("="*60)
        print(f"Question: {query.question}")
        print(f"\nGround Truth: {query.answer}")
        print(f"\nAgent Answer: {answer}")
        print("\nCalling GPT-4o judge...")

    # Use OpenAI GPT-4o for evaluation
    response = await acompletion(
        model="openai/gpt-4o",
        messages=messages,
        temperature=0,
        caching=True,
        max_tokens=2,
    )

    judge_result = response.choices[0].message.content.strip().lower().startswith("t")  # type: ignore
    
    if verbose:
        print(f"\nJudge Decision: {'✓ CORRECT' if judge_result else '✗ INCORRECT'}")
        print(f"Judge Response: {response.choices[0].message.content.strip()}")
        print("="*60)
    
    logger.info(
        f"Judge evaluation - Question: {query.question[:50]}..., "
        f"Ground truth: {query.answer[:50]}..., "
        f"Agent answer: {answer[:50]}..., "
        f"Result: {judge_result}"
    )

    return judge_result


@limit_concurrency(int(os.environ.get("ROLLOUT_CONCURRENCY", "10")), derive_key=lambda model, scenario, **kwargs: model.name)
async def rollout(
    model: Model,
    scenario: SyntheticQuery,
) -> Trajectory:
    """Execute a single rollout of the agent on a scenario. Returns Trajectory object with messages, reward, and metrics.
    
    Args:
        model: The model to use for rollout
        scenario: The scenario to evaluate
    """
    rollout_start_time = datetime.now()
    rubric = EvaluationRubric()
    traj = Trajectory(
        messages_and_choices=[],
        reward=0,
        metadata={"email_inbox": scenario.inbox_address, "scenario_id": scenario.id},
    )
    assert isinstance(model.config, PolicyConfig)

    # Construct system prompt
    system_prompt = textwrap.dedent(f"""\
        You are an email search agent. You are given a user query and a list of tools you can use to search the user's email. Use the tools to search the user's emails and find the answer to the user's query. You may take up to {model.config.max_turns} turns to find the answer, so if your first search doesn't find the answer, you can try with different keywords.

        User's email address is {scenario.inbox_address}
        Today's date is {scenario.query_date}
    """)

    if model.config.use_tools:
        traj.tools = tools
    else:
        system_prompt += textwrap.dedent(f"""\
            
            Here are the tools you can use:
            {tools}
            
            Respond with a valid JSON object with the following fields:
            - tool_name: (str) the name of the tool to use
            - tool_args: (JSON) the arguments to pass to the tool

            For example, to read a specific email, you should respond with:
            {{
                "tool_name": "read_email",
                "tool_args": {{
                    "message_id": "<12635597.1075855702772.JavaMail.evans@thyme>"
                }}
            }}
        """)

    traj.messages_and_choices = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": scenario.question},
    ]
    llm_response: ModelResponse | None = None

    # Main agent loop
    while True:
        rubric.num_turns += 1

        if rubric.num_turns > model.config.max_turns:
            rubric.ran_out_of_turns = True
            break

        # Verbose logging
        if model.config.verbose:
            print(f"\n--- Turn {rubric.num_turns}/{model.config.max_turns} ---")

        # Determine model name for LiteLLM
        litellm_model_name = model.config.litellm_model_name
        if litellm_model_name is None:
            # Check if using art.TrainableModel with base_url attribute
            base_url = getattr(model, 'inference_base_url', None) or getattr(model, 'base_url', None)
            if base_url:
                # For ART models, use hosted_vllm format
                if hasattr(model, 'trainable') and model.trainable:
                    litellm_model_name = f"hosted_vllm/{model.name}"
                else:
                    litellm_model_name = f"openai/{model.get_inference_name()}"
            else:
                # Fallback to hosted_vllm
                litellm_model_name = f"hosted_vllm/{model.name}"

        # Check if model is trainable
        is_trainable = getattr(model, 'trainable', False)
        
        # Get base_url and api_key - support both ART Model and local Model
        base_url = getattr(model, 'inference_base_url', None) or getattr(model, 'base_url', None)
        api_key = getattr(model, 'inference_api_key', None) or getattr(model, 'api_key', None)

        # Build completion kwargs
        completion_kwargs = {
            "model": litellm_model_name,
            "base_url": base_url,
            "messages": traj.messages(),
            "caching": not is_trainable,
            "api_key": api_key,
            "max_completion_tokens": model.config.max_tokens,
            # Make timeout and retries configurable for faster failure visibility
            "timeout": int(os.environ.get("LITELLM_TIMEOUT", "60")),
            "max_retries": int(os.environ.get("LITELLM_MAX_RETRIES", "0")),
        }
        
        # Add tools if configured
        if model.config.use_tools:
            completion_kwargs["tools"] = tools
            # For trainable models, don't force tool_choice
            # For non-trainable comparison models, optionally enforce tool usage
            if not is_trainable:
                completion_kwargs["tool_choice"] = None  # Let model decide
        
        # Make the API call with robust error logging
        try:
            llm_response = await acompletion(**completion_kwargs)  # type: ignore
        except Exception as e:
            # Attach detailed context for debugging
            safe_kwargs = {
                k: ("***" if k == "api_key" else v)
                for k, v in completion_kwargs.items()
            }
            traj.logs.append(f"LiteLLM error: {getattr(e, 'message', repr(e))}")
            traj.logs.append(f"Completion kwargs: {json.dumps(safe_kwargs, default=str)[:2000]}")
            if model.config.verbose:
                print("LiteLLM error:", repr(e))
                print("Request:", safe_kwargs)
            # Mark as bad call and break to avoid infinite retries upstream
            rubric.bad_tool_call_args = True
            break

        assert isinstance(llm_response, ModelResponse)
        rubric.prompt_tokens += llm_response.usage.prompt_tokens  # type: ignore
        rubric.completion_tokens += llm_response.usage.completion_tokens  # type: ignore
        choice = llm_response.choices[0]  # type: ignore
        assert isinstance(choice, Choices)

        # Handle multiple tool calls (only use the first one)
        if choice.message.tool_calls is not None and len(choice.message.tool_calls) > 1:
            choice.message.tool_calls = choice.message.tool_calls[:1]
        
        if is_trainable:
            traj.messages_and_choices.append(convert_litellm_choice_to_openai(choice))
        else:
            traj.messages_and_choices.append(choice.message.to_dict())  # type: ignore

        # Parse tool call
        if model.config.use_tools:
            tool_call = (
                choice.message.tool_calls[0].get("function")
                if choice.message.tool_calls
                else None
            )
            if tool_call is None:
                rubric.bad_tool_call_args = True
                traj.logs.append(f"Turn {rubric.num_turns}: No tool call found in model response")
                if model.config.verbose:
                    print(f"ERROR: No tool call found. Response: {choice.message.content}")
                break
            tool_name = tool_call["name"]
            try:
                tool_args = json.loads(tool_call["arguments"])
                assert isinstance(tool_args, dict)
            except Exception as e:
                rubric.bad_tool_call_args = True
                traj.logs.append(f"Turn {rubric.num_turns}: Failed to parse tool arguments: {e}")
                if model.config.verbose:
                    print(f"ERROR: Failed to parse tool arguments: {e}")
                break
        else:
            raw_content = choice.message.content
            if raw_content is None:
                rubric.cant_parse_tool_call = True
                traj.logs.append(f"Turn {rubric.num_turns}: No content in model response")
                if model.config.verbose:
                    print(f"ERROR: No content in model response")
                break
            start_index = raw_content.find("{")
            end_index = raw_content.rfind("}")
            if not (start_index != -1 and end_index != -1 and start_index < end_index):
                rubric.cant_parse_tool_call = True
                traj.logs.append(f"Turn {rubric.num_turns}: No valid JSON found in response: {raw_content[:200]}")
                if model.config.verbose:
                    print(f"ERROR: No valid JSON found in response: {raw_content[:200]}")
                break
            json_str = raw_content[start_index : end_index + 1]

            try:
                tool_call = json.loads(json_str)
            except Exception as e:
                traj.logs.append(f"Error parsing tool call: {e}")
                rubric.cant_parse_tool_call = True
                break

            if "tool_args" not in tool_call:
                rubric.bad_tool_call_args = True
                traj.logs.append(f"Turn {rubric.num_turns}: Tool call missing tool_args: {tool_call}")
                if model.config.verbose:
                    print(f"ERROR: Tool call missing tool_args: {tool_call}")
                break
            tool_name = tool_call.get("tool_name")
            tool_args = tool_call.get("tool_args")

        if model.config.verbose:
            print(f"Tool: {tool_name}")
            print(f"Args: {tool_args}")

        # Execute tool
        match tool_name:
            case "search_emails":
                try:
                    search_results = search_emails(
                        **tool_args,
                        inbox=scenario.inbox_address,
                    )
                    traj.messages_and_choices.append(
                        tool_response(
                            [asdict(r) for r in search_results],
                            choice.message,
                        )
                    )
                    for r in search_results:
                        if r.message_id == scenario.message_ids[0]:
                            rubric.ever_found_right_email = True
                    
                    if model.config.verbose:
                        print(f"Found {len(search_results)} emails")
                except Exception as e:
                    rubric.bad_tool_call_args = True
                    error_msg = f"Error searching emails: {e}"
                    traj.logs.append(error_msg)
                    if model.config.verbose:
                        print(f"ERROR: {error_msg}")
                        import traceback
                        traceback.print_exc()
                    break
            case "read_email":
                message_id_to_read = tool_args.get("message_id")
                if not isinstance(message_id_to_read, str):
                    rubric.bad_tool_call_args = True
                    traj.logs.append(f"Turn {rubric.num_turns}: Invalid message_id type: {type(message_id_to_read)}")
                    if model.config.verbose:
                        print(f"ERROR: Invalid message_id type: {type(message_id_to_read)}")
                    break
                if message_id_to_read == scenario.message_ids[0]:
                    rubric.ever_read_right_email = True
                email_content = read_email(message_id_to_read)
                if email_content is None:
                    traj.messages_and_choices.append(
                        tool_response({"error": "Email not found"}, choice.message)
                    )
                    rubric.ever_tried_to_read_invalid_email = True
                else:
                    traj.messages_and_choices.append(
                        tool_response(email_content.model_dump(), choice.message)
                    )
                    if model.config.verbose:
                        print(f"Read email: {email_content.subject}")
            case "return_final_answer":
                final_answer = tool_args.get("answer")
                final_sources = tool_args.get("sources")

                if (
                    final_answer is None
                    or final_sources is None
                    or not isinstance(final_sources, list)
                ):
                    rubric.bad_tool_call_args = True
                    traj.logs.append(f"Turn {rubric.num_turns}: Invalid return_final_answer args - answer: {type(final_answer)}, sources: {type(final_sources)}")
                    if model.config.verbose:
                        print(f"ERROR: Invalid return_final_answer args - answer: {type(final_answer)}, sources: {type(final_sources)}")
                    break

                rubric.num_sources = len(final_sources)
                
                if final_answer == "I don't know":
                    rubric.returned_i_dont_know = True
                    if model.config.verbose:
                        print("Agent returned: I don't know")
                else:
                    rubric.attempted_answer = True
                    rubric.answer_correct = await determine_if_answer_is_correct(
                        final_answer, scenario, verbose=model.config.verbose
                    )
                    rubric.sources_correct = scenario.message_ids[0] in final_sources
                    
                    if model.config.verbose:
                        print(f"\n{'='*60}")
                        print("FINAL RESULTS")
                        print(f"{'='*60}")
                        print(f"Answer correct: {'✓ YES' if rubric.answer_correct else '✗ NO'}")
                        print(f"Sources correct: {'✓ YES' if rubric.sources_correct else '✗ NO'}")
                        print(f"{'='*60}")
                break
            case _:
                rubric.bad_tool_call_name = True
                traj.logs.append(f"Turn {rubric.num_turns}: Unknown tool name: {tool_name}")
                if model.config.verbose:
                    print(f"ERROR: Unknown tool name: {tool_name}")
                break

    # Calculate final reward and metrics
    reward = calculate_reward(model.config, rubric, traj)
    traj.reward = reward
    traj.metrics = rubric.to_metrics()
    rollout_end_time = datetime.now()
    duration_seconds = (rollout_end_time - rollout_start_time).total_seconds()
    traj.metrics["duration"] = duration_seconds

    if model.config.verbose:
        print(f"\nRollout completed in {duration_seconds:.2f}s")
        print(f"Reward: {reward:.2f}")
        print(f"Answer correct: {rubric.answer_correct}")

    # Log rollout summary
    logger.info(
        f"Rollout completed - Model: {model.name}, "
        f"Duration: {duration_seconds:.2f}s, "
        f"Reward: {reward:.2f}, "
        f"Answer correct: {rubric.answer_correct}, "
        f"Turns: {rubric.num_turns}, "
        f"Prompt tokens: {rubric.prompt_tokens}, "
        f"Completion tokens: {rubric.completion_tokens}"
    )
    
    # Log detailed metrics if verbose
    if model.config.verbose:
        logger.debug(f"Trajectory metrics: {traj.metrics}")
        logger.debug(f"Trajectory metadata: {traj.metadata}")

    return traj


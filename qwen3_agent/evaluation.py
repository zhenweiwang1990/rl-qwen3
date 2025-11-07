"""Shared evaluation utilities for both SB3 and ART training."""

from dataclasses import dataclass, asdict
from typing import Dict
from qwen3_agent.data.types import SyntheticQuery
from litellm import acompletion
from tenacity import retry, stop_after_attempt
import logging

logger = logging.getLogger(__name__)


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

    def to_metrics(self) -> Dict[str, float | int]:
        """Convert rubric to metrics dictionary."""
        return {k: int(v) for k, v in asdict(self).items()}


@retry(stop=stop_after_attempt(3))
async def determine_if_answer_is_correct(
    answer: str, 
    query: SyntheticQuery, 
    verbose: bool = False
) -> bool:
    """Use GPT-4o to determine if the answer is correct.
    
    Args:
        answer: The answer provided by the agent
        query: The synthetic query with ground truth
        verbose: Whether to print detailed judge logs
    
    Returns:
        True if answer is semantically correct, False otherwise
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


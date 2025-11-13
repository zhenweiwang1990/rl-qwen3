"""Simple Math Agent Example.

This example demonstrates how to create a custom agent using the generic framework.
The agent can solve simple math problems using basic operations.
"""

import asyncio
from typing import Any, Dict, List
from dataclasses import dataclass

from qwen3_agent.core.framework import (
    BaseTask,
    BaseAgent,
    BaseEvaluator,
    BaseRubric,
    ActionResult,
    LLMInference,
    generic_rollout,
)


# ==================== 1. Define Task ====================

class MathTask(BaseTask):
    """A simple math problem task."""
    
    problem: str  # e.g., "What is 15 + 27?"
    answer: float  # Ground truth answer
    
    def get_query(self) -> str:
        return self.problem
    
    def get_ground_truth(self) -> Any:
        return self.answer


# ==================== 2. Implement Agent ====================

class MathAgent(BaseAgent):
    """A simple agent that can perform basic math operations."""
    
    def get_system_prompt(self, task: BaseTask) -> str:
        return (
            "You are a math problem solver. Use the available tools to solve the problem.\n"
            "Tools:\n"
            "- add(a, b): Add two numbers\n"
            "- subtract(a, b): Subtract b from a\n"
            "- multiply(a, b): Multiply two numbers\n"
            "- divide(a, b): Divide a by b\n"
            "- answer(result): Submit your final answer\n"
            "\nSolve the problem step by step."
        )
    
    def get_tools_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "add",
                    "description": "Add two numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number", "description": "First number"},
                            "b": {"type": "number", "description": "Second number"},
                        },
                        "required": ["a", "b"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "subtract",
                    "description": "Subtract b from a",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"},
                        },
                        "required": ["a", "b"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "multiply",
                    "description": "Multiply two numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"},
                        },
                        "required": ["a", "b"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "divide",
                    "description": "Divide a by b",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"},
                        },
                        "required": ["a", "b"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "answer",
                    "description": "Submit final answer",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "result": {"type": "number", "description": "Final answer"},
                        },
                        "required": ["result"],
                    },
                },
            },
        ]
    
    def execute_action(
        self, tool_name: str, tool_args: Dict[str, Any], task: BaseTask
    ) -> ActionResult:
        """Execute math operations."""
        try:
            if tool_name == "add":
                result = tool_args["a"] + tool_args["b"]
                return ActionResult(success=True, data={"result": result})
            
            elif tool_name == "subtract":
                result = tool_args["a"] - tool_args["b"]
                return ActionResult(success=True, data={"result": result})
            
            elif tool_name == "multiply":
                result = tool_args["a"] * tool_args["b"]
                return ActionResult(success=True, data={"result": result})
            
            elif tool_name == "divide":
                if tool_args["b"] == 0:
                    return ActionResult(success=False, error="Division by zero")
                result = tool_args["a"] / tool_args["b"]
                return ActionResult(success=True, data={"result": result})
            
            elif tool_name == "answer":
                return ActionResult(success=True, data={"answer": tool_args["result"]})
            
            else:
                return ActionResult(success=False, error=f"Unknown tool: {tool_name}")
        
        except Exception as e:
            return ActionResult(success=False, error=str(e))
    
    def is_terminal_action(self, tool_name: str) -> bool:
        return tool_name == "answer"


# ==================== 3. Implement Evaluator ====================

@dataclass
class MathRubric(BaseRubric):
    """Rubric for math problem evaluation."""
    answer_correct: bool = False
    num_operations: int = 0


class MathEvaluator(BaseEvaluator[MathRubric]):
    """Evaluator for math tasks."""
    
    def __init__(self, tolerance: float = 0.01):
        self.tolerance = tolerance
    
    def create_rubric(self) -> MathRubric:
        return MathRubric()
    
    async def evaluate_trajectory(
        self, trajectory: Any, task: BaseTask, rubric: MathRubric
    ) -> float:
        """Calculate reward based on correctness."""
        if rubric.answer_correct:
            # Bonus for using fewer operations
            efficiency_bonus = max(0, (10 - rubric.num_operations) / 10)
            return 1.0 + 0.5 * efficiency_bonus
        else:
            return -1.0
    
    def on_action_executed(
        self,
        rubric: MathRubric,
        tool_name: str,
        tool_args: Dict[str, Any],
        result: Any,
        task: BaseTask,
    ) -> None:
        """Track operations and check answer."""
        assert isinstance(task, MathTask)
        
        # Count operations
        if tool_name in ("add", "subtract", "multiply", "divide"):
            rubric.num_operations += 1
        
        # Check final answer
        if tool_name == "answer" and isinstance(result, dict):
            agent_answer = result.get("answer")
            if agent_answer is not None:
                ground_truth = task.get_ground_truth()
                rubric.answer_correct = abs(agent_answer - ground_truth) < self.tolerance


# ==================== 4. Example Usage ====================

async def main():
    """Run example."""
    print("\n" + "="*80)
    print("Simple Math Agent Example")
    print("="*80 + "\n")
    
    # Create task
    task = MathTask(
        id="1",
        problem="What is 15 + 27 multiplied by 2?",
        answer=84.0,  # (15 + 27) * 2 = 84
    )
    
    print(f"Problem: {task.problem}")
    print(f"Expected answer: {task.answer}")
    print()
    
    # Create agent and evaluator
    agent = MathAgent()
    evaluator = MathEvaluator(tolerance=0.01)
    
    # Create LLM inference (using GPT-4o-mini for demo)
    llm = LLMInference("openai/gpt-4o-mini", {})
    
    # Execute rollout
    print("Solving problem...")
    print("-"*80)
    
    trajectory = await generic_rollout(
        llm=llm,
        task=task,
        agent=agent,
        evaluator=evaluator,
        max_turns=10,
        use_native_tools=True,
        verbose=True,
    )
    
    print("-"*80)
    print()
    
    # Display results
    print("Results:")
    print(f"  Reward: {trajectory.reward:.2f}")
    print(f"  Answer correct: {trajectory.metrics.get('answer_correct', False)}")
    print(f"  Operations used: {trajectory.metrics.get('num_operations', 0)}")
    print(f"  Turns: {trajectory.metrics.get('num_turns', 0)}")
    print()
    
    print("="*80)
    print("✓ Example completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())


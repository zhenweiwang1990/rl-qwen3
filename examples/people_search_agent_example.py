#!/usr/bin/env python3
"""Example usage of People Search Agent."""

import asyncio
from qwen3_agent.agents.people_search_agent import (
    PeopleSearchAgent,
    PeopleSearchEvaluator,
    PeopleSearchTask,
)
from qwen3_agent.agents.people_search_agent.data_loader import load_default_benchmark


def example_basic_task():
    """Example: Create a basic people search task."""
    print("=" * 80)
    print("Example 1: Basic Task Creation")
    print("=" * 80)
    
    task = PeopleSearchTask(
        id="example-1",
        query="AI researchers with experience in machine learning and NLP",
        expected_profiles=[
            "john-smith-ai",
            "jane-doe-ml",
            "bob-johnson-nlp",
        ],
        batch="example",
    )
    
    print(f"Task ID: {task.id}")
    print(f"Query: {task.query}")
    print(f"Expected profiles: {task.expected_profiles}")
    print(f"Ground truth: {task.get_ground_truth()}")
    print()


def example_load_benchmark():
    """Example: Load benchmark tasks."""
    print("=" * 80)
    print("Example 2: Load Benchmark Tasks")
    print("=" * 80)
    
    tasks = load_default_benchmark()
    
    print(f"Total tasks: {len(tasks)}")
    print()
    
    # Show first 3 tasks
    for i, task in enumerate(tasks[:3], 1):
        print(f"Task {i}:")
        print(f"  ID: {task.id}")
        print(f"  Query: {task.query[:80]}...")
        print(f"  Expected profiles: {len(task.expected_profiles)}")
        print(f"  Batch: {task.batch}")
        print()


def example_agent_setup():
    """Example: Set up agent and evaluator."""
    print("=" * 80)
    print("Example 3: Agent and Evaluator Setup")
    print("=" * 80)
    
    # Create agent
    agent = PeopleSearchAgent()
    print(f"Agent created: {agent.__class__.__name__}")
    
    # Create evaluator
    evaluator = PeopleSearchEvaluator(
        max_turns=10,
        verbose=True,
        simple_reward=False,
    )
    print(f"Evaluator created: {evaluator.__class__.__name__}")
    print(f"  - Max turns: {evaluator.max_turns}")
    print(f"  - Verbose: {evaluator.verbose}")
    print(f"  - Simple reward: {evaluator.simple_reward}")
    print()
    
    # Associate evaluator with agent
    agent.evaluator = evaluator
    
    # Get tools
    tools = agent.get_tools_schema()
    print(f"Available tools: {len(tools)}")
    for tool in tools:
        print(f"  - {tool['function']['name']}: {tool['function']['description'][:60]}...")
    print()
    
    # Get system prompt for a sample task
    task = PeopleSearchTask(
        id="sample",
        query="AI researchers",
        expected_profiles=["sample-profile"],
    )
    
    prompt = agent.get_system_prompt(task)
    print("System Prompt Preview:")
    print(prompt[:300] + "...")
    print()


def example_tool_usage():
    """Example: Direct tool usage."""
    print("=" * 80)
    print("Example 4: Direct Tool Usage")
    print("=" * 80)
    
    from qwen3_agent.agents.people_search_agent import (
        search_profiles,
        read_profile,
    )
    
    # Search for profiles
    print("Searching for AI researchers...")
    results = search_profiles(
        keywords=["AI", "machine learning"],
        max_results=3,
    )
    
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.linkedin_handle}")
        print(f"   Snippet: {result.snippet[:100]}...")
        print()
    
    # Read a profile
    if results:
        handle = results[0].linkedin_handle
        print(f"Reading profile: {handle}")
        profile = read_profile(handle)
        
        if profile:
            print(f"  Name: {profile.name}")
            print(f"  About: {profile.about[:150] if profile.about else 'N/A'}...")
            print()


def example_reward_calculation():
    """Example: Understand reward calculation."""
    print("=" * 80)
    print("Example 5: Reward Calculation")
    print("=" * 80)
    
    from qwen3_agent.agents.people_search_agent import PeopleSearchRubric
    
    evaluator = PeopleSearchEvaluator(simple_reward=False)
    
    # Example 1: Perfect match
    rubric1 = PeopleSearchRubric()
    rubric1.expected_profiles = {"alice", "bob", "charlie"}
    rubric1.answer_profiles = {"alice", "bob", "charlie"}
    rubric1.attempted_answer = True
    rubric1.answer_perfect_match = True
    rubric1.answer_correct = True
    rubric1.profiles_found_in_search = {"alice", "bob", "charlie"}
    rubric1.profiles_read_correct = {"alice", "bob"}
    
    reward1 = evaluator._calculate_reward(rubric1)
    print(f"Example 1 - Perfect Match:")
    print(f"  Expected: {rubric1.expected_profiles}")
    print(f"  Returned: {rubric1.answer_profiles}")
    print(f"  Reward: {reward1:.2f}")
    print()
    
    # Example 2: Partial match (70% overlap)
    rubric2 = PeopleSearchRubric()
    rubric2.expected_profiles = {"alice", "bob", "charlie", "dave"}
    rubric2.answer_profiles = {"alice", "bob", "charlie", "eve"}  # 3/4 = 75%
    rubric2.attempted_answer = True
    rubric2.answer_correct = True
    rubric2.profiles_found_in_search = {"alice", "bob", "charlie"}
    rubric2.profiles_read_correct = {"alice", "bob"}
    
    reward2 = evaluator._calculate_reward(rubric2)
    print(f"Example 2 - Partial Match (75% overlap):")
    print(f"  Expected: {rubric2.expected_profiles}")
    print(f"  Returned: {rubric2.answer_profiles}")
    print(f"  Reward: {reward2:.2f}")
    print()
    
    # Example 3: Wrong answer
    rubric3 = PeopleSearchRubric()
    rubric3.expected_profiles = {"alice", "bob", "charlie"}
    rubric3.answer_profiles = {"dave", "eve", "frank"}  # 0% overlap
    rubric3.attempted_answer = True
    rubric3.answer_correct = False
    rubric3.profiles_found_in_search = {"alice"}
    
    reward3 = evaluator._calculate_reward(rubric3)
    print(f"Example 3 - Wrong Answer (0% overlap):")
    print(f"  Expected: {rubric3.expected_profiles}")
    print(f"  Returned: {rubric3.answer_profiles}")
    print(f"  Reward: {reward3:.2f}")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("PEOPLE SEARCH AGENT EXAMPLES")
    print("=" * 80 + "\n")
    
    example_basic_task()
    example_load_benchmark()
    example_agent_setup()
    example_tool_usage()
    example_reward_calculation()
    
    print("=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()


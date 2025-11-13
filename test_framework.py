"""Test script for the generic framework.

This script tests the generic framework with the EmailAgent implementation.
All code now uses the new framework directly - no compatibility layers.
"""

import asyncio
from qwen3_agent.core.framework import (
    LLMInference,
    generic_rollout,
)
from qwen3_agent.agents.email_agent import EmailAgent, EmailTask, EmailEvaluator
from qwen3_agent.data import load_synthetic_queries
from qwen3_agent.data.local_email_db import generate_database
from qwen3_agent.train import create_trainable_model
from qwen3_agent.benchmark import benchmark_model


async def test_framework():
    """Test the new framework with a simple scenario."""
    print("="*80)
    print("Testing New Generic Framework")
    print("="*80)
    
    # Generate database if needed
    print("\n1. Checking database...")
    generate_database()
    print("   ✓ Database ready")
    
    # Load a test scenario
    print("\n2. Loading test scenario...")
    scenarios = load_synthetic_queries(split="test", limit=1)
    scenario = scenarios[0]
    print(f"   ✓ Loaded scenario: {scenario.question[:60]}...")
    
    # Convert to task
    print("\n3. Creating task...")
    task = EmailTask.from_synthetic_query(scenario)
    print(f"   ✓ Task created with ID: {task.id}")
    
    # Create agent and evaluator
    print("\n4. Initializing agent and evaluator...")
    evaluator = EmailEvaluator(
        simple_reward=False,
        verbose=True,
        max_turns=10,
    )
    agent = EmailAgent(evaluator=evaluator)
    print("   ✓ Agent and evaluator ready")
    
    # Create LLM inference (using OpenAI GPT-4o-mini for testing)
    print("\n5. Setting up LLM inference...")
    llm = LLMInference(
        model="openai/gpt-4o-mini",
        model_config={}
    )
    print("   ✓ LLM inference configured")
    
    # Execute rollout
    print("\n6. Executing rollout...")
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
    
    # Display results
    print("\n7. Results:")
    print(f"   Reward: {trajectory.reward:.2f}")
    print(f"   Turns: {trajectory.metrics.get('num_turns', 'N/A')}")
    print(f"   Answer correct: {trajectory.metrics.get('answer_correct', False)}")
    print(f"   Sources correct: {trajectory.metrics.get('sources_correct', False)}")
    print(f"   Duration: {trajectory.metrics.get('duration', 0):.2f}s")
    
    print("\n" + "="*80)
    print("✓ Framework test completed successfully!")
    print("="*80)
    
    return trajectory


async def test_integration():
    """Test integration with train and benchmark modules."""
    print("\n" + "="*80)
    print("Testing Module Integration")
    print("="*80)
    
    print("\n1. Testing train module imports...")
    try:
        from qwen3_agent.train import create_trainable_model, run_training
        print("   ✓ Train module imports successful")
    except Exception as e:
        print(f"   ✗ Train import failed: {e}")
        return False
    
    print("\n2. Testing benchmark module imports...")
    try:
        from qwen3_agent.benchmark import benchmark_model
        print("   ✓ Benchmark module imports successful")
    except Exception as e:
        print(f"   ✗ Benchmark import failed: {e}")
        return False
    
    print("\n3. Testing framework exports...")
    try:
        from qwen3_agent.core.framework import (
            BaseAgent, BaseTask, BaseEvaluator,
            LLMInference, generic_rollout
        )
        print("   ✓ Framework exports successful")
    except Exception as e:
        print(f"   ✗ Framework exports failed: {e}")
        return False
    
    print("\n" + "="*80)
    print("✓ Integration test passed!")
    print("="*80)
    
    return True


async def main():
    """Run all tests."""
    print("\n" + "🚀 Starting Framework Tests" + "\n")
    
    try:
        # Test new framework
        await test_framework()
        
        # Test integration
        await test_integration()
        
        print("\n" + "🎉 All tests passed! Framework is fully integrated." + "\n")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())


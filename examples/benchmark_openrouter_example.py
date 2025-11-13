#!/usr/bin/env python3
"""Example: Benchmark People Search Agent with OpenRouter."""

import asyncio
import os
import sys

# Make sure we can import from the project
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen3_agent.agents.people_search_agent.benchmark import benchmark_agent


async def main():
    """Run benchmark with OpenRouter."""
    
    # Check for API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ Error: OPENROUTER_API_KEY environment variable not set")
        print("\nPlease set your OpenRouter API key:")
        print('  export OPENROUTER_API_KEY="sk-or-v1-..."')
        print("\nGet your API key at: https://openrouter.ai/")
        return
    
    print("🚀 Running People Search Agent Benchmark with OpenRouter")
    print("=" * 80)
    
    # Run benchmark
    df = await benchmark_agent(
        provider="openrouter",
        model="qwen3-30b-a3b-instruct-2507",  # Or use: qwen/qwen3-30b-a3b-instruct-2507
        num_samples=5,  # Start with just 5 samples for testing
        max_turns=20,
        debug=True,  # Enable debug mode
        debug_count=2,  # Show details for first 2 tasks
        output_dir="./benchmark_results",
    )
    
    print("\n✅ Benchmark complete!")
    print(f"Evaluated {len(df)} tasks")
    print(f"Average reward: {df['reward'].mean():.3f}")
    print("\nResults saved to: benchmark_results/")


if __name__ == "__main__":
    asyncio.run(main())


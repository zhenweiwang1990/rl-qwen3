#!/usr/bin/env python3
"""Example: People Search Agent with MCP Tools Integration.

This example demonstrates how to use the People Search Agent with MCP tools.
The agent fetches tool schemas directly from the MCP server and executes
searches and reads through the server.

Prerequisites:
1. MCP server running at http://localhost:4111/api/mcp/profile-mcp-server/mcp
2. Environment variable PROFILE_MCP_BASE_URL (optional, defaults to above)

Usage:
    python examples/people_search_mcp_example.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qwen3_agent.agents.people_search_agent.agent import PeopleSearchAgent
from qwen3_agent.agents.people_search_agent.tasks import PeopleSearchTask
from qwen3_agent.agents.people_search_agent.evaluator import PeopleSearchEvaluator
from qwen3_agent.core.framework.rollout import generic_rollout
from qwen3_agent.core.framework.llm_inference import LLMInference


async def main():
    """Run a simple people search example."""
    
    print("=" * 60)
    print("People Search Agent - MCP Tools Example")
    print("=" * 60)
    
    # Step 1: Initialize agent (fetches MCP tool schemas)
    print("\n1. Initializing agent...")
    try:
        agent = PeopleSearchAgent()
        print("   ✓ Agent initialized successfully")
        print(f"   ✓ Loaded {len(agent.get_tools_schema())} tools from MCP server")
    except Exception as e:
        print(f"   ✗ Failed to initialize agent: {e}")
        print("\n   Make sure MCP server is running:")
        print("   - Check: curl http://localhost:4111/api/mcp/profile-mcp-server/mcp")
        print("   - Or run: python qwen3_agent/agents/people_search_agent/test_mcp.py")
        return
    
    # Step 2: Create a task
    print("\n2. Creating task...")
    task = PeopleSearchTask(
        id="example-001",
        query="Find software engineers with Python experience in San Francisco",
        expected_profiles=[
            # These would be the ground truth profiles for evaluation
            "john-doe-123",
            "jane-smith-456",
        ]
    )
    print(f"   ✓ Task created: {task.query}")
    
    # Step 3: Initialize LLM
    print("\n3. Initializing LLM...")
    llm = LLMInference(
        model="gpt-4",  # or your preferred model
        temperature=0.0,
    )
    print("   ✓ LLM initialized")
    
    # Step 4: Initialize evaluator
    print("\n4. Initializing evaluator...")
    evaluator = PeopleSearchEvaluator(verbose=True)
    agent.evaluator = evaluator
    print("   ✓ Evaluator initialized")
    
    # Step 5: Run the agent
    print("\n5. Running agent...")
    print("-" * 60)
    
    try:
        trajectory = await generic_rollout(
            llm=llm,
            task=task,
            agent=agent,
            evaluator=evaluator,
        )
        
        print("-" * 60)
        print("\n6. Results:")
        print(f"   - Turns taken: {trajectory.num_turns}")
        print(f"   - Reward: {trajectory.reward:.2f}")
        print(f"   - Success: {trajectory.success}")
        
        # Show final answer
        if trajectory.actions:
            last_action = trajectory.actions[-1]
            if last_action.tool_name == "return_final_answer":
                profiles = last_action.tool_args.get("profiles", [])
                print(f"   - Profiles found: {len(profiles)}")
                for i, profile in enumerate(profiles[:5], 1):
                    print(f"     {i}. {profile}")
                if len(profiles) > 5:
                    print(f"     ... and {len(profiles) - 5} more")
        
    except Exception as e:
        print(f"\n   ✗ Agent execution failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Example completed")
    print("=" * 60)


def test_mcp_connection():
    """Quick test to verify MCP server connectivity."""
    print("Testing MCP server connection...")
    
    try:
        from qwen3_agent.agents.people_search_agent.mcp_tools import get_mcp_client
        
        client = get_mcp_client()
        tools = client.list_tools()
        
        print(f"✓ Connected to MCP server")
        print(f"✓ Found {len(tools)} tools:")
        for tool in tools:
            print(f"  - {tool.get('name')}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to connect to MCP server: {e}")
        return False


if __name__ == "__main__":
    # First test MCP connection
    if not test_mcp_connection():
        print("\nPlease ensure MCP server is running before running this example.")
        sys.exit(1)
    
    print("\n")
    
    # Run the example
    asyncio.run(main())


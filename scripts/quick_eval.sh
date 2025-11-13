#!/bin/bash
set -e

# Quick evaluation script for testing a single scenario
# Usage: ./scripts/quick_eval.sh [model_name]
# Default: Uses Qwen3 14B via vLLM (hosted locally)

MODEL_NAME="${1:-OpenPipe/Qwen3-14B-Instruct}"

echo "=========================================="
echo "Quick Evaluation - Single Scenario"
echo "Agent Model: $MODEL_NAME"
echo "Judge Model: gpt-4o"
echo "=========================================="
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please run ./scripts/setup.sh first."
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Error: .env file not found. Please create one based on .env.example"
    exit 1
fi

# Load environment variables
set -a
source .env
set +a

# Create quick eval script
cat > /tmp/quick_eval.py << 'EOF'
import asyncio
import art
from dotenv import load_dotenv
import sys
from qwen3_agent.config import PolicyConfig
from qwen3_agent.core.framework import LLMInference, generic_rollout
from qwen3_agent.agents.email_agent import EmailAgent, EmailTask, EmailEvaluator
from qwen3_agent.data import load_synthetic_queries
import yaml
from tabulate import tabulate

load_dotenv()

async def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "OpenPipe/Qwen3-14B-Instruct"
    
    # Load one test scenario
    scenarios = load_synthetic_queries(split="test", limit=2)
    scenario = scenarios[0]
    
    print("\n" + "="*60)
    print("TEST SCENARIO")
    print("="*60)
    print(f"Question: {scenario.question}")
    print(f"Ground Truth: {scenario.answer}")
    print(f"Inbox: {scenario.inbox_address}")
    print(f"Date: {scenario.query_date}")
    print("="*60)
    
    # Determine model configuration
    import os
    
    if model_name.startswith("qwen3-"):
        # Use our trained model from local API
        api = art.LocalAPI()
        model = art.Model(
            name=model_name,
            project="qwen3_email_agent",
            config=PolicyConfig(verbose=True),
        )
        await model.register(api)
    elif "Qwen" in model_name or "qwen" in model_name:
        # Use Qwen model via local vLLM server
        vllm_base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        model = art.Model(
            name=model_name,
            project="qwen3_email_agent",
            inference_api_key="EMPTY",
            inference_base_url=vllm_base_url,
            config=PolicyConfig(
                litellm_model_name=f"openai/{model_name}",
                verbose=True,
            ),
        )
    else:
        # Use external model (e.g., gpt-4o) via OpenAI API
        model = art.Model(
            name=model_name,
            project="qwen3_email_agent",
            inference_api_key=os.getenv("OPENAI_API_KEY"),
            inference_base_url="https://api.openai.com/v1",
            config=PolicyConfig(
                litellm_model_name=f"openai/{model_name}",
                verbose=True,
            ),
        )
    
    # Create agent and evaluator
    config = model.config
    evaluator = EmailEvaluator(
        simple_reward=config.stupid_simple_reward_fn,
        verbose=True,
        max_turns=config.max_turns,
    )
    agent = EmailAgent(evaluator=evaluator)
    llm = LLMInference(model)
    
    # Run rollout
    task = EmailTask.from_synthetic_query(scenario)
    traj = await generic_rollout(
        llm=llm,
        task=task,
        agent=agent,
        evaluator=evaluator,
        max_turns=config.max_turns,
        use_native_tools=config.use_tools,
        verbose=True,
    )
    
    # Prepare summary table
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    # Key metrics table
    key_metrics = [
        ["Metric", "Value"],
        ["─" * 30, "─" * 15],
        ["Answer Correct", "✓ YES" if traj.metrics.get("answer_correct") else "✗ NO"],
        ["Sources Correct", "✓ YES" if traj.metrics.get("sources_correct") else "✗ NO"],
        ["Final Reward", f"{traj.reward:.2f}"],
        ["Number of Turns", traj.metrics.get("num_turns", 0)],
        ["Duration (seconds)", f"{traj.metrics.get('duration', 0):.2f}"],
        ["Prompt Tokens", traj.metrics.get("prompt_tokens", 0)],
        ["Completion Tokens", traj.metrics.get("completion_tokens", 0)],
    ]
    print(tabulate(key_metrics, headers="firstrow", tablefmt="simple"))
    
    # Agent behavior table
    print("\n" + "="*80)
    print("AGENT BEHAVIOR DETAILS")
    print("="*80)
    behavior_metrics = [
        ["Behavior", "Status"],
        ["─" * 40, "─" * 10],
        ["Attempted Answer", "✓" if traj.metrics.get("attempted_answer") else "✗"],
        ["Ever Found Right Email", "✓" if traj.metrics.get("ever_found_right_email") else "✗"],
        ["Ever Read Right Email", "✓" if traj.metrics.get("ever_read_right_email") else "✗"],
        ["Ran Out of Turns", "✓" if traj.metrics.get("ran_out_of_turns") else "✗"],
        ["Returned 'I Don't Know'", "✓" if traj.metrics.get("returned_i_dont_know") else "✗"],
        ["Invalid Email Read Attempt", "✓" if traj.metrics.get("ever_tried_to_read_invalid_email") else "✗"],
    ]
    print(tabulate(behavior_metrics, headers="firstrow", tablefmt="simple"))
    
    # Error tracking table
    errors = []
    if traj.metrics.get("cant_parse_tool_call"):
        errors.append("Cannot parse tool call")
    if traj.metrics.get("bad_tool_call_name"):
        errors.append("Bad tool call name")
    if traj.metrics.get("bad_tool_call_args"):
        errors.append("Bad tool call arguments")
    
    if errors:
        print("\n" + "="*80)
        print("ERRORS DETECTED")
        print("="*80)
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
    
    print("\n" + "="*80)
    print("FULL METRICS")
    print("="*80)
    all_metrics = [[k, v] for k, v in sorted(traj.metrics.items())]
    print(tabulate(all_metrics, headers=["Metric", "Value"], tablefmt="grid"))
    
    # Optional: Show trajectory details
    if os.getenv("SHOW_TRAJECTORY_DETAILS", "false").lower() == "true":
        print("\n" + "="*80)
        print("TRAJECTORY DETAILS (YAML)")
        print("="*80)
        print(yaml.dump(traj.for_logging(), default_flow_style=False))
    
    print("\n" + "="*80)

asyncio.run(main())
EOF

# Run quick eval
uv run python /tmp/quick_eval.py "$MODEL_NAME"

# Cleanup
rm /tmp/quick_eval.py

echo ""
echo "=========================================="
echo "Quick Evaluation Complete!"
echo "=========================================="


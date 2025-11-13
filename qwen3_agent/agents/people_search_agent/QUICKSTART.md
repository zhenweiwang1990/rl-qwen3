# People Search Agent - Quick Start Guide

## Prerequisites

- Python 3.12+
- UV package manager
- SQLite3 (pre-installed on most systems)

## Setup (5 minutes)

### 1. Install Dependencies

From project root:

```bash
cd /Users/zhenwei/workspace/rl-qwen3
uv sync
```

### 2. Verify Database

The database should already be created. Verify:

```bash
ls -lh qwen3_agent/agents/people_search_agent/profiles.db
```

If not found, create it:

```bash
cd qwen3_agent/agents/people_search_agent
uv run python create_profile_db.py
```

## Quick Test (2 minutes)

### Test Tools

```bash
cd qwen3_agent/agents/people_search_agent
uv run python test_tools.py
```

Expected output:
```
Testing search_profiles...
Found 5 results:
1. robinluo-18gatech
   ...
✓ All tests completed!
```

### Run Examples

```bash
cd /Users/zhenwei/workspace/rl-qwen3
uv run python examples/people_search_agent_example.py
```

Expected: 5 examples demonstrating all features

## Usage in Code

### Minimal Example

```python
from qwen3_agent.agents.people_search_agent import (
    PeopleSearchAgent,
    PeopleSearchEvaluator,
)
from qwen3_agent.agents.people_search_agent.data_loader import load_default_benchmark

# Load a task
tasks = load_default_benchmark()
task = tasks[0]

# Create agent and evaluator
agent = PeopleSearchAgent()
evaluator = PeopleSearchEvaluator(max_turns=10, verbose=True)
agent.evaluator = evaluator

# Get system prompt
prompt = agent.get_system_prompt(task)
print(prompt)

# Get tools
tools = agent.get_tools_schema()
print(f"Available tools: {[t['function']['name'] for t in tools]}")
```

### With Framework

```python
from qwen3_agent.agents.people_search_agent import (
    PeopleSearchAgent,
    PeopleSearchEvaluator,
)
from qwen3_agent.core.framework.rollout import generic_rollout
from qwen3_agent.core.framework.llm_inference import LLMInference

# Setup
agent = PeopleSearchAgent()
evaluator = PeopleSearchEvaluator(max_turns=10)
agent.evaluator = evaluator

# Create LLM
llm = LLMInference(model="qwen3-14b")

# Load task
from qwen3_agent.agents.people_search_agent.data_loader import load_default_benchmark
task = load_default_benchmark()[0]

# Run
trajectory = await generic_rollout(
    llm=llm,
    task=task,
    agent=agent,
    evaluator=evaluator,
)

print(f"Reward: {trajectory.reward}")
print(f"Turns: {evaluator.rubric.num_turns}")
```

## Quick Reference

### Tools

```python
# Search profiles
from qwen3_agent.agents.people_search_agent import search_profiles
results = search_profiles(keywords=["AI", "researcher"], max_results=10)

# Read profile
from qwen3_agent.agents.people_search_agent import read_profile
profile = read_profile("john-smith-123")
```

### Load Benchmark

```python
from qwen3_agent.agents.people_search_agent.data_loader import load_default_benchmark

# All tasks (932)
all_tasks = load_default_benchmark()

# Filter by batch
batch_1 = [t for t in all_tasks if t.batch == "1"]

# First task
task = all_tasks[0]
print(f"Query: {task.query}")
print(f"Expected: {len(task.expected_profiles)} profiles")
```

### Reward Calculation

Rewards range from **-3 to +3**:

| Score | Meaning |
|-------|---------|
| +3.0  | Perfect match (exact set) |
| +1.0 to +2.5 | Correct with good overlap |
| 0.0 to +1.0 | Partial progress or timeout |
| -2.0 to 0.0 | Wrong answer but some progress |
| -2.0 to -3.0 | Wrong answer with errors |

## Troubleshooting

### Database Error
```bash
# Recreate database
cd qwen3_agent/agents/people_search_agent
rm profiles.db
uv run python create_profile_db.py
```

### Import Error
```bash
# Make sure you're in project root
cd /Users/zhenwei/workspace/rl-qwen3

# Use module format
uv run python -m qwen3_agent.agents.people_search_agent.test_tools
```

### No Results in Search
- Check if database exists
- Try simpler keywords
- Increase max_results

## Next Steps

1. **Training**: Use with RL training loop
2. **Benchmark**: Run on all 932 tasks
3. **Evaluation**: Compare models with different configurations
4. **Optimization**: Tune reward function and max_turns

## Resources

- **README.md**: Detailed documentation
- **IMPLEMENTATION_SUMMARY.md**: Implementation details
- **examples/people_search_agent_example.py**: Full examples
- **test_tools.py**: Tool testing

## Database Stats

- **Profiles**: 93,417
- **Size**: ~450 MB
- **Fields**: 13 per profile
- **Search**: FTS5 full-text search
- **Performance**: < 10ms typical query

## Support

For issues or questions:
1. Check README.md for detailed docs
2. Run test_tools.py to verify setup
3. Run examples to see working code
4. Check IMPLEMENTATION_SUMMARY.md for design details

---

**Ready to go! 🚀**

Run the example to see it in action:
```bash
uv run python examples/people_search_agent_example.py
```


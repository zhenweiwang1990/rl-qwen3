# People Search Agent

A LinkedIn profile search agent that uses reinforcement learning to find people matching user queries.

## Overview

This agent searches and reads LinkedIn profiles to find people matching specific criteria. It uses:
- **Qwen3 14B** as the language model
- **SQLite database** with full-text search for fast profile lookup
- **Three tools**: `search_profiles`, `read_profile`, `return_final_answer`
- **Reward-based evaluation** with partial credit for progress

## Components

### Database
- **profiles.db**: SQLite database containing 93,417 LinkedIn profiles
- Full-text search index for fast keyword matching
- Fields: linkedin_handle, name, about, summary, experiences, education, skills

### Tools

1. **search_profiles(keywords, max_results=10)**
   - Search profiles using keyword matching
   - Returns list of SearchResult with linkedin_handle and snippet
   - Max 20 results per search

2. **read_profile(linkedin_handle)**
   - Read full profile details
   - Returns Profile object or None if not found

3. **return_final_answer(profiles)**
   - Return final list of matching profiles
   - Takes list of linkedin_handle strings
   - Terminates the episode

### Reward Function

Rewards range from **[-3, 3]**:

**Base Rewards:**
- Parse/tool errors: -2.8 to -2.0
- Wrong answer (no overlap): -2.0 + partial
- No answer/timeout: 0.0 + partial
- Correct answer: 1.0 + partial

**Partial Rewards (max ~2.0):**
- +0.1 per correct profile found in search (max 0.5)
- +0.1 per correct profile read (max 0.5)
- +0.1 if no invalid profile reads
- +0.1 if answer contains all expected profiles
- **Perfect match bonus: +2.0** (exact match with ground truth)
- **Overlap penalty:**
  - < 30% overlap: -0.5
  - 30-70% overlap: -0.2
  - > 70% overlap: -0.1

## Setup

### 1. Create Database

If you don't have the database yet:

```bash
cd qwen3_agent/agents/people_search_agent
uv run python create_profile_db.py
```

This will create `profiles.db` from `data/profile_detail.csv`.

### 2. Test Tools

```bash
cd qwen3_agent/agents/people_search_agent
uv run python test_tools.py
```

### 3. Load Benchmark Tasks

```python
from qwen3_agent.agents.people_search_agent.data_loader import load_default_benchmark

tasks = load_default_benchmark()
print(f"Loaded {len(tasks)} tasks")
```

## Quick Start with CLI 🚀

The easiest way to use the People Search Agent is through the interactive CLI:

### 1. Start Ollama (if not already running)
```bash
ollama serve
```

### 2. Pull Qwen3 model (first time only)
```bash
ollama pull qwen3:14b
```

### 3. Run the CLI
```bash
# From project root
./scripts/people_search_cli.sh

# Or directly
uv run python -m qwen3_agent.agents.people_search_agent.cli
```

### 4. Ask questions!
```
🔍 Your query: Find AI researchers with NLP experience

💭 Processing query...
🔧 Executing: search_profiles
📊 Found 10 profiles...
🔧 Executing: read_profile
📊 Profile Details...
✅ Found 5 matching profiles:
  1. john-smith-nlp
  2. jane-doe-ai
  ...
```

**See [CLI_GUIDE.md](CLI_GUIDE.md) for detailed CLI documentation.**

## Usage

### Basic Usage

```python
from qwen3_agent.agents.people_search_agent import (
    PeopleSearchAgent,
    PeopleSearchEvaluator,
    PeopleSearchTask
)
from qwen3_agent.core.framework.rollout import generic_rollout
from qwen3_agent.core.framework.llm_inference import LLMInference

# Create agent and evaluator
agent = PeopleSearchAgent()
evaluator = PeopleSearchEvaluator(max_turns=10, verbose=True)

# Create task
task = PeopleSearchTask(
    id="1",
    query="AI researchers with experience in machine learning",
    expected_profiles=["profile-1", "profile-2", "profile-3"],
)

# Create LLM inference
llm = LLMInference(model="qwen3-14b")

# Run rollout
trajectory = await generic_rollout(
    llm=llm,
    task=task,
    agent=agent,
    evaluator=evaluator,
)

print(f"Reward: {trajectory.reward}")
```

### Load from Benchmark

```python
from qwen3_agent.agents.people_search_agent.data_loader import load_default_benchmark

# Load all benchmark tasks
tasks = load_default_benchmark()

# Filter by batch if needed
batch_1_tasks = [t for t in tasks if t.batch == "1"]

# Run on first task
task = tasks[0]
print(f"Query: {task.query}")
print(f"Expected: {len(task.expected_profiles)} profiles")
```

## Benchmark Data

- **Source**: `data/benchmark-queries-flattened.csv`
- **Tasks**: 932 queries
- **Batches**: 10 batches
- **Format**: Each task has a query and list of expected linkedin_handles

Example queries:
- "C-level executives in shared services, HR, or transportation in Saudi Arabia"
- "Content marketers and editors specializing in SEO for AI or SaaS companies"
- "Robotics AI researchers and engineers with experience in reinforcement learning"

## Files

```
people_search_agent/
├── __init__.py           # Package exports
├── agent.py              # PeopleSearchAgent implementation
├── evaluator.py          # PeopleSearchEvaluator with reward calculation
├── tasks.py              # PeopleSearchTask definition
├── tools.py              # search_profiles, read_profile, return_final_answer
├── data_loader.py        # Load benchmark queries
├── create_profile_db.py  # Script to create SQLite database
├── test_tools.py         # Tool testing script
├── profiles.db           # SQLite database (93,417 profiles)
├── data/
│   ├── profile_detail.csv           # Raw profile data (2.8 GB)
│   └── benchmark-queries-flattened.csv  # Benchmark queries
└── README.md             # This file
```

## Development

### Adding New Tools

1. Define tool function in `tools.py`
2. Add tool schema in `agent.py` `_prepare_tools()`
3. Implement execution in `agent.py` `execute_action()`
4. Update evaluator tracking in `evaluator.py` `on_action_executed()`

### Modifying Reward Function

Edit `PeopleSearchEvaluator._calculate_reward()` in `evaluator.py`.

### Testing

```bash
# Test tools
python test_tools.py

# Test data loader
python -m qwen3_agent.agents.people_search_agent.data_loader

# Run with framework
python -m qwen3_agent.core.framework.rollout \
    --agent people_search \
    --task benchmark \
    --model qwen3-14b
```

## Notes

- Database is read-only in tools (uses `mode=ro`)
- FTS5 full-text search for fast keyword matching
- Profiles have large text fields (experiences, education) stored as JSON strings
- Some profiles have missing fields (about, summary, etc.)
- LinkedIn handles are unique identifiers

## Troubleshooting

**Database not found:**
```bash
cd qwen3_agent/agents/people_search_agent
uv run python create_profile_db.py
```

**CSV field size error:**
Already handled with `csv.field_size_limit(sys.maxsize)` in create_profile_db.py

**Import errors:**
Make sure you're running from project root:
```bash
cd /path/to/rl-qwen3
uv run python -m qwen3_agent.agents.people_search_agent.test_tools
```


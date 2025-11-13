# 快速命令参考

## ⚠️ 重要
**所有命令都必须从项目根目录运行！**

```bash
cd /Users/zhenwei/workspace/rl-qwen3
```

---

## 设置（一次性）

```bash
cd qwen3_agent/agents/people_search_agent

# 1. 创建训练数据库
python3 create_training_db.py

# 2. 筛选 benchmark
python3 filter_benchmark.py

# 3. 验证
python3 verify_setup.py
```

---

## 日常使用

### 运行 Benchmark

```bash
# 从项目根目录
cd /Users/zhenwei/workspace/rl-qwen3

# 方法 1: 使用脚本（推荐）
./scripts/benchmark_people_search.sh -n 100

# 方法 2: 使用模块
uv run python -m qwen3_agent.agents.people_search_agent.benchmark -n 100
```

### 运行 CLI

```bash
# 从项目根目录
cd /Users/zhenwei/workspace/rl-qwen3

# 方法 1: 使用脚本（推荐）
./scripts/people_search_cli.sh

# 方法 2: 使用模块
uv run python -m qwen3_agent.agents.people_search_agent.cli
```

### 训练

```bash
# 从项目根目录
cd /Users/zhenwei/workspace/rl-qwen3

python3 qwen3_agent/train.py --agent people_search --episodes 1000
```

---

## 常见错误

### ❌ ImportError: attempted relative import with no known parent package

**错误原因：** 在错误的目录运行或使用了错误的命令

**解决方案：**
```bash
# ❌ 错误
cd qwen3_agent
python agents/people_search_agent/benchmark.py

# ✅ 正确
cd /Users/zhenwei/workspace/rl-qwen3
./scripts/benchmark_people_search.sh -n 100
```

---

## 快速测试

```bash
cd /Users/zhenwei/workspace/rl-qwen3

# 测试 10 个查询（约 2-3 分钟）
./scripts/benchmark_people_search.sh -n 10
```


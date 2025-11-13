# 训练数据库说明

## 简介

为了加快训练速度，从完整的 profile 数据库中筛选出了 10,000 个 profiles 创建了训练数据库。

## 为什么需要筛选 Benchmark？

**`filter_benchmark.py` 的作用：**

原始 benchmark 文件有 932 个查询，每个查询期望找到特定的 profiles。但训练数据库只有 10,000 个 profiles，不是所有期望的 profiles 都在其中。

**问题示例：**
```
查询："找到所有在 Google 工作的 AI 研究员"
期望的 profiles: ['john-smith', 'jane-doe', 'bob-wilson']

但是 'bob-wilson' 不在训练数据库的 10,000 个 profiles 中
→ agent 无法找到这个 profile
→ 评估会错误地认为 agent 表现不好
```

**解决方案：**
`filter_benchmark.py` 筛选掉那些期望 profiles 不完全在训练数据库中的查询。

**结果：**
- 原始 benchmark: 932 个查询
- 筛选后: 872 个查询（保留率 93.6%）
- 被过滤: 60 个查询（因为期望的某些 profiles 不在训练数据库中）

这样确保评估结果准确，不会因为数据库中缺少 profiles 而影响评估。

## 文件结构

```
qwen3_agent/agents/people_search_agent/
├── profiles_training.db                    # 训练数据库（339MB，10,000 profiles）
│
├── data/
│   ├── 10000-training-linkedin-handle.csv  # 训练 profile 列表
│   ├── profile_detail.csv                  # 完整 profile 数据（2.8GB）
│   ├── benchmark-queries-flattened.csv     # 完整 benchmark（932 查询）
│   └── benchmark-queries-training.csv      # 训练 benchmark（872 查询）✨
│
├── create_training_db.py                   # 创建训练数据库
├── filter_benchmark.py                     # 筛选 benchmark 查询 ✨
├── test_training_db.py                     # 测试数据库
└── verify_setup.py                         # 验证配置
```

## 快速开始

### 1. 创建训练数据库（一次性）

```bash
cd qwen3_agent/agents/people_search_agent
python3 create_training_db.py
```

输出：
```
✓ Filtered database created successfully!
  - Inserted: 10000 profiles
  - Coverage: 100.0%
```

### 2. 筛选 Benchmark 查询（一次性）

```bash
python3 filter_benchmark.py
```

输出：
```
✓ Filtering complete!
  - Kept: 872 queries
  - Filtered out: 60 queries
  - Retention rate: 93.6%
```

### 3. 验证设置

```bash
python3 verify_setup.py
```

### 4. 使用

**重要：所有命令都从项目根目录运行**

```bash
cd /Users/zhenwei/workspace/rl-qwen3

# 运行 benchmark（使用脚本，推荐）
./scripts/benchmark_people_search.sh -n 100

# 或使用模块方式
uv run python -m qwen3_agent.agents.people_search_agent.benchmark -n 100

# 运行 CLI
uv run python -m qwen3_agent.agents.people_search_agent.cli

# 训练
python3 qwen3_agent/train.py --agent people_search
```

## 默认行为

**所有组件现在都使用训练数据库和训练 benchmark：**

✅ `tools.py` → 使用 `profiles_training.db`
✅ `data_loader.py` → 加载 `benchmark-queries-training.csv`
✅ `benchmark.py` → 使用训练 benchmark
✅ `agent.py` → 通过 tools 使用训练数据库
✅ `cli.py` → 通过 tools 使用训练数据库

## 性能对比

| 指标 | 完整数据库 | 训练数据库 | 提升 |
|------|-----------|-----------|------|
| Profiles 数量 | 93,417 | 10,000 | - |
| 文件大小 | 3GB+ | 339MB | **9x 小** |
| 搜索速度 | ~500ms | ~100ms | **5x 快** |
| Benchmark | 932 查询 | 872 查询 | 93.6% |

## 工作流程

```
10000-training-linkedin-handle.csv (你提供的列表)
            ↓
    create_training_db.py
            ↓
    profiles_training.db (339MB, 10,000 profiles)
            ↓
    filter_benchmark.py
            ↓
    benchmark-queries-training.csv (872 queries)
            ↓
    benchmark.py / train.py / cli.py
    （所有组件自动使用训练数据库和训练 benchmark）
```

## 总结

1. ✅ 训练数据库已创建（10,000 profiles）
2. ✅ Benchmark 已筛选（872 个匹配的查询）
3. ✅ 所有组件默认使用训练数据库
4. ✅ 不需要任何配置，直接使用即可
5. ✅ 查询速度提升 5 倍，文件大小减少 9 倍


# 快速训练验证指南

本指南帮助你用最小配置快速验证整个训练流程是否正常工作。

## 当前训练循环说明

**重要**: 当前版本是**简化的独立训练脚本**，专注于：

✅ **已实现的功能**:
1. 数据加载 (Enron 邮件数据集)
2. 轨迹生成 (rollout with tool calling)
3. 模型评估 (benchmark)
4. 训练循环结构
5. 奖励计算
6. 异常处理和日志

⚠️ **未实现的功能**:
- **梯度更新**: 当前代码生成训练数据但不更新模型权重
- **权重保存/加载**: 需要集成训练后端（如 vLLM + LoRA）
- **完整 RL 算法**: 需要 PPO/GRPO 实现

## 快速验证步骤

### 1. 快速训练测试（5-10分钟）

验证整个训练循环能正常执行：

```bash
# 使用最小配置快速测试
./scripts/quick_train.sh
```

这会执行：
- ✅ 加载 10 个训练样本
- ✅ 初始评估（5个验证样本）
- ✅ 生成训练轨迹（2 groups × 2 trajectories）
- ✅ 每步后评估
- ✅ 最终评估

预期输出：
```
============================================================
Starting training: qwen3-email-agent-quick-test
...
--- Evaluating at Global Step 0 ---
=== Benchmarking qwen3-email-agent-quick-test ===
Evaluating on 5 test scenarios...
...
Generating 2 trajectory groups...
Generated 4 trajectories
Average reward: 0.xxx
NOTE: Training step skipped - implement gradient updates for full training
...
Training finished for qwen3-email-agent-quick-test
============================================================
```

### 2. 验证评估流程

单独测试评估：

```bash
# 小规模评估
export TEST_SET_SIZE=5
export BENCH_SEQUENTIAL=true
./scripts/benchmark.sh
```

### 3. 单样本详细测试

查看完整的推理过程：

```bash
export SHOW_TRAJECTORY_DETAILS=true
export VERBOSE=true
./scripts/quick_eval.sh
```

## 配置说明

### 快速测试配置 (quick_train.sh)

| 参数 | 快速测试值 | 默认值 | 说明 |
|------|-----------|--------|------|
| TRAINING_DATASET_SIZE | 10 | 4000 | 训练样本数 |
| VAL_SET_SIZE | 5 | 100 | 验证样本数 |
| TRAJECTORIES_PER_GROUP | 2 | 6 | 每组轨迹数 |
| GROUPS_PER_STEP | 2 | 8 | 每步的组数 |
| EVAL_STEPS | 1 | 30 | 评估频率 |
| NUM_EPOCHS | 2 | 4 | 训练轮数 |
| MAX_TURNS | 5 | 10 | 最大对话轮数 |
| MAX_TOKENS | 512 | 2048 | 最大生成tokens |

总步数计算：
```
总步数 = (TRAINING_DATASET_SIZE / GROUPS_PER_STEP) * NUM_EPOCHS
       = (10 / 2) * 2
       = 10 步
```

每步会：
1. 生成 GROUPS_PER_STEP × TRAJECTORIES_PER_GROUP = 4 个轨迹
2. 如果 step % EVAL_STEPS == 0，执行评估

### 渐进式扩展

验证通过后，逐步增加规模：

#### 阶段1: 最小测试（5-10分钟）
```bash
./scripts/quick_train.sh
```

#### 阶段2: 小规模训练（30分钟-1小时）
```bash
export TRAINING_DATASET_SIZE=100
export VAL_SET_SIZE=20
export EVAL_STEPS=5
./scripts/train.sh
```

#### 阶段3: 中等规模训练（2-4小时）
```bash
export TRAINING_DATASET_SIZE=1000
export VAL_SET_SIZE=50
export EVAL_STEPS=20
./scripts/train.sh
```

#### 阶段4: 完整训练（8-12小时）
```bash
# 使用默认配置
./scripts/train.sh
```

## 实现完整训练的步骤

要实现真正的权重更新，需要：

### 方案 A: 集成 vLLM + LoRA

1. 修改 `qwen3_agent/train.py` 在训练步骤中：
   ```python
   # 当前 (第157行):
   # In a full implementation, you would train on these trajectories here
   
   # 需要添加:
   # 1. 将轨迹转换为训练数据
   # 2. 使用 vLLM 的 LoRA 训练接口
   # 3. 更新模型权重
   # 4. 保存 checkpoint
   ```

2. 启动支持 LoRA 的 vLLM 服务器
3. 在训练循环中调用 LoRA 更新 API

### 方案 B: 使用 HuggingFace Transformers

1. 加载模型到本地
2. 使用 `transformers.Trainer` 或手动梯度更新
3. 定期保存 checkpoint
4. 在评估时加载最新权重

### 方案 C: 集成 OpenPipe/ART 完整框架

使用完整的 ART 库提供的训练后端。

## 监控训练进度

### 查看日志
```bash
# 实时监控
tail -f training.log

# 搜索评估结果
grep "Benchmark Results" training.log
```

### 检查指标
训练过程中会输出：
- `reward`: 平均奖励 (-2 到 2)
- `answer_correct`: 答案正确率 (0-1)
- `sources_correct`: 来源正确率 (0-1)
- `num_turns`: 平均轮数
- `duration`: 平均时间

### 预期改进趋势

**注意**: 由于当前版本不更新权重，指标不会随训练改善。

在完整实现中，预期看到：
- ✅ `reward` 从 ~0 提升到 ~1.5
- ✅ `answer_correct` 从 ~0.3 提升到 ~0.8
- ✅ `num_turns` 逐渐减少（更高效）

## 故障排查

### 问题1: 所有轨迹都失败

**症状**: `exceptions=100`

**解决**:
```bash
export ROLLOUT_CONCURRENCY=1
export LITELLM_TIMEOUT=120
export BENCH_SEQUENTIAL=true
```

### 问题2: vLLM 超时

**症状**: `Request timed out`

**解决**:
- 降低并发: `ROLLOUT_CONCURRENCY=1`
- 增加超时: `LITELLM_TIMEOUT=120`
- 使用顺序模式: `BENCH_SEQUENTIAL=true`
- 检查 vLLM 服务器负载

### 问题3: 内存不足

**症状**: CUDA OOM

**解决**:
```bash
# 减少批量大小
export GROUPS_PER_STEP=2
export TRAJECTORIES_PER_GROUP=2

# 减少生成长度
export MAX_TOKENS=512
```

### 问题4: 数据集未找到

**症状**: `FileNotFoundError: enron_emails.db`

**解决**:
```bash
./scripts/generate_database.sh
```

## 性能优化

### GPU 优化
- 使用 tensor parallelism: `vllm serve --tensor-parallel-size 2`
- 增加 GPU 利用率: `--gpu-memory-utilization 0.95`

### 吞吐量优化
- 增加并发: `ROLLOUT_CONCURRENCY=5`（确保不超时）
- 使用并行评估: `BENCH_SEQUENTIAL=false`
- 减少评估频率: `EVAL_STEPS=50`

### 磁盘优化
- 使用 SSD 存储数据集
- 将日志写入单独的磁盘

## 总结

1. **快速验证**: 使用 `./scripts/quick_train.sh` 验证流程（10分钟）
2. **检查输出**: 确认每个步骤都正常执行
3. **渐进扩展**: 逐步增加数据规模
4. **集成训练**: 添加梯度更新实现完整 RL

当前实现已经包含了 RL 训练的所有框架代码，只需要添加实际的权重更新逻辑即可。


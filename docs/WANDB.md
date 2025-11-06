# WandB 集成指南

本项目集成了 [Weights & Biases (WandB)](https://wandb.ai/) 用于实验跟踪和可视化训练进度。

## 快速开始

### 1. 安装 WandB

```bash
pip install wandb
# 或
uv sync  # 已包含在 requirements.txt
```

### 2. 登录 WandB

```bash
wandb login
```

或者在 `.env` 文件中设置 API key:

```bash
WANDB_API_KEY=your_api_key_here
```

### 3. 配置环境变量

在 `.env` 文件中添加：

```bash
# 启用 WandB
WANDB_MODE=online

# 项目名称（可选）
WANDB_PROJECT=qwen3-email-agent

# 团队/用户名（可选）
WANDB_ENTITY=your_username
```

### 4. 运行训练

```bash
./scripts/train.sh
# 或快速测试
./scripts/quick_train.sh
```

训练开始时会看到：

```
✓ WandB initialized: https://wandb.ai/username/project/runs/xxxxx
```

## 记录的指标

### 评估指标 (eval/*)

每 `EVAL_STEPS` 步记录一次：

- `eval/reward`: 平均奖励 (-2 到 2)
- `eval/answer_correct`: 答案正确率 (0-1)
- `eval/sources_correct`: 来源引用正确率 (0-1)
- `eval/num_turns`: 平均对话轮数
- `eval/duration`: 平均执行时间（秒）
- `eval/prompt_tokens`: 平均 prompt tokens
- `eval/completion_tokens`: 平均 completion tokens
- `eval/n_trajectories`: 成功评估的轨迹数

### 训练指标 (train/*)

每个训练步骤记录：

- `train/total_trajectories`: 生成的轨迹总数
- `train/avg_reward`: 平均奖励
- `train/epoch`: 当前 epoch
- `train/epoch_step`: epoch 内的步数
- `train/answer_correct`: 答案正确率
- `train/attempted_answer`: 尝试回答的比例
- `train/num_turns`: 平均轮数
- `train/prompt_tokens`: 平均 prompt tokens
- `train/completion_tokens`: 平均 completion tokens

### 最终评估指标 (final_eval/*)

训练结束时记录：

- `final_eval/reward`: 最终平均奖励
- `final_eval/answer_correct`: 最终答案正确率
- 其他所有评估指标

## 查看训练进度

### 在 WandB Dashboard

1. 打开训练输出中的 URL
2. 或访问 https://wandb.ai/your_username/qwen3-email-agent
3. 查看实时更新的：
   - 📈 指标曲线图
   - 📊 系统资源监控
   - 📝 配置参数
   - 🔍 运行日志

### 关键可视化

**推荐添加的图表：**

1. **Reward 曲线**
   - X轴: step
   - Y轴: eval/reward, train/avg_reward
   - 查看模型性能提升

2. **答案正确率**
   - X轴: step
   - Y轴: eval/answer_correct, train/answer_correct
   - 监控学习效果

3. **效率指标**
   - X轴: step
   - Y轴: eval/num_turns, eval/duration
   - 查看推理效率

4. **Token 使用**
   - X轴: step  
   - Y轴: eval/prompt_tokens, eval/completion_tokens
   - 监控成本

## 禁用 WandB

如果不想使用 WandB，在 `.env` 中设置：

```bash
WANDB_MODE=disabled
```

或者不安装 wandb 包。训练会自动检测并跳过 WandB 日志。

## 多次运行对比

### 标记不同实验

使用不同的 `RUN_ID`:

```bash
# 实验1: 基线
export RUN_ID=baseline
./scripts/train.sh

# 实验2: 更高学习率
export RUN_ID=high-lr
export LEARNING_RATE=2e-5
./scripts/train.sh

# 实验3: 更多轨迹
export RUN_ID=more-traj
export TRAJECTORIES_PER_GROUP=10
./scripts/train.sh
```

### 在 WandB 中对比

1. 在 Dashboard 选择多个 runs
2. 点击 "Compare" 
3. 查看并排对比的指标曲线

## 最佳实践

### 1. 命名规范

使用描述性的 `RUN_ID`:

```bash
# 好的命名
RUN_ID=qwen3-14b-lr1e5-traj6
RUN_ID=baseline-high-turns
RUN_ID=ablation-no-tools

# 避免
RUN_ID=test1
RUN_ID=run2
```

### 2. 添加标签 (Tags)

在代码中已自动添加标签：

```python
tags=["qwen3", "email-agent", "rl-training"]
```

可以在 WandB UI 中手动添加更多标签，如：
- `baseline`
- `hyperparameter-tuning`
- `ablation-study`

### 3. 添加笔记

在 WandB UI 的 "Notes" 中记录：
- 实验目的
- 预期结果
- 观察到的问题
- 下一步计划

### 4. 保存重要运行

对于特别好的结果，点击 ⭐ 标记为 favorite

## 高级功能

### 自定义项目名称

```bash
export WANDB_PROJECT=my-custom-project
./scripts/train.sh
```

### 团队协作

```bash
export WANDB_ENTITY=my-team-name
./scripts/train.sh
```

### 离线模式

如果网络不稳定，先离线运行：

```bash
export WANDB_MODE=offline
./scripts/train.sh

# 训练完成后同步
wandb sync wandb/latest-run
```

### 自定义日志

如果想添加额外指标，修改 `qwen3_agent/train.py`:

```python
if use_wandb:
    wandb.log({
        "custom/my_metric": value,
        "custom/another_metric": another_value,
    }, step=global_step)
```

## 故障排查

### 问题1: WandB 未初始化

**症状**: 看到 "⚠ WandB not installed"

**解决**:
```bash
pip install wandb
# 或
uv sync
```

### 问题2: 登录失败

**症状**: `wandb.errors.UsageError: api_key not configured`

**解决**:
```bash
# 方法1: 交互式登录
wandb login

# 方法2: 环境变量
export WANDB_API_KEY=your_key

# 方法3: .env 文件
echo "WANDB_API_KEY=your_key" >> .env
```

### 问题3: 同步失败

**症状**: 网络超时或上传失败

**解决**:
```bash
# 使用离线模式
export WANDB_MODE=offline
./scripts/train.sh

# 后续手动同步
wandb sync wandb/offline-run-xxxxx
```

### 问题4: 不想用 WandB

**解决**:
```bash
# 完全禁用
export WANDB_MODE=disabled

# 或者删除 wandb 包
pip uninstall wandb
```

## 示例 Dashboard

训练完成后，你的 WandB dashboard 将显示：

### 概览页面
- ✅ 运行状态（Running/Finished/Failed）
- ⏱️ 运行时间
- 📊 关键指标摘要
- 💻 系统信息（GPU, CPU, Memory）

### 图表页面
- 📈 Reward 随时间变化
- 📉 Loss 曲线（如果实现了梯度更新）
- 📊 答案正确率提升
- ⚡ 推理速度变化

### 配置页面
- 🔧 所有超参数
- 📝 模型配置
- 🎯 训练设置

### 日志页面
- 📋 完整训练日志
- ⚠️ 警告和错误
- 💬 Verbose 输出

## 相关资源

- [WandB 官方文档](https://docs.wandb.ai/)
- [Python API 参考](https://docs.wandb.ai/ref/python)
- [WandB 示例项目](https://wandb.ai/gallery)
- [社区论坛](https://community.wandb.ai/)

## 总结

WandB 集成让你能够：

1. 📊 **实时监控** - 随时查看训练进度
2. 📈 **可视化** - 美观的图表和仪表板
3. 🔄 **对比实验** - 轻松比较多次运行
4. 🤝 **团队协作** - 共享实验结果
5. 📱 **移动访问** - 在手机上查看进度
6. 🎯 **超参数跟踪** - 自动记录所有配置

开始使用吧！🚀


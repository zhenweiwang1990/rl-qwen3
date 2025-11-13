# Qwen3 Email Agent - Generic RL Training Framework

🎉 **重大更新**: 完全重构为通用的 Agent 训练框架！

这是一个基于强化学习的 Email 搜索 Agent，使用 [OpenPipe ART](https://github.com/OpenPipe/ART) 框架进行训练。现在已经重构为通用的 RL Agent 训练框架，支持任意类型的 Agent。

## ✨ 特性

- 🤖 **通用 Agent 框架** - 支持任意类型的 Agent 和任务
- 🔥 **完整 RL 训练** - 使用 PPO/GRPO 进行梯度更新
- 📧 **Email Agent** - 完整的 email 搜索和阅读 Agent 实现
- 🎯 **灵活评估** - 可自定义的 reward 函数和评估指标
- 🔌 **LLM 统一接口** - 支持训练模型和外部模型
- 📊 **详细追踪** - 完整的训练和评估指标追踪

## 🏗️ 新架构

```
qwen3_agent/
├── core/framework/          # 通用框架 ⭐
│   ├── task.py             # 任务抽象
│   ├── agent.py            # Agent 抽象
│   ├── evaluator.py        # 评估器抽象
│   ├── llm_inference.py    # LLM 推理
│   └── rollout.py          # 通用 rollout
├── agents/                  # Agent 实现
│   └── email_agent/        # Email Agent
└── train.py                # 训练脚本
```

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/your-org/rl-qwen3
cd rl-qwen3

# 使用 uv 安装依赖
./scripts/setup.sh
```

### 训练 Email Agent

```bash
# 启动 vLLM 服务器（如果使用本地模型）
./scripts/start_vllm.sh

# 训练
./scripts/train_with_rl.sh
```

### 评估

```bash
# 快速评估单个场景
./scripts/quick_eval.sh

# 完整 benchmark
./scripts/benchmark.sh
```

## 📖 文档

- **[FRAMEWORK.md](FRAMEWORK.md)** ⭐ 框架使用指南
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - 5 分钟快速参考
- **[REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)** - 重构总结
- **[examples/simple_math_agent.py](examples/simple_math_agent.py)** - 示例 Agent

## 🎓 创建自定义 Agent

只需实现 3 个类即可创建新 Agent：

```python
from qwen3_agent.core.framework import BaseAgent, BaseTask, BaseEvaluator

# 1. 定义任务
class MyTask(BaseTask):
    def get_query(self): return self.question
    def get_ground_truth(self): return self.answer

# 2. 实现 Agent
class MyAgent(BaseAgent):
    def get_system_prompt(self, task): ...
    def get_tools_schema(self): ...
    def execute_action(self, tool_name, args, task): ...
    def is_terminal_action(self, tool_name): ...

# 3. 实现评估器
class MyEvaluator(BaseEvaluator):
    def create_rubric(self): ...
    async def evaluate_trajectory(self, traj, task, rubric): ...
    def on_action_executed(self, rubric, ...): ...

# 使用
trajectory = await generic_rollout(llm, task, agent, evaluator)
```

完整示例请查看 `examples/simple_math_agent.py`。

## 🧪 测试

```bash
# 测试框架
uv run python test_framework.py

# 测试导入
uv run python test_import.py

# 测试示例 Agent
uv run python examples/simple_math_agent.py
```

## 📊 Email Agent 性能

Email Agent 可以：
- 🔍 搜索用户的 email
- 📖 读取特定 email 内容
- 🎯 基于内容回答问题
- 📝 引用正确的 email 来源

使用 Enron email 数据集进行训练和评估。

## 🛠️ 技术栈

- **RL 框架**: [OpenPipe ART](https://github.com/OpenPipe/ART)
- **LLM**: Qwen3 14B (可自定义)
- **推理**: vLLM (本地) 或 OpenAI API
- **数据**: Enron Email 数据集
- **评估**: GPT-4o 作为 judge

## 📁 项目结构

```
rl-qwen3/
├── qwen3_agent/
│   ├── core/framework/      # 通用框架
│   ├── agents/email_agent/  # Email Agent 实现
│   ├── data/                # 数据加载
│   ├── train.py             # 训练脚本
│   ├── benchmark.py         # 评估脚本
│   └── config.py            # 配置
├── examples/                # 示例 Agent
├── scripts/                 # 辅助脚本
├── docs/                    # 文档
└── test_framework.py        # 测试
```

## 🤝 贡献

欢迎贡献！特别欢迎：
- 新的 Agent 实现
- 改进现有 Agent
- 文档改进
- Bug 修复

## 📝 License

MIT License

## 🔗 相关链接

- [OpenPipe ART](https://github.com/OpenPipe/ART)
- [Qwen3](https://github.com/QwenLM/Qwen)
- [vLLM](https://github.com/vllm-project/vllm)

---

⭐ 如果觉得有用，请给个 Star！

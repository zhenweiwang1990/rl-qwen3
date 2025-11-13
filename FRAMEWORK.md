# Generic RL Agent Training Framework

## 📋 概述

这个框架将 Email Agent 的训练系统彻底重构为通用的 RL Agent 训练框架，支持任意类型的 Agent 接入。

**重要**：所有代码已完全迁移到新框架，不再有兼容层。这确保代码库的一致性和可维护性。

## 🏗️ 架构

```
qwen3_agent/
├── core/framework/          # 通用框架层
│   ├── task.py             # BaseTask - 任务抽象
│   ├── agent.py            # BaseAgent - Agent 抽象  
│   ├── evaluator.py        # BaseEvaluator - 评估器抽象
│   ├── llm_inference.py    # LLMInference - 统一推理接口
│   └── rollout.py          # generic_rollout - 通用 rollout
├── agents/                  # 具体 Agent 实现
│   └── email_agent/        # Email Agent 实现
│       ├── agent.py        # EmailAgent
│       ├── tasks.py        # EmailTask
│       ├── evaluator.py    # EmailEvaluator
│       └── tools.py        # Email 工具
├── train.py                # 训练脚本（使用新框架）
└── benchmark.py            # 评估脚本（使用新框架）
```

## 🚀 快速开始

### 使用 Email Agent

训练和评估代码直接使用新框架：

```python
from qwen3_agent.core.framework import LLMInference, generic_rollout
from qwen3_agent.agents.email_agent import EmailAgent, EmailTask, EmailEvaluator

# 创建组件
task = EmailTask.from_synthetic_query(scenario)
evaluator = EmailEvaluator(verbose=True, max_turns=10)
agent = EmailAgent(evaluator=evaluator)
llm = LLMInference(model)

# 执行 rollout
trajectory = await generic_rollout(
    llm=llm,
    task=task,
    agent=agent,
    evaluator=evaluator,
    max_turns=10,
    use_native_tools=True,
    verbose=True,
)
```

### 创建新的 Agent

#### 1. 定义 Task

```python
from qwen3_agent.core.framework import BaseTask

class MyTask(BaseTask):
    """自定义任务."""
    
    question: str
    answer: str
    
    def get_query(self) -> str:
        return self.question
    
    def get_ground_truth(self):
        return self.answer
```

#### 2. 实现 Agent

```python
from qwen3_agent.core.framework import BaseAgent, ActionResult

class MyAgent(BaseAgent):
    """自定义 Agent."""
    
    def get_system_prompt(self, task: BaseTask) -> str:
        return f"You are an agent. Task: {task.get_query()}"
    
    def get_tools_schema(self) -> List[Dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "my_tool",
                    "description": "My custom tool",
                    "parameters": {...}
                }
            }
        ]
    
    def execute_action(self, tool_name: str, tool_args: Dict, task: BaseTask) -> ActionResult:
        if tool_name == "my_tool":
            result = my_tool_implementation(**tool_args)
            return ActionResult(success=True, data=result)
        return ActionResult(success=False, error="Unknown tool")
    
    def is_terminal_action(self, tool_name: str) -> bool:
        return tool_name == "finish"
```

#### 3. 实现 Evaluator

```python
from dataclasses import dataclass
from qwen3_agent.core.framework import BaseEvaluator, BaseRubric

@dataclass
class MyRubric(BaseRubric):
    """自定义评估指标."""
    answer_correct: bool = False
    num_steps: int = 0

class MyEvaluator(BaseEvaluator[MyRubric]):
    """自定义评估器."""
    
    def create_rubric(self) -> MyRubric:
        return MyRubric()
    
    async def evaluate_trajectory(self, trajectory, task, rubric) -> float:
        # 计算最终奖励
        return 1.0 if rubric.answer_correct else 0.0
    
    def on_action_executed(self, rubric, tool_name, tool_args, result, task):
        # 更新评估指标
        rubric.num_steps += 1
        if tool_name == "check_answer":
            rubric.answer_correct = (result == task.get_ground_truth())
```

#### 4. 使用 Generic Rollout

```python
from qwen3_agent.core.framework import LLMInference, generic_rollout

# 创建组件
task = MyTask(id="1", question="...", answer="...")
agent = MyAgent()
evaluator = MyEvaluator()
llm = LLMInference(art_model)  # 或外部模型

# 执行 rollout
trajectory = await generic_rollout(
    llm=llm,
    task=task,
    agent=agent,
    evaluator=evaluator,
    max_turns=10,
    use_native_tools=True,
    verbose=True,
)

print(f"Reward: {trajectory.reward}")
print(f"Metrics: {trajectory.metrics}")
```

## 🔧 核心组件

### 1. BaseTask

任务的抽象表示，定义：
- `get_query()` - 返回给 Agent 的查询
- `get_ground_truth()` - 返回正确答案（用于评估）
- `get_context()` - 返回额外上下文信息

### 2. BaseAgent

Agent 的抽象表示，定义：
- `get_system_prompt(task)` - 生成系统提示
- `get_tools_schema()` - 返回工具定义
- `execute_action(tool_name, args, task)` - 执行工具
- `is_terminal_action(tool_name)` - 判断是否终止
- `parse_action(message, use_native)` - 解析 LLM 响应（有默认实现）

### 3. BaseEvaluator

评估器的抽象表示，定义：
- `create_rubric()` - 创建评估指标实例
- `evaluate_trajectory(traj, task, rubric)` - 计算最终奖励（async）
- `on_action_executed(rubric, ...)` - 每步后更新指标
- `on_parsing_error(rubric, ...)` - 处理错误

### 4. LLMInference

统一的 LLM 推理接口：
- 支持 ART 训练模型
- 支持外部模型（OpenAI、Anthropic 等）
- 自动处理 caching、token 追踪等

### 5. generic_rollout

通用的 rollout 执行函数：
- 管理对话循环
- 调用 LLM 生成响应
- 执行 Agent 动作
- 追踪评估指标
- 计算最终奖励

## 📊 LLM 推理接口特性

### 支持多种模型类型

```python
# ART 训练模型
llm = LLMInference(art_trainable_model)

# ART 冻结模型
llm = LLMInference(art_model)

# OpenAI 模型
llm = LLMInference("openai/gpt-4o", {"api_key": "..."})

# Anthropic 模型
llm = LLMInference("anthropic/claude-3-5-sonnet-20241022", {"api_key": "..."})

# 自定义端点
llm = LLMInference("openai/custom", {
    "base_url": "http://localhost:8000/v1",
    "api_key": "dummy"
})
```

### 自动配置

- **Caching**: 训练模型禁用缓存，其他模型启用
- **Token 追踪**: 自动记录 prompt 和 completion tokens
- **错误处理**: 统一的错误处理和重试机制

## ✅ 优势

1. **完全解耦**: Agent、Task、Evaluator 独立实现
2. **易于扩展**: 新 Agent 只需实现 4-5 个接口
3. **LLM 复用**: 外部 Agent 可以使用训练中的模型
4. **类型安全**: 使用 ABC 和 Pydantic 保证类型检查
5. **代码一致**: 所有代码使用统一的新框架，无技术债务

## 📝 示例：新架构

### 训练脚本 (train.py)

```python
from qwen3_agent.core.framework import LLMInference, generic_rollout
from qwen3_agent.agents.email_agent import EmailAgent, EmailTask, EmailEvaluator

# 创建组件
evaluator = EmailEvaluator(...)
agent = EmailAgent(evaluator=evaluator)
llm = LLMInference(model)

# 生成 trajectories
groups = await art.gather_trajectory_groups(
    (
        art.TrajectoryGroup(
            (
                generic_rollout(
                    llm=llm,
                    task=EmailTask.from_synthetic_query(scenario),
                    agent=agent,
                    evaluator=evaluator,
                    ...
                )
                for _ in range(trajectories_per_group)
            )
        )
        for scenario in batch
    )
)
```

### 评估脚本 (benchmark.py)

```python
from qwen3_agent.core.framework import LLMInference, generic_rollout
from qwen3_agent.agents.email_agent import EmailAgent, EmailTask, EmailEvaluator

# 创建组件
evaluator = EmailEvaluator(...)
agent = EmailAgent(evaluator=evaluator)
llm = LLMInference(model)

# 运行评估
trajectories = await gather_trajectories(
    (
        generic_rollout(
            llm=llm,
            task=EmailTask.from_synthetic_query(scenario),
            agent=agent,
            evaluator=evaluator,
            ...
        )
        for scenario in scenarios
    )
)
```

## 🧪 测试

运行测试脚本：

```bash
uv run python test_framework.py
```

测试内容：
1. ✅ 框架基础功能
2. ✅ LLM 推理接口
3. ✅ Agent 执行逻辑
4. ✅ 评估器奖励计算
5. ✅ 模块集成测试

## 📚 扩展阅读

- 查看 `qwen3_agent/agents/email_agent/` 了解完整的 Agent 实现示例
- 查看 `qwen3_agent/core/framework/` 了解框架接口定义
- 查看 `examples/simple_math_agent.py` 了解如何创建新 Agent

## 🤝 贡献新 Agent

1. 在 `qwen3_agent/agents/` 下创建新目录
2. 实现 `BaseTask`、`BaseAgent`、`BaseEvaluator`
3. 添加测试脚本
4. 更新文档

欢迎贡献！

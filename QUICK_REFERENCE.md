# Quick Reference - Generic Framework

## 🚀 5 分钟快速入门

### 创建自定义 Agent 只需 3 步

#### 1️⃣ 定义 Task

```python
from qwen3_agent.core.framework import BaseTask

class MyTask(BaseTask):
    question: str
    answer: str
    
    def get_query(self) -> str:
        return self.question
    
    def get_ground_truth(self):
        return self.answer
```

#### 2️⃣ 实现 Agent

```python
from qwen3_agent.core.framework import BaseAgent, ActionResult

class MyAgent(BaseAgent):
    def get_system_prompt(self, task):
        return f"Solve: {task.get_query()}"
    
    def get_tools_schema(self):
        return [{
            "type": "function",
            "function": {
                "name": "solve",
                "parameters": {...}
            }
        }]
    
    def execute_action(self, tool_name, tool_args, task):
        if tool_name == "solve":
            result = do_something(**tool_args)
            return ActionResult(success=True, data=result)
        return ActionResult(success=False, error="Unknown tool")
    
    def is_terminal_action(self, tool_name):
        return tool_name == "answer"
```

#### 3️⃣ 实现 Evaluator

```python
from dataclasses import dataclass
from qwen3_agent.core.framework import BaseEvaluator, BaseRubric

@dataclass
class MyRubric(BaseRubric):
    correct: bool = False

class MyEvaluator(BaseEvaluator[MyRubric]):
    def create_rubric(self):
        return MyRubric()
    
    async def evaluate_trajectory(self, traj, task, rubric):
        return 1.0 if rubric.correct else 0.0
    
    def on_action_executed(self, rubric, tool_name, tool_args, result, task):
        if tool_name == "answer":
            rubric.correct = (result == task.get_ground_truth())
```

### 使用你的 Agent

```python
from qwen3_agent.core.framework import LLMInference, generic_rollout

# 创建组件
task = MyTask(id="1", question="...", answer="...")
agent = MyAgent()
evaluator = MyEvaluator()
llm = LLMInference("openai/gpt-4o-mini", {})

# 执行
trajectory = await generic_rollout(
    llm=llm,
    task=task,
    agent=agent,
    evaluator=evaluator,
    max_turns=10,
)

print(f"Reward: {trajectory.reward}")
```

## 📦 核心组件速查

### BaseTask
| 方法 | 必须实现 | 说明 |
|------|---------|------|
| `get_query()` | ✅ | 返回给 Agent 的问题 |
| `get_ground_truth()` | ✅ | 返回正确答案 |
| `get_context()` | ❌ | 返回额外上下文 |

### BaseAgent
| 方法 | 必须实现 | 说明 |
|------|---------|------|
| `get_system_prompt(task)` | ✅ | 生成系统提示 |
| `get_tools_schema()` | ✅ | 返回工具列表 |
| `execute_action(...)` | ✅ | 执行工具 |
| `is_terminal_action(name)` | ✅ | 判断是否终止 |
| `parse_action(msg, native)` | ❌ | 解析响应（有默认实现）|

### BaseEvaluator
| 方法 | 必须实现 | 说明 |
|------|---------|------|
| `create_rubric()` | ✅ | 创建评估指标 |
| `evaluate_trajectory(...)` | ✅ | 计算最终奖励 |
| `on_action_executed(...)` | ✅ | 更新评估指标 |
| `on_parsing_error(...)` | ❌ | 处理错误 |

### LLMInference
```python
# ART 模型
llm = LLMInference(art_model)

# 外部模型
llm = LLMInference("openai/gpt-4o", {"api_key": "..."})

# 调用
response = await llm.complete(messages, tools=tools)
```

## 💡 常用模式

### 1. 工具定义（OpenAI 格式）

```python
def get_tools_schema(self):
    return [{
        "type": "function",
        "function": {
            "name": "tool_name",
            "description": "Tool description",
            "parameters": {
                "type": "object",
                "properties": {
                    "arg1": {"type": "string", "description": "..."},
                    "arg2": {"type": "number"}
                },
                "required": ["arg1"]
            }
        }
    }]
```

### 2. 工具执行

```python
def execute_action(self, tool_name, tool_args, task):
    try:
        if tool_name == "my_tool":
            result = my_implementation(**tool_args)
            return ActionResult(success=True, data=result)
        else:
            return ActionResult(success=False, error="Unknown tool")
    except Exception as e:
        return ActionResult(success=False, error=str(e))
```

### 3. 评估指标更新

```python
def on_action_executed(self, rubric, tool_name, tool_args, result, task):
    # 追踪操作次数
    rubric.num_operations += 1
    
    # 检查是否找到关键信息
    if tool_name == "search" and "important_data" in result:
        rubric.found_data = True
    
    # 检查最终答案
    if tool_name == "answer":
        rubric.answer_correct = (result == task.get_ground_truth())
```

### 4. 奖励计算

```python
async def evaluate_trajectory(self, traj, task, rubric):
    # 简单奖励
    if rubric.correct:
        return 1.0
    else:
        return 0.0
    
    # 复杂奖励（部分分）
    reward = 0.0
    if rubric.correct:
        reward += 1.0
    if rubric.found_data:
        reward += 0.3
    if rubric.efficient:
        reward += 0.2
    return reward
```

## 🔄 向后兼容

现有代码无需修改：

```python
# 自动使用新框架
from qwen3_agent.rollout_compat import rollout

trajectory = await rollout(model, scenario)
```

## 📚 完整文档

- [FRAMEWORK.md](FRAMEWORK.md) - 详细文档
- [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) - 迁移指南
- [examples/simple_math_agent.py](examples/simple_math_agent.py) - 完整示例

## 🧪 测试

```bash
# 测试新框架
uv run python test_framework.py

# 测试示例 Agent
uv run python examples/simple_math_agent.py
```

---

**提示**: 查看 `examples/simple_math_agent.py` 获取完整的可运行示例！


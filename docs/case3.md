# LangGraph

LangGraph 作为 LangChain 生态系统的重要扩展，代表了智能体框架设计的一个全新方向。LangGraph 将智能体的执行流程建模为一种状态机（State Machine），并将其表示为有向图（Directed Graph）。在这种范式中，图的节点（Nodes）代表一个具体的计算步骤（如调用 LLM、执行工具），而边（Edges）则定义了从一个节点到另一个节点的跳转逻辑。这种设计的革命性之处在于它天然支持循环，使得构建能够进行迭代、反思和自我修正的复杂智能体工作流变得前所未有的直观和简单。

LangGraph 的三个基本构成要素：
1. **全局状态（State）**： 整个图的执行过程都围绕一个共享的状态对象进行。这个状态通常被定义为一个 Python 的 `TypedDict`，它可以包含任何你需要追踪的信息，如对话历史、中间结果、迭代次数等。所有的节点都能读取和更新这个中心状态。

```python
from typing import TypedDict, List

# 定义全局状态的数据结构
class AgentState(TypedDict):
    messages: List[str]      # 对话历史
    current_task: str        # 当前任务
    final_answer: str        # 最终答案
    # ... 其他任何需要追踪的状态
```

2. **节点（Nodes）**：每个节点都是一个接收当前状态作为输入、并返回一个更新后的状态作为输出的 Python 函数。节点是执行具体工作的单元。

```python
# 定义一个“规划者“节点函数
def planner_node(state: AgentState) -> AgentState:
    """根据当前任务制定计划，并更新状态。"""
    current_task = state['current_task']
    # ... 调用 LLM 生成计划 ...
    plan = f"为任务 {current_task} 生成的计划..."

    # 将新消息追加到状态中
    state['messages'].append(plan)
    return state

# 定义一个“执行者“节点函数
def executor_node(state: AgentState) -> AgentState:
    """执行最新计划，并更新状态。"""
    latest_plan = state['messages'][-1]
    # ... 执行计划并获得结果 ...
    result = f"执行计划 '{latest_plan}' 的结果..."

    state['messages'].append(result)
    return state
```

3. **边（Edges）**：边负责连接节点。定义工作流的反向。最简单的边是常规边，它指定了一个节点的输出总是流向另一个固定的节点。而 LangGraph 最强大的功能在于**条件边（Conditional Edges）**。它通过一个函数来判断当前的状态，然后动态决定下一步应该跳转到哪个节点。这正是实现循环和复杂逻辑分支的关键。

```python
def should_continue(state: AgentState) -> str:
    """条件函数：根据状态决定下一步路由。"""
    # 假设如果消息少于 3 条，则需要继续规划
    if len(state['messages']) < 3:
        # 则返回的字符串需要与添加条件边时定义的键匹配
        return 'continue_to_planner'
    else:
        state['final_answer'] = state['messages'][-1]
        return 'end_workflow'
```

定义了状态、节点和边之后，就可以像搭积木一样将它们组装成一个可执行的工作流。

```python
from langgraph.graph import StateGraph, END

# 初始化一个状态图，并绑定我们定义的状态结构
workflow = StateGraph(AgentState)

# 将节点函数添加到图
workflow.add_node('planner', planner_node)
workflow.add_node('executor', executor_node)

# 设置图的入口
workflow.set_entry_point('planner')

# 添加常规边、连接 planner 和 executor
workflow.add_edge('planner', 'executor')

# 添加条件边，实现动态路由
workflow.add_conditional_edges(
    "executor", # 起始节点
    should_continue, # 判断函数
    { # 路由映射：将判断函数的返回值映射到目标节点
        "continue_to_planner": "planner", # 如果返回"continue_to_planner"，则跳回planner节点
        "end_workflow": END, # 如果返回"end_workflow"，则结束流程
    }
)

# 编译图，生成可执行的应用
app = workflow.compile()

# 运行图
inputs = {"current_task": "分析最近的 AI 行业新闻", "messages": []}
for event in app.stream(inputs):
    print(event)
```

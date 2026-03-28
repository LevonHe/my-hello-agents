import os
from ToolExecutor import ToolExecutor
from tools import search, get_search_description, get_search_name
from llm import HelloAgentsLLM
from ReActAgent import ReActAgent
from PlanAndSolveAgent import PlanAndSolveAgent
from ReflectionAgent import ReflectionAgent
from dotenv import load_dotenv

load_dotenv()

llm_client = HelloAgentsLLM()

toolExecutor = ToolExecutor()
toolExecutor.registerTool(get_search_name(), get_search_description(), search)

react_agent = ReActAgent(llm_client=llm_client, tool_executor=toolExecutor)
plan_and_solve_agent = PlanAndSolveAgent(llm_client=llm_client)
reflection_agent = ReflectionAgent(llm_client=llm_client)

def test_toolexecutor():
    print("\n--- 可用的工具 ---")
    print(toolExecutor.getAvailableTools())

    print("\n--- 执行 Action: Search['河南的省会是哪个城市？'] ---")

    tool_name = "Search"
    tool_input = "河南的省会是哪个城市？"
    tool_function = toolExecutor.getTool(tool_name)

    if tool_function:
        observation = tool_function(tool_input)
        print("--- 观察 (Observation) ---")
        print(observation)
    else:
        print(f"错误:未找到名为 '{tool_name}' 的工具。")

def test_react_agent():
    try:
        response_text = react_agent.run('2026年Apple最新手机型号及主要卖点。')
        if response_text:
            print(response_text)
    except ValueError as e:
        print(e)

def test_plan_and_solve_agent():
    try:
        response_text = plan_and_solve_agent.run('一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？')
        if response_text:
            print(response_text)
    except ValueError as e:
        print(e)

def test_reflection_agent():
    try:
        response_text = reflection_agent.run('编写一个Python函数，找出1到n之间所有的素数 (prime numbers)。')
        if response_text:
            print(response_text)
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    test_toolexecutor()
    # test_react_agent()
    # test_plan_and_solve_agent()
    # test_reflection_agent()

from typing import Callable, Dict, List
from tools import Tool
from agent.types import HistoryItem
import json
from agent.prompt import build_action_prompt, parse_action


def run_action(
    task: str,
    reflections: List[str],
    tools: Dict[str, Tool],
    llm: Callable[[str], str],
    max_steps: int = 6,
    verbose: bool = False,
) -> str:
    """Action：基于任务与 reflections，用 ReAct 生成输出（可含工具）。"""

    # history 每条为 (thought, action, action_input, observation)
    history: List[HistoryItem] = []

    # 最多循环max_steps次“思考=>行动=>观察”
    for step in range(1, max_steps + 1):
        # 构建上下文
        prompt = build_action_prompt(task, reflections, history, tools)
        if verbose:
            print(f"\n--- Action Step {step} ---")
            print(prompt[:800] + ("..." if len(prompt) > 800 else ""))

        # 调用模型，完成“思考”这个步骤
        out = llm(prompt)
        if verbose:
            print(out)

        # 由此可知每次只允许一个action
        action, action_input = parse_action(out)

        # 有合法工具可执行时，先执行工具；忽略同条回复里的 Final Answer
        if action and action in tools:
            # 截取思考过程出来
            thought = ""
            for line in out.splitlines():
                if line.startswith("Thought:"):
                    thought = line.split("Thought:", 1)[1].strip()
                    break
            # 执行action，获得observation
            obs = tools[action].run(action_input or "")
            # 打印执行结果-json格式
            _print_tool_execution(action, action_input or "", obs)
            # 把结果放回history中
            history.append((thought, action, action_input or "", obs))
            continue

        # 最终结果
        if "Final Answer:" in out:
            answer = out.split("Final Answer:", 1)[1].strip()
            if verbose:
                print(f">>> Final Answer: {answer}")
            return answer

        obs = f"ERROR: invalid Action. Must be one of {list(tools.keys())}."
        _print_tool_execution(action or "", action_input or "", obs)
        history.append(("", action or "", action_input or "", obs))

    return "Failed: max steps exceeded."


def _print_tool_execution(action: str, action_input: str, observation: str) -> None:
    print("\n=======>>> Tool execution ======\n")
    print(json.dumps(
        {"action": action, "input": action_input, "observation": observation},
        indent=2,
        ensure_ascii=False,
    ))

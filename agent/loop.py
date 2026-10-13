from typing import Callable, Dict
from tools.types import Tool
from agent.action import run_action
from agent.evaluator import evaluate
from agent.reflector import reflect, trim_reflections


def reflect_until_success(
    task: str,
    llm: Callable[[str], str],
    tools: Dict[str, Tool],
    max_trials: int = 3, # 最大尝试次数
    max_action_steps: int = 6,  # 最大动作步骤
    max_reflections: int = 5, # 最大反思次数
    verbose: bool = False, # 是否打印详细信息
) -> str:
    """
    Reflection 主循环：Action(ReActLoop) → Evaluation(判断结果是否成功) → Reflection(结果失败进行反思) → Retry(重新开始行动=>评估=>反思)。

    reflections = []
    repeat until success or max_trials:
        Action: 基于 task + reflections 生成输出
        Evaluation: score, feedback = Evaluator(output)
        if success: return output
        Reflection: reflections.append(Reflector(feedback))
    """
    reflections: list[str] = []
    last_output = ""

    for trial in range(1, max_trials + 1):
        if verbose:
            print(f"\n========== Trial {trial}/{max_trials} ==========")
            if reflections:
                print("Strategy memory:")
                for i, r in enumerate(reflections, 1):
                    print(f"  {i}. {r}")

        # ReAct Loop
        output = run_action(
            task,
            reflections,
            tools,
            llm,
            max_steps=max_action_steps,
            verbose=verbose,
        )
        if verbose:
            print(f"\n--- Evaluation ---")
            print(f"Output: {output}")
        # trial用完，评估处理结果还是不通过可以考虑返回当前output
        last_output = output

        # 评估处理结果
        result = evaluate(task, output, llm)
        if verbose:
            print(
                f"Success={result.success} Score={result.score}\n"
                f"Feedback: {result.feedback}"
            )
        if result.success:
            if verbose:
                print(f"\n>>> Success on trial {trial}")
            return output

        # 最后trial不会再运行action了，所以不需要再调用一次reflection了
        if trial == max_trials:
            break

        # 反思: 结果为什么失败
        reflection = reflect(
            task,
            output,
            result.feedback,
            result.score,
            reflections,
            llm,
        )
        reflections = trim_reflections(reflections + [reflection], max_reflections)
        if verbose:
            print(f"\n--- Reflection appended ---\n{reflection}")

    return last_output or "Failed: max trials exceeded."

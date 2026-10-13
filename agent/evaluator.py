from agent.types import EvaluationResult
from typing import Callable
from agent.prompt import build_evaluator_prompt, parse_evaluation


def _rule_check(task: str, output: str) -> EvaluationResult | None:
    """
    可验证任务的规则评估（示例：21+21 且统计答案字符串字符数）。
    命中规则时返回确定结果，否则返回 None 交给模型评估。
    """
    return None


def evaluate(
    task: str,
    output: str,
    llm: Callable[[str], str],
    use_rules: bool = True,
) -> EvaluationResult:
    """Evaluation：规则优先，否则由 LLM 给出成功标志、分数与批评。"""

    # 通过规则判定
    if use_rules:
        ruled = _rule_check(task, output)
        if ruled is not None:
            return ruled

    # 构建评估结果的上下文
    prompt = build_evaluator_prompt(task, output)
    # 调用模型进行判断
    raw = llm(prompt)
    # 提取模型判断结果
    success, score, feedback = parse_evaluation(raw)
    # 只要success了，score值设置为1.0
    if success:
        score = 1.0
    # 返回评估结果
    return EvaluationResult(success=success, score=score, feedback=feedback)

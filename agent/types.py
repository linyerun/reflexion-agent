from typing import Tuple
from dataclasses import dataclass


# history 每条为 (thought, action, action_input, observation)
HistoryItem = Tuple[str, str, str, str]


@dataclass
class EvaluationResult:
    """评估结果：二元成功标志 + 细粒度分数与反馈。"""
    success: bool
    score: float
    feedback: str

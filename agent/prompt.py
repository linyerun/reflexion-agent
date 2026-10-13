from typing import Dict, List, Tuple
from agent.types import HistoryItem
from tools import Tool


def build_action_prompt(
    task: str,
    reflections: List[str],
    history: List[HistoryItem],
    tools: Dict[str, Tool],
) -> str:
    tool_names = "[" + ", ".join(tools.keys()) + "]"
    tool_desc = "\n".join(f"- {t.name}: {t.description}" for t in tools.values())
    lines = [
        "You solve tasks with tools. Use EXACTLY this format each turn:",
        "Thought: ...",
        f"Action: one of {tool_names}",
        "Action Input: ...",
        "Stop when you can answer:",
        "Final Answer: ...",
        "",
        "IMPORTANT: Output only ONE turn per response. Do not include Final Answer until you have received Observation from a tool.",
        f"Tools available: {tool_names}",
        "Tool descriptions:",
        tool_desc,
        "",
        f"Task: {task}",
        "",
    ]
    if reflections:
        lines += ["", "Past reflections (strategy memory from failed attempts):"]
        for i, r in enumerate(reflections, 1):
            lines.append(f"{i}. {r}")
    for thought, action, action_input, obs in history:
        lines += [
            f"Thought: {thought}",
            f"Action: {action}",
            f"Action Input: {action_input}",
            f"Observation: {obs}",
            "",
        ]
    return "\n".join(lines)


def parse_action(text: str) -> Tuple[str | None, str | None]:
    action = None
    action_input = None
    for line in text.splitlines():
        if line.startswith("Action:"):
            action = line.split("Action:", 1)[1].strip()
        if line.startswith("Action Input:"):
            action_input = line.split("Action Input:", 1)[1].strip()
    return action, action_input


def build_evaluator_prompt(task: str, output: str) -> str:
    return "\n".join(
        [
            "You are an Evaluator. Judge whether the agent output fully satisfies the task.",
            "Reply in EXACTLY this format (no extra sections):",
            "Success: yes | no",
            "Score: <float 0.0 to 1.0>",
            "Feedback: <concise critique; what is wrong or missing>",
            "",
            f"Task: {task}",
            "",
            "Agent output:",
            output,
        ]
    )


def parse_evaluation(text: str) -> Tuple[bool, float, str]:
    success = False
    score = 0.0
    feedback = text.strip()
    for line in text.splitlines():
        lower = line.lower()
        if lower.startswith("success:"):
            val = line.split(":", 1)[1].strip().lower()
            success = val in ("yes", "true", "1")
        elif lower.startswith("score:"):
            try:
                score = float(line.split(":", 1)[1].strip())
            except ValueError:
                score = 0.0
        elif line.startswith("Feedback:"):
            feedback = line.split("Feedback:", 1)[1].strip()
    score = max(0.0, min(1.0, score))
    return success, score, feedback


def build_reflector_prompt(
    task: str,
    output: str,
    feedback: str,
    score: float,
    reflections: List[str],
) -> str:
    lines = [
        "You are a Reflector. Given evaluation feedback, write ONE short improvement",
        "strategy the agent should apply on the next attempt (max 120 words).",
        "Focus on: tool choice, format, constraints, missing steps. Do not repeat the answer.",
        "Reply with a single line:",
        "Reflection: <your strategy text>",
        "",
        f"Task: {task}",
        f"Score: {score}",
        f"Feedback: {feedback}",
        "",
        "Agent output:",
        output,
    ]
    if reflections:
        lines += ["", "Prior reflections (avoid duplicating):"]
        for i, r in enumerate(reflections, 1):
            lines.append(f"{i}. {r}")
    return "\n".join(lines)


MAX_REFLECTION_LEN = 400


def parse_reflection(text: str) -> str:
    text = text.strip()
    for line in text.splitlines():
        if line.startswith("Reflection:"):
            text = line.split("Reflection:", 1)[1].strip()
            break
    if len(text) > MAX_REFLECTION_LEN:
        text = text[:MAX_REFLECTION_LEN] + "..."
    return text

import json
from openai import OpenAI
from typing import Callable, Tuple


def print_message(message) -> None:
    """格式化打印单条消息（dict 或 ChatCompletionMessage），自动去掉 None 字段。"""
    if hasattr(message, "model_dump"):
        data = message.model_dump(exclude_none=True)
    elif isinstance(message, dict):
        data = {k: v for k, v in message.items() if v is not None}
    else:
        data = message
    print(json.dumps(data, indent=2, ensure_ascii=False))


def print_messages(messages: list) -> None:
    for i, msg in enumerate(messages):
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "?")
        print(f"--- [{i}] {role} ---")
        print_message(msg)


def create_llm(
        api_key: str,
        base_url: str,
        model: str,
        system_prompt: str = "You are a helpful assistant. Follow instructions precisely."
) -> Callable[[str], str]:
    client = OpenAI(api_key=api_key, base_url=base_url)

    def llm(prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        print("\n====>>> LLM request \n")
        print_messages(messages)

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
        )
        msg = response.choices[0].message

        print("\n====<<< LLM response\n")
        print_message(msg)

        print("\n\n")
        print(f"Model>\t {msg.content}")
        print("\n\n")
        return msg.content or ""

    return llm

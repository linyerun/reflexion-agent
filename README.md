###### 实现 Action → Evaluation → Reflection → Retry 工作流：失败时将反思写入策略记忆，跨尝试复用，形成可累积的改进线索。

# 目录结构
```text
reflexion-agent/
├── agent/
│   ├── action.py        # ReAct Loop With reflection
│   ├── evaluator.py     # 评估ReAct执行结果是否成功
│   ├── loop.py          # reflect_loop（核心）
│   ├── prompt.py        # ReAct/Evalution/Reflection 提示词+结果解析
│   ├── reflector.py     # 反思ReAct执行结果不正确的原因
│   └── types.py         # HistoryItem, EvaluationResult 类型
├── tools/
│   ├── calculator.py
│   ├── current_time.py
│   ├── word_count.py
│   └── types.py             # Tool 定义
├── .env.example
├── llm.py                   # 模型调用 + JSON 日志 + print_message / print_messages
├── main.py                  # 程序入口
└── requirements.txt
```

# 配置apikey
```bash
cp .env.example .env
```

# 运行
```bash
uv sync
```

```bash
uv run main.py
```

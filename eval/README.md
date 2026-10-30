# τ-bench 评测

用 [τ-bench](https://github.com/sierra-research/tau-bench)（MIT 协议）的 airline 领域评测翼枢的多 Agent 系统，并和 τ-bench 自带的单 Agent 基线对比。

## 安装（在项目的 Python 环境里，只需一次）

```bash
pip install litellm==1.106.0
pip install --no-deps "tau-bench @ git+https://github.com/sierra-research/tau-bench@59a200c6d575d595120f1cb70fea53cef0632f6b"
pytest agent/tests -q          # 确认装完后原有测试仍然通过（含 4 项评测链路测试）
```

- 用 `--no-deps` 是因为 tau-bench 声明的依赖里有 anthropic、mistralai、google-generativeai 等用不到的包；它实际只需要 litellm。
- litellm 会把 openai 包降到 2.x，langchain-openai 1.x 支持这个版本（`openai>=2.45,<4`）。
- 固定到具体 commit，保证评测代码可复现。

## 跑分

在项目根目录执行，需要 `agent/.env` 里的 `DASHSCOPE_API_KEY`，**不需要启动任何数据库**（τ-bench 用自己的内存数据）。

```bash
python eval/run_eval.py --agent multi  --task-ids 0 1 2 3 4     # 翼枢多 Agent，先跑 5 题估算花费
python eval/run_eval.py --agent single --task-ids 0 1 2 3 4     # 单 Agent 基线，同样 5 题
python eval/run_eval.py --agent multi  --start 0 --end 50       # 全量 50 题
python eval/run_eval.py --agent multi  --lang zh --start 0 --end 50   # 中文任务
python eval/run_eval.py --agent multi  --task-ids 0 1 --trials 3      # 每题跑 3 次，计算 pass^1..pass^3
```

结果写在 `eval/results/tau_<时间>_<agent>_<lang>_<model>.json`：`summary` 是汇总，`rows` 是逐题记录（奖励、对话轮数、实际调用的工具、标准答案动作、每轮路由、完整对话、token 用量），用于失败分析。

## 设计说明

| 项目 | 做法 |
|---|---|
| 被测模型 | 两种 Agent 都用 qwen-plus，temperature 0 |
| 模拟用户 | qwen-plus（τ-bench 原版提示词）。**模拟用户的模型会影响成绩**，报告分数时要注明 |
| 单 Agent 基线 | `tau_bench.agents.tool_calling_agent.ToolCallingAgent`，官方代码，英文原版政策 wiki 做系统提示词 |
| 多 Agent | `eval/wingdesk_agent.py`：orchestrator + 5 个专业 Agent，`mode="eval"` |
| 工具 | 两者都用 τ-bench 原版 14 个工具，参数原样交给 `env.step`（`TauBenchBackend.call`） |
| 评测模式与产品模式的差异 | 没有确认卡片（写操作前在对话里得到 yes）；工具带 user_id、没有 UserIdInjector；不用 Neo4j / RAG / 缓存 / 记忆，政策直接注入提示词 |
| 步数上限 | 每题最多 30 个动作（工具调用 + 回复），与 τ-bench 默认一致；多 Agent 在两轮对话之间检查 |
| 中文任务 | `eval/tasks_zh.json`：50 条 instruction 的中文译本，ID / 金额 / 日期不变；评分（actions、outputs）与英文完全相同；模拟用户被要求全程说中文 |
| 花费 | token 数为实测；金额按 `--price-in/--price-out`（元/百万 token）估算，以百炼控制台账单为准 |

## 评分规则（τ-bench 原版）

一题 reward 为 1 需要同时满足：
1. 对话结束时，τ-bench 数据库的哈希值与"按标准答案动作执行后"的哈希值一致（写操作完全正确，多做、少做、参数不同都算错）；
2. 如果任务要求说出某些信息（outputs，如总金额 23553），回复里必须出现这些内容。

对话结束的条件：模拟用户输出 `###STOP###`，或 Agent 调用 `transfer_to_human_agents`，或达到步数上限（此时 reward 为 0）。

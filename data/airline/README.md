# 航空数据说明

## 来源

本目录的业务数据来自 [τ-bench](https://github.com/sierra-research/tau-bench)（Sierra，MIT 协议）的 airline 领域，
原始路径 `tau_bench/envs/airline/`，取自提交 `59a200c`。版权声明见 `LICENSE-tau-bench`。

| 文件 | 说明 | 是否改动 |
|---|---|---|
| `users.json` | 500 个用户 | 原样 |
| `flights.json` | 300 个航班 × 30 天 = 9000 条航班实例 | 原样 |
| `reservations.json` | 2000 个订单 | 原样 |
| `wiki.md` | 航空客服政策（英文原文） | 原样 |
| `tasks_test.py` | 50 个评测任务 | 原样 |

## 本项目补充的文件

| 文件 | 说明 |
|---|---|
| `airports_zh.json` | 20 个机场的中文城市名、中英文机场全称 |
| `enums_zh.json` | 舱位、行程类型、会员等级、航班状态、支付方式等枚举的中文对照（只用于展示，不改数据） |
| `policy/core/` | 7 篇核心政策：`wiki.md` 的中文译本，按主题拆分，规则与原文逐条对应 |
| `policy/extended/` | 18 篇**仿真扩充政策**：为演示 RAG 生成的虚构政策，覆盖 τ-bench 没有的主题（随身行李、宠物、特殊旅客、值机等），**不是任何真实航空公司的规定**；与核心政策冲突时以核心政策为准 |

## 导入

在项目根目录依次执行：

```bash
python agent/database/import_airline.py            # MySQL：airline 库，10 张表
python agent/database/check_airline_roundtrip.py   # 校验 MySQL 数据能还原成原版 JSON
python agent/database/import_airline_graph.py      # Neo4j：20 个机场 + 300 条航线
python agent/database/import_airline_policy.py     # Milvus：政策文档切分后入库
```

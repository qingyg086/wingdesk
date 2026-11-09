# 对话追踪与故障定位

每次 `/api/chat` 请求生成独立 `trace_id`。HTTP 响应头 `X-Trace-ID`、SSE 末尾 `meta.trace_id`、网页“处理详情”里的追踪编号，以及后端和 MCP 服务的 JSON 日志使用同一值。拒绝并发请求也有单独的编号。浏览器中途断开后，后台继续完成这一轮，仍记录 `turn.finished`；若 SSE 未到达末尾，前端仍从响应头保存编号；跨域访问时该响应头已对浏览器开放。

追踪事件写到进程的标准错误输出，每个事件是一行 JSON；启动信息等旧日志仍保留原格式。排查时先用编号筛选，再按时间看 `turn.started`、`cache.lookup`、`route.selected`、`model.finished`、`policy.retrieval`、`policy.decided`、`tool.finished` / `mcp.finished` / `mcp.db.finished`、`turn.finished`。失败时分别出现 `model.failed`、`policy.decision_failed`、`mcp.failed` 或 `turn.graph_failed`，只记录异常类型，不打印异常正文。MCP 客户端通过请求 `_meta.trace_id` 将编号传给 stdio 子进程；子进程在数据库操作日志中读取同一编号。连接重试按 `attempt` 区分。工具参数和结果、旅客提问、回答、订单号、用户 ID 不写入这些追踪事件。

`turn.finished` 记录路线、缓存命中情况、总耗时、模型及工具调用数、输入/输出 token 数。模型日志记录每次调用耗时与 token 数。政策检索日志记录模式、候选数量、是否降级和耗时。SSE 的 `timing` 另有缓存查询时间、总控节点完成前的时间、首字和总耗时。路由时间包含少量图调度开销，不等于单纯模型耗时。模型提供商没有返回用量时，token 字段为 `null` 且 `token_usage_complete=false`，不会把未知当作 0。

本地查看：启动后端并发送一条对话；在网页“处理详情”复制追踪编号，或查看 HTTP 响应头。若后端日志保存为 `server.log`，用 `rg '编号' server.log` 找这一轮；MCP 进程写到父进程的 stderr，能用同一编号查到其数据库调用。JSON 行也可以交给日志系统按 `trace_id` 聚合。暂未接 Langfuse；当前实现以现有日志和 SSE 为准。

验证：自动测试覆盖缓存命中、会话忙碌、浏览器断线、模型用量与故障、MCP 元数据传递、数据库操作日志，以及敏感正文不会进入追踪事件。真实只读 `list_all_airports` 调用中，客户端 `mcp.finished` 与子进程 `mcp.db.finished` 的 `trace_id` 一致。真实政策对话的路由、检索、两次模型判断、最终回复也共享编号。以上为本机单轮验证，不是并发负载或线上日志平台的验收。

## 本机验证结果（2026-11-09）

- 后端全量回归：`POLICY_INTEGRATION_TEST=1 pytest agent/tests -q`，170 项通过；之后仅调整追踪阶段标签，29 项政策/追踪定向回归通过；新增的并发隔离测试所在文件 7 项通过。前端 7 项测试通过，`npm run build` 通过。
- 真实只读 MCP 调用 `list_all_airports`：客户端 `mcp.finished` 和子进程 `mcp.db.finished` 记录相同的 32 位追踪编号，工具返回非空。
- 真实政策咨询：总控路由 931 ms，政策检索 2348 ms，政策选材 902 ms，政策复核 879 ms；SSE 中的路由时间 941 ms，总耗时 5089 ms。3 次模型调用合计输入 2401、输出 56 token，并记录在同一追踪编号下。这是一次本机样例，不是延迟基准或成本承诺。

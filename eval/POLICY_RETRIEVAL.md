# 政策检索评测

## 题集与方法

`policy_retrieval_cases.json` 在首次检索前建立：48 条人工中文题，36 条有明确答案、8 条知识库无法完整回答、4 条需要澄清。每题有文档/小节标注和预期行为，重复记载同一事实的小节可作为等价证据。32 条开发题用于方案比较，16 条留出题在选择方案后验收。题目属于定向回归测试，不代表线上流量；标注来自项目政策，其中扩充政策为虚构演示资料。

沿用原导入脚本的切分、上下文标题和 `text-embedding-v2`，25 份文档得到 107 个片段。文档使用 `embed_documents`，问题使用 `embed_query`。在独立 Milvus 2.5.7 集合重建原 HNSW/COSINE 参数进行基线测试，不更改正式旧集合。混合方案采用 Milvus 原生 BM25（jieba 分词、小写过滤）与 dense 各召回 10 条，RRF 的 k=60；重排对融合后的 10 条调用 `qwen3-rerank`。这些参数在开发集比较前固定。

指标是有答案题中至少一份有效证据排在前 k 名的比例（Hit@1/3），以及前 10 条的首个有效证据倒数排名均值（MRR@10）。它们不等于最终回答准确率。无法回答/需要澄清的题单独检查回答边界，不能按“检索出了相似文档”记为答对。

耗时包含真实问题向量请求、Milvus 查询以及可选重排请求；连接先预热，排除建库时间。四种方案按题目轮换执行顺序，各题一次，返回前 10 条。样本小、无并发，外部 API 波动会影响结果，不能当作线上 SLA。

选择规则：开发集 Hit@3 不低于纯向量基线、检索 p50 不超过 1.5 秒时，优先选择 Hit@1 更高的方案；准确率持平时优先选更简单、耗时更低的方案。确认选择后再查看留出集，留出集不用于调参。

## 复现

```bash
python agent/scripts/bench_policy_retrieval.py --split dev --modes vector --output eval/results/my_baseline.json
python agent/scripts/bench_policy_retrieval.py --split dev --modes vector bm25 hybrid hybrid_rerank --output eval/results/my_compare.json
```

使用 `agent/.env` 中的 DashScope 和 Milvus 配置；脚本创建并删除独立临时集合，结果文件禁止覆盖。原始结果记录题集/语料/实现哈希、参数、逐题排名和耗时。

方法依据：[Milvus 2.5 全文检索](https://milvus.io/docs/v2.5.x/full-text-search.md)、[中文分词](https://milvus.io/docs/v2.5.x/chinese-analyzer.md)、[DashScope 文本排序 API](https://help.aliyun.com/zh/model-studio/text-rerank-api)。重排分数用于本次候选排序，不把它当作跨问题通用的“有答案概率”。

## 实测结果（2026-11-08）

开发集包含 24 条可回答题和 8 条边界题。首次比较后发现 4 条开发题漏标了其他政策中重复记载的同一事实，补全了等价证据。原始结果保留，见 [标注修订记录](policy_retrieval_label_audit.json)；下表按修订标注重新计算，未重新生成排名、未改变耗时。留出题的标注没有修改。原标注下，混合重排的 Hit@1 为 22/24，仍高于纯向量的 21/24。

| 开发集方法 | Hit@1 | Hit@3 | MRR@10 | p50 / p95 |
|---|---:|---:|---:|---:|
| 纯向量 | 21/24 | 24/24 | 0.924 | 529 / 594 ms |
| BM25 | 13/24 | 19/24 | 0.688 | 350 / 409 ms |
| RRF 混合 | 21/24 | 21/24 | 0.906 | 598 / 610 ms |
| 混合 + qwen3-rerank | 24/24 | 24/24 | 1.000 | 797 / 916 ms |

原始：[首次纯向量基线](results/20261108_retrieval_baseline.json)、[四种方法对比](results/20261108_retrieval_compare_dev.json)；补全标注后的[基线](results/20261108_retrieval_baseline_labels_v2.json)和[对比](results/20261108_retrieval_compare_dev_labels_v2.json)。BM25 单独使用和不带重排的混合没有优于当前基线，因此没有直接将它们设为默认。

查看留出结果前，已在 [选择记录](policy_retrieval_selection.json) 中固定使用 `hybrid_rerank`。留出集为 12 条可回答题、2 条无法回答题、2 条需澄清题：

| 留出集方法 | Hit@1 | Hit@3 | MRR@10 | p50 / p95 |
|---|---:|---:|---:|---:|
| 纯向量 | 7/12 | 11/12 | 0.743 | 541 / 604 ms |
| 混合 + qwen3-rerank | 10/12 | 12/12 | 0.889 | 797 / 1648 ms |

[留出原始记录](results/20261108_retrieval_holdout.json)。Hit@1/3 未低于基线、p50 低于 1.5 秒，达到预先设定的验收条件。首条结果仍有两题未命中：婴儿单独占座、查询他人行程，正确条款均在第 3 条。重排会增加网络请求和尾部延迟；不能把这一小批题的 100% Hit@3 说成真实用户回答正确率。

```bash
python agent/scripts/bench_policy_retrieval.py --split holdout --modes vector hybrid_rerank --output eval/results/my_holdout.json
```

## 产品接入与维护

政策工具现在默认使用 `hybrid_rerank`，返回前 3 条带文档、小节及核心/仿真标记的候选证据。重排服务失败时退回开发集表现更好的纯向量路径，并记录降级日志；索引不可用时返回明确的“暂时无法核实”。相似度或重排分数不直接决定“能不能回答”，政策 Agent 必须核对条款是否覆盖具体问题；缺少用户条件就追问，未写明的信息说明无法确认，部分有依据时只回答该部分。

```bash
python agent/database/import_airline_policy.py
```

默认导入按语料内容生成新集合 `airline_policy_hybrid_v1_<语料哈希前12位>`，旧集合保留。再次导入同样内容会校验版本和条数并复用；新内容需重新导入并重启后端。Milvus 版本至少需要支持 BM25 和 jieba analyzer（本次实测 2.5.7）。现有纯向量集合 `airline_policy_docs` 不会被删除。默认精确问答缓存与这条 RAG 链路独立。

如果政策发生变化，还需要使缓存答案与新政策一致：运行导入时加 `--invalidate-cache`，更新 `data/airline/cache/preset_qa.json` 后再运行 `python app/preload_cache.py`。一般索引升级不清空已有缓存。

`agent/.env` 可设置 `POLICY_RETRIEVAL_MODE=vector` 回退纯向量；重排模型默认 `POLICY_RERANK_MODEL=qwen3-rerank`。模式还支持 `bm25` 和 `hybrid` 供对照实验。修改配置后重启后端。完整性校验会阻止读取版本不符或未完成入库的集合。

## 集成验证

首次真实 Agent 试跑发现 Milvus 在传入 schema 时忽略了外层 `create_collection(description=...)` 参数，导致索引被版本检查拒绝。已将描述写入 schema 并新增真实服务器回归测试，检查创建、重复导入和重新连接查询。[首次失败记录](results/20261108_policy_answers_initial_unavailable.json)保留用于复核。该修正不改变文档切分、检索算法或排名参数。

运行本地测试：

```bash
pytest agent/tests/test_policy_retrieval.py -q
POLICY_INTEGRATION_TEST=1 pytest agent/tests/test_policy_retrieval.py -q
python agent/scripts/bench_policy_answers.py --output eval/results/my_policy_answers.json
```

第一条使用可控依赖检查标签、故障提示和回退行为；第二条额外在独立临时 Milvus 集合验证真实 schema/中文 BM25。对话脚本实际调用产品政策 Agent、检索工具和模型，覆盖 6 条正常问题、8 条未知问题、4 条需澄清问题，保留完整回答供人工复核。它不经过总控路由或问答缓存，不会办理业务。首次失败后修正了 prompt 中“每题必须先检索”与“指代不明应先追问”的冲突；边界问法的实测是提示词回归检查，不作为全新盲测成绩。


## 政策回答收尾（2026-11-09）

产品模式的政策节点改为专用证据子图，τ-bench 模式保持原路径：

1. 用原问题检索；用户回答上一轮追问时，拼接最近连续的澄清上下文（最多四轮）。
2. 模型只输出严格 JSON：可答、部分可答、资料不足、需追问或实时业务查询，以及所选小节下标。
3. 可答、部分可答、追问再由独立模型调用核对，审核不通过不输出条款。审核与选材使用同一模型，不视为统计独立保障。
4. 程序校验下标、结构和引用；回复仅包含完整小节原文及固定模板，不允许模型自由补写金额、许可、法规、解释或承诺。完整原文保留否定词和适用条件，代价是有时比自然语言概括更长。
5. 无资料、查询服务不可用、模型判断失败分别提示；实际订单和退款状态不由政策资料推断。政策咨询不执行业务写入；转人工仍需用户明确提出。
6. 私有选择/审核 token 不向前端转发；最终经校验的政策回复只输出一次，同时写入会话历史。政策回答需等待审核完成，不再逐字流式输出；其他专员不受影响。

边界验收覆盖有答案、部分可答、政策未记载、追问、多轮补充、诱导凭常识回答，以及规则咨询与办理请求的路由区别。`bench_policy_answers.py` 现在发现检查失败会以非零状态退出，处理故障不能算成正常“无法确认”。`bench_policy_boundaries.py` 经过真实路由、检索、模型、公共流式 runner 与 MemorySaver，绕过缓存且不注册业务工具；因此不声称是完整网页/数据库写入验收。

```bash
python agent/scripts/bench_policy_answers.py --output eval/results/my_policy_answers.json
python agent/scripts/bench_policy_boundaries.py --output eval/results/my_policy_boundaries.json
POLICY_INTEGRATION_TEST=1 pytest agent/tests -q
```

保留每轮原始结果，不覆盖失败记录。旧的 18 条题是开发回归；新增 12 组边界题首次通过 9/12，发现封闭范围条款过度拒答、部分可答的格式问题，以及减少行李的规则咨询被路由到业务专员。修复后的重复运行属于回归，不能再称为盲测。一次 JSON 模式兼容错误也保留在中间结果中；已修正审核提示并将故障回答从通过条件中排除。

适用范围：确定性渲染能限制资料外新增事实，但模型仍可能选错小节、误判是否充分或过度拒答；不能保证零错误。默认精确缓存仍可直接回答，它不经过此审核，预置答案需独立维护。演示政策不是现实航空公司的规定。本轮不改变检索算法参数、线上缓存开关或业务专员的规则体系。

### 最终验证记录

- 最终代码全量回归：`POLICY_INTEGRATION_TEST=1 pytest agent/tests -q`，**164 passed**，80.02 秒。包含真实 Milvus 临时索引、政策故障降级、引用与下标校验、私有输出不泄露、单次回复及会话历史、路由与业务流程回归。
- [18 条直接政策 Agent 回归](results/20261109_policy_answers_bounded.json)：**18/18**，有答案 6/6、资料不足 8/8、必要追问 4/4。已逐条复核最终正文；部分额度题展示整张表，会比针对性概括更长。端到端中位耗时 **1811 ms**，最大 **2851 ms**（包含模型、检索及审核；不含路由/缓存）。单轮样本有明显外部 API 波动，不是生产延迟承诺。

- [12 组完整图边界回归](results/20261109_policy_boundary_bounded.json)：**12/12**，共 14 轮（含两组连续追问）。已复核完整回答和结构化审核记录；所有问题均进入政策 Agent。每轮含路由的中位耗时 **2657.5 ms**，最大 **3912 ms**。结果中的实现哈希与当前代码核对一致。
- 一次复测中模型连接持续不结束，人工中止了该不完整运行，不将已输出的通过项计为整套通过。最终对政策选材和审核分别增加 **30 秒总时限**及 **256 token 输出上限**；超时取消模型调用并返回核实失败提示。测试覆盖取消行为。此限制不等于整个网页请求 30 秒超时，检索、路由与两次模型调用各有耗时。
- 新边界脚本逐题另存 `.progress.jsonl`，即使后续中断也保留已完成题；整套通过必须以最终 JSON 及成功退出为准。最终报告对应 `_bounded.json`，较早文件仅用于保留开发与故障过程。

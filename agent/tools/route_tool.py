"""
航线查询工具（Neo4j）：查两个机场之间有哪些航线结构（直飞、一次中转、两次中转）。

为什么不用 GraphCypherQAChain（让大模型现场写 Cypher）：慢（多一次 LLM 调用）、不稳定（可能写错语法、
查不存在的属性），还有注入风险。这里用固定的参数化 Cypher：图结构就两种元素（Airport 节点、FLIGHT 边），查询模式是固定的，
  没必要让模型生成。参数化查询也天然防注入。

定位：图里只有"航线结构"（航班号 + 计划时刻），没有当天的余座、价格和状态。
它用来回答"A 到 B 有哪些走法、经哪里转"，要订票还得用 search_direct_flight / search_onestop_flight 查当天情况。
两次中转是 τ-bench 原版工具没有的能力，属于图数据库擅长的多跳路径查询。
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

_driver = None

# 可变长度路径：1~3 段航班。约束：
#  1) 每段的到达时间 <= 下一段的出发时间（与 τ-bench 原版一致，直接比较时刻字符串，含 +1 的次日到达也照此比较）；
#  2) 不经过重复机场（不绕圈）。
ROUTE_QUERY = """
MATCH p = (a:Airport {code: $origin})-[:FLIGHT*1..3]->(b:Airport {code: $destination})
WITH p, relationships(p) AS rs, nodes(p) AS ns
WHERE size(rs) <= $max_legs
  AND ALL(i IN range(0, size(rs) - 2)
          WHERE rs[i].scheduled_arrival_time_est <= rs[i + 1].scheduled_departure_time_est)
  AND ALL(n IN ns WHERE single(m IN ns WHERE m = n))
RETURN size(rs) - 1 AS stops,
       [n IN ns | n.code] AS airports,
       [r IN rs | r.flight_number] AS flights,
       rs[0].scheduled_departure_time_est AS departure,
       rs[size(rs) - 1].scheduled_arrival_time_est AS arrival
ORDER BY stops, departure
LIMIT $limit
"""


def _get_driver():
    global _driver
    if _driver is None:
        from neo4j import GraphDatabase
        _driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "")),
        )
    return _driver


@tool
def query_flight_routes(origin: str, destination: str, max_stops: int = 1) -> str:
    """查询两个机场之间的航线走法（直飞 / 中转），返回航班号、经停机场和计划起降时刻（美东时间）。
    只反映航线结构，不含某一天的余座、价格和航班状态；要订票或查某天能不能订，请再用
    search_direct_flight / search_onestop_flight 按日期查询。
    origin / destination 为三字机场代码（如 JFK、SEA）；max_stops 为最多中转次数（0、1 或 2）。"""
    max_stops = max(0, min(int(max_stops), 2))
    try:
        with _get_driver().session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as s:
            rows = s.run(ROUTE_QUERY, origin=origin.upper(), destination=destination.upper(),
                         max_legs=max_stops + 1, limit=30).data()
    except Exception as e:  # 图数据库不可用时降级，而不是让整个 Agent 报错
        return (f"航线图查询暂时不可用（{type(e).__name__}）。"
                "请改用 search_direct_flight / search_onestop_flight 按日期查询。")
    if not rows:
        return json.dumps({"routes": [], "note": f"{origin}→{destination} 在 {max_stops} 次中转以内没有可衔接的航线"},
                          ensure_ascii=False)
    return json.dumps({"routes": rows}, ensure_ascii=False)

"""
把 τ-bench airline 的航线网络导入 Neo4j（确定性脚本生成，不调用大模型）。

用法（在项目根目录）：
    python agent/database/import_airline_graph.py

图模型：
    (:Airport {code, city, city_zh, name_zh})                     20 个
    (:Airport)-[:FLIGHT {...}]->(:Airport)                         300 条，一个航班号一条边
    FLIGHT 的属性：
        flight_number
        scheduled_departure_time_est   'HH:MM:SS'，与 τ-bench 原样一致
        scheduled_arrival_time_est     'HH:MM:SS' 或 'HH:MM:SS+1'，与 τ-bench 原样一致
        overnight                      到达时间是否带 +1（次日到达）
        duration_min                   计划飞行时长（分钟）

注意：
- 图里只放"航线结构"（静态，很少变），不放每天的余座、价格、状态（每天都变，放在 MySQL）。
  查中转时先在图里找路径，再去 MySQL 查当天是否 available。
- 脚本会先清空整个 Neo4j 库（MATCH (n) DETACH DELETE n）。
- 可重复执行。
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "airline"
load_dotenv(ROOT / "agent" / ".env")


def to_min(t: str) -> int:
    h, m, s = map(int, t.replace("+1", "").split(":"))
    return h * 60 + m + (1440 if "+1" in t else 0)


def build():
    airports = json.loads((DATA / "airports_zh.json").read_text(encoding="utf-8"))
    flights = json.loads((DATA / "flights.json").read_text(encoding="utf-8"))
    nodes = [{"code": c, "city": a["city"], "city_zh": a["city_zh"], "name_zh": a["name_zh"]}
             for c, a in airports.items() if not c.startswith("_")]
    edges = []
    for fn, f in flights.items():
        dep, arr = f["scheduled_departure_time_est"], f["scheduled_arrival_time_est"]
        edges.append({
            "flight_number": fn, "origin": f["origin"], "destination": f["destination"],
            "scheduled_departure_time_est": dep, "scheduled_arrival_time_est": arr,
            "overnight": "+1" in arr, "duration_min": to_min(arr) - to_min(dep),
        })
    return nodes, edges


# 一次中转（与 τ-bench search_onestop_flight 的判断一致：
# 第一段到达时间的原始字符串 <= 第二段出发时间，才算能接上）
ONESTOP_QUERY = """
MATCH (a:Airport {code: $origin})-[f1:FLIGHT]->(m:Airport)-[f2:FLIGHT]->(b:Airport {code: $destination})
WHERE f1.scheduled_arrival_time_est <= f2.scheduled_departure_time_est
RETURN f1.flight_number AS first, m.code AS via, f2.flight_number AS second,
       f1.scheduled_departure_time_est AS dep, f2.scheduled_arrival_time_est AS arr, f1.overnight AS overnight
ORDER BY dep
"""


def main():
    nodes, edges = build()
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "")),
    )
    db = os.getenv("NEO4J_DATABASE", "neo4j")
    with driver.session(database=db) as s:
        old = s.run("MATCH (n) RETURN count(n) AS n").single()["n"]
        print(f"清空 Neo4j 现有数据（{old} 个节点）…")
        # 分批删除，避免节点多时单个事务太大
        while s.run("MATCH (n) WITH n LIMIT 5000 DETACH DELETE n RETURN count(*) AS c").single()["c"]:
            pass

        s.run("CREATE CONSTRAINT airport_code IF NOT EXISTS FOR (a:Airport) REQUIRE a.code IS UNIQUE")
        s.run("UNWIND $rows AS r "
              "CREATE (:Airport {code: r.code, city: r.city, city_zh: r.city_zh, name_zh: r.name_zh})",
              rows=nodes)
        s.run("""
            UNWIND $rows AS r
            MATCH (a:Airport {code: r.origin}), (b:Airport {code: r.destination})
            CREATE (a)-[:FLIGHT {
                flight_number: r.flight_number,
                scheduled_departure_time_est: r.scheduled_departure_time_est,
                scheduled_arrival_time_est: r.scheduled_arrival_time_est,
                overnight: r.overnight,
                duration_min: r.duration_min
            }]->(b)
        """, rows=edges)

        n_nodes = s.run("MATCH (a:Airport) RETURN count(a) AS n").single()["n"]
        n_edges = s.run("MATCH ()-[f:FLIGHT]->() RETURN count(f) AS n").single()["n"]
        n_routes = s.run("MATCH (a:Airport)-[:FLIGHT]->(b:Airport) "
                         "RETURN count(DISTINCT [a.code, b.code]) AS n").single()["n"]
        ok = (n_nodes, n_edges, n_routes) == (20, 300, 108)
        print(f"{'[OK]' if n_nodes == 20 else '[FAIL]'} Airport 节点：{n_nodes} / 期望 20")
        print(f"{'[OK]' if n_edges == 300 else '[FAIL]'} FLIGHT 关系：{n_edges} / 期望 300")
        print(f"{'[OK]' if n_routes == 108 else '[FAIL]'} 航线（去重的起降机场对）：{n_routes} / 期望 108")

        rows = s.run(ONESTOP_QUERY, origin="JFK", destination="SEA").data()
        print(f"示例：JFK→SEA 一次中转的航班组合 {len(rows)} 种，前 3 种：")
        for r in rows[:3]:
            print(f"   {r['first']} → {r['via']} → {r['second']}  "
                  f"{r['dep']} 出发，{r['arr']} 到达{'（第一段次日到达）' if r['overnight'] else ''}")
    driver.close()
    print("图谱导入完成" if ok else "[FAIL] 核对未通过")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

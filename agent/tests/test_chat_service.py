"""
chat_service 的 SSE 协议测试：
- 待确认操作要推 confirm 事件；下一次请求带 confirm=true（前端按钮）即恢复执行；
- 流式输出：回复是逐 token 推的（多条 content 事件），节点 / 工具步骤在文字之前推出，meta 和 done 在最后。
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app"))

from test_graph import DATE, MIA, R, World, say, tool_call   # noqa: E402
from core.workflow.runner import make_config  # noqa: E402


def parse(events):
    return [json.loads(e[len("data: "):]) for e in events]


def meta_of(ev):
    return next(e["meta"] for e in ev if "meta" in e)


def text_of(ev):
    return "".join(e.get("content", "") for e in ev)


async def collect(gen):
    return parse([e async for e in gen])


@pytest.mark.asyncio
async def test_sse_confirm_event_and_resume(rw):
    from service import chat_service as cs      # semantic_cache 未初始化时 get_cache 直接返回 None

    async with World() as w:
        cs.graph, cs.memory = w.graph, None
        w.llm.routes = [R("booking_agent")]
        w.llm.script = [w.booking_call(), say("预订成功！")]
        before = w.reservations(MIA)

        ev = await collect(cs.stream_chat("就订这个", MIA, "web1"))
        assert meta_of(ev)["route"] == ["orchestrator", "booking_agent"]
        assert [e["step"]["key"] for e in ev if e.get("step", {}).get("kind") == "node"] == \
            ["orchestrator", "booking_agent"]
        assert [e["step"]["name"] for e in ev if e.get("step", {}).get("kind") == "tool"] == ["book_reservation"]
        assert next(e["confirm"] for e in ev if "confirm" in e)["action"] == "book_reservation"
        assert "请确认以下操作" in text_of(ev)
        assert ev[-2].keys() == {"meta"} and ev[-1] == {"done": True}
        assert w.reservations(MIA) == before

        ev = await collect(cs.stream_chat("", MIA, "web1", confirm=True))     # 前端点【确认】
        assert meta_of(ev)["route"] == ["confirm", "booking_agent"]
        assert text_of(ev) == "预订成功！"
        assert len(w.reservations(MIA)) == len(before) + 1


@pytest.mark.asyncio
async def test_tokens_stream_incrementally(rw):
    """真流式：一句回复拆成多条 content 事件，且第一条 content 早于 meta；timing 里记录首字和总耗时。"""
    from service import chat_service as cs

    async with World() as w:
        cs.graph, cs.memory = w.graph, None
        w.llm.routes = [R("policy_agent")]
        w.llm.script = [say("经济舱可免费托运一件行李。")]
        ev = await collect(cs.stream_chat("经济舱能托运几件", MIA, "s1"))
        contents = [i for i, e in enumerate(ev) if "content" in e]
        assert len(contents) > 1                                   # 不是一次性整段
        assert text_of(ev) == "经济舱可免费托运一件行李。"
        steps = [i for i, e in enumerate(ev) if "step" in e]
        assert max(steps) < contents[0]                            # 路由步骤先于文字
        timing = meta_of(ev)["timing"]
        assert 0 <= timing["first_token_ms"] <= timing["total_ms"]


@pytest.mark.asyncio
async def test_client_disconnect_does_not_cut_the_turn(rw):
    """前端中途断开（生成器被关闭）后，后台任务仍把这一轮跑完：历史以最终回复结尾，没有挂起的步骤。"""
    import asyncio
    from langchain_core.messages import AIMessage as AIMsg
    from service import chat_service as cs

    async with World() as w:
        cs.graph, cs.memory = w.graph, None
        w.llm.routes = [R("flight_search_agent")]
        w.llm.script = [tool_call("search_direct_flight", origin="JFK", destination="SEA", date=DATE),
                        say("找到 3 个直飞航班。")]
        gen = cs.stream_chat("查一下航班", MIA, "d1")
        first = json.loads((await gen.__anext__())[len("data: "):])
        assert first["step"]["key"] == "orchestrator"
        await gen.aclose()                                         # 模拟浏览器关闭
        await asyncio.gather(*list(cs._background_tasks))
        state = await w.graph.aget_state(make_config(MIA, "d1"))
        assert state.next == ()
        last = state.values["messages"][-1]
        assert isinstance(last, AIMsg) and last.content == "找到 3 个直飞航班。"


@pytest.mark.asyncio
async def test_error_is_reported_and_stream_closes(rw, monkeypatch):
    from service import chat_service as cs

    async def boom(*a, **k):
        raise RuntimeError("model down")
        yield  # noqa

    async with World() as w:
        cs.graph, cs.memory = w.graph, None
        monkeypatch.setattr(cs, "stream_turn", boom)
        ev = await collect(cs.stream_chat("你好", MIA, "e1"))
        assert any("error" in e for e in ev)
        assert ev[-1] == {"done": True}


class FakeCache:
    """固定命中的缓存：模拟"退款多久到账"命中预置问答。"""
    available = True

    async def get_cache(self, query, user_id):
        if "到账" in query:
            return {"answer": "退款将在 5 至 7 个工作日内原路退回。", "matched_question": "退款多久能到账？",
                    "level": "L1_SEMANTIC", "similarity": 0.95}
        return None


@pytest.mark.asyncio
async def test_cache_hit_is_recorded_in_history(rw, monkeypatch):
    """缓存命中时图没有运行；这一问一答要补写进 checkpointer，下一轮 Agent 才能看到。"""
    from langchain_core.messages import AIMessage as AIMsg
    from service import chat_service as cs

    async with World() as w:
        cs.graph, cs.memory = w.graph, None
        monkeypatch.setattr(cs, "semantic_cache", FakeCache())
        ev = await collect(cs.stream_chat("退款多久能到账", MIA, "c1"))
        assert meta_of(ev)["cache"] == "L1_SEMANTIC"
        assert text_of(ev) == "退款将在 5 至 7 个工作日内原路退回。"
        w.llm.routes = [R("cancel_refund_agent")]
        w.llm.script = [say("好的，请提供订单号。")]
        await collect(cs.stream_chat("那帮我退一张票", MIA, "c1"))
        seen = w.llm.agent_inputs[-1]                       # Agent 这一轮看到的消息里有上一轮的缓存问答
        assert any(isinstance(m, AIMsg) and "5 至 7 个工作日" in m.content for m in seen)
        assert (await w.graph.aget_state(make_config(MIA, "c1"))).next == ()   # 补写后没有挂起的步骤


class FakeMemory:
    def __init__(self):
        self.turns, self.extracted = 0, []

    class _ST:
        available = True
    short_term = _ST()

    class _LT:
        available = False
    long_term = _LT()

    async def save_conversation(self, u, s, msgs):
        self.turns += 1
        return self.turns

    should_extract = staticmethod(lambda t: t > 0 and t % 3 == 0)

    async def background_extract(self, u, s, llm):
        self.extracted.append((u, s))
        return []


@pytest.mark.asyncio
async def test_background_extraction_every_three_turns(rw, monkeypatch):
    import asyncio
    from service import chat_service as cs

    async with World() as w:
        fm = FakeMemory()
        cs.graph, cs.memory = w.graph, fm
        w.llm.routes = [R("policy_agent")] * 4
        w.llm.script = [say(f"回复{i}") for i in range(4)]
        for i in range(4):
            await collect(cs.stream_chat(f"问题{i}", MIA, "m1"))
            await asyncio.sleep(0)                          # 让后台任务有机会执行
        assert fm.extracted == [(MIA, "m1")]                # 只在第 3 轮触发了一次
        cs.memory = None

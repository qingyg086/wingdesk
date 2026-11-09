"""Trace correlation across SSE, background graph, model callback and MCP metadata."""

import asyncio
import json
import logging
import sys
from pathlib import Path
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app"))

from core import observability as obs
from core.mcp.mcp_manager import _SessionProxy
from test_chat_service import collect, meta_of, text_of
from test_graph import MIA, R, World, say


class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.rows = []

    def emit(self, record):
        self.rows.append(json.loads(record.getMessage()))


@pytest.fixture
def events():
    handler = Capture()
    obs._logger.addHandler(handler)
    try:
        yield handler.rows
    finally:
        obs._logger.removeHandler(handler)


@pytest.mark.asyncio
async def test_one_trace_covers_sse_route_model_and_completion_without_content(rw, events):
    from service import chat_service as cs

    async with World() as world:
        cs.graph, cs.memory = world.graph, None
        world.llm.routes = [R("policy_agent")]
        world.llm.script = [say("仅测试正文 secret-answer")]
        trace_id = uuid4().hex
        sse = await collect(cs.stream_chat("private-question", MIA, "trace-model", trace_id=trace_id))
        assert text_of(sse) == "仅测试正文 secret-answer"
        assert meta_of(sse)["trace_id"] == trace_id
        assert meta_of(sse)["timing"]["route_ms"] is not None
        rows = [r for r in events if r["trace_id"] == trace_id]
        assert {"turn.started", "cache.lookup", "route.selected", "model.finished", "turn.finished"} <= {
            r["event"] for r in rows}
        assert next(r for r in rows if r["event"] == "turn.finished")["status"] == "ok"
        route_log = next(r for r in rows if r["event"] == "route.selected")
        assert abs(meta_of(sse)["timing"]["route_ms"] - route_log["duration_ms"]) < 500
        joined = json.dumps(rows)
        assert "private-question" not in joined and "secret-answer" not in joined and MIA not in joined


@pytest.mark.asyncio
async def test_cache_busy_and_disconnect_have_correlated_terminal_logs(rw, events, monkeypatch):
    from service import chat_service as cs
    from test_chat_service import FakeCache

    async with World() as world:
        cs.graph, cs.memory = world.graph, None
        monkeypatch.setattr(cs, "semantic_cache", FakeCache())
        trace_id = uuid4().hex
        sse = await collect(cs.stream_chat("退款多久到账", MIA, "trace-cache", trace_id=trace_id))
        assert meta_of(sse)["trace_id"] == trace_id
        assert next(r for r in events if r["trace_id"] == trace_id and r["event"] == "turn.finished")["cache"] == "L1_SEMANTIC"

        gate = asyncio.Event()
        from test_chat_service import gated_stream_turn
        monkeypatch.setattr(cs, "stream_turn", gated_stream_turn([], gate))
        monkeypatch.setattr(cs, "semantic_cache", type("Miss", (), {"get_cache": staticmethod(
            lambda query, user_id: asyncio.sleep(0, result=None))})())
        first = cs.stream_chat("A", MIA, "trace-disconnect", trace_id=uuid4().hex)
        await first.__anext__()
        await first.aclose()
        busy_id = uuid4().hex
        busy = await collect(cs.stream_chat("B", MIA, "trace-disconnect", trace_id=busy_id))
        assert meta_of(busy)["trace_id"] == busy_id and meta_of(busy)["busy"]
        assert next(r for r in events if r["trace_id"] == busy_id and r["event"] == "turn.finished")["status"] == "busy"
        gate.set()
        await asyncio.gather(*list(cs._background_tasks))
        assert any(r["event"] == "turn.finished" and r["route"] == ["orchestrator"] for r in events)


@pytest.mark.asyncio
async def test_model_usage_and_errors_are_numbers_and_error_types_only(events):
    trace_id = uuid4().hex
    token = obs.set_trace_id(trace_id)
    try:
        metrics = obs.TurnMetrics(trace_id)
        run_id = uuid4()
        await metrics.on_chat_model_start({"name": "fake"}, [[AIMessage("private-prompt")]],
                                          run_id=run_id, metadata={"langgraph_node": "orchestrator"})
        result = LLMResult(generations=[[ChatGeneration(message=AIMessage("private-answer"))]],
                           llm_output={"token_usage": {"prompt_tokens": 7, "completion_tokens": 3}})
        await metrics.on_llm_end(result, run_id=run_id)
        failed = uuid4()
        await metrics.on_chat_model_start({}, [], run_id=failed)
        await metrics.on_llm_error(RuntimeError("secret-error"), run_id=failed)
        assert metrics.summary()["input_tokens"] == 7 and metrics.summary()["output_tokens"] == 3
        assert events[-2]["input_tokens"] == 7 and events[-2]["output_tokens"] == 3
        assert events[-2]["phase"] == "orchestrator"
        assert events[-1]["error_type"] == "RuntimeError"
        assert metrics._phase(["policy_private_decision", "policy_select"]) == "policy_selection"
        assert metrics._phase(["policy_private_decision", "policy_verify"]) == "policy_verification"
        assert all("private" not in json.dumps(row) and "secret" not in json.dumps(row) for row in events)
    finally:
        obs.reset_trace_id(token)


@pytest.mark.asyncio
async def test_mcp_request_metadata_and_logs_keep_same_trace(events):
    class Session:
        async def call_tool(self, name, arguments=None, **kwargs):
            self.meta = kwargs.get("meta")
            return "ok"
    class Registry:
        call_timeout = 1
        async def wait_session(self, server):
            return session
    session = Session()
    trace_id = uuid4().hex
    token = obs.set_trace_id(trace_id)
    try:
        assert await _SessionProxy(Registry(), "airline").call_tool("list_all_airports", {}) == "ok"
    finally:
        obs.reset_trace_id(token)
    assert session.meta == {"trace_id": trace_id}
    assert any(r["trace_id"] == trace_id and r["event"] == "mcp.finished" for r in events)


@pytest.mark.asyncio
async def test_concurrent_sessions_keep_distinct_traces(rw, events, monkeypatch):
    from service import chat_service as cs
    from test_chat_service import gated_stream_turn

    async with World() as world:
        cs.graph, cs.memory = world.graph, None
        gate = asyncio.Event()
        monkeypatch.setattr(cs, "stream_turn", gated_stream_turn([], gate))
        first_id, second_id = uuid4().hex, uuid4().hex
        first = asyncio.create_task(collect(cs.stream_chat("A", MIA, "trace-concurrent-a", trace_id=first_id)))
        second = asyncio.create_task(collect(cs.stream_chat("B", MIA, "trace-concurrent-b", trace_id=second_id)))
        await asyncio.sleep(0.05)
        gate.set()
        first_sse, second_sse = await asyncio.gather(first, second)
        assert meta_of(first_sse)["trace_id"] == first_id
        assert meta_of(second_sse)["trace_id"] == second_id
        assert len([r for r in events if r["event"] == "turn.finished" and r["trace_id"] == first_id]) == 1
        assert len([r for r in events if r["event"] == "turn.finished" and r["trace_id"] == second_id]) == 1


@pytest.mark.asyncio
async def test_mcp_server_reads_request_metadata_without_logging_arguments(events, monkeypatch):
    from mcp_servers import airline_server as server

    class Context:
        request_context = type("Request", (), {"meta": type("Meta", (), {"trace_id": "fixed-trace"})()})()
    monkeypatch.setattr(server.mcp, "get_context", lambda: Context())
    assert await server.run(lambda value: value * 2, 3) == 6
    assert events[-1]["trace_id"] == "fixed-trace" and events[-1]["event"] == "mcp.db.finished"
    assert "arguments" not in events[-1] and "value" not in events[-1]


@pytest.mark.asyncio
async def test_http_response_exposes_the_same_trace_id_as_its_sse_meta(rw):
    from router.chat import chat_endpoint
    from schemas.chat import ChatRequest
    from service import chat_service as cs

    async with World() as world:
        cs.graph, cs.memory = world.graph, None
        world.llm.routes = [R("policy_agent")]
        world.llm.script = [say("已回答")]
        response = await chat_endpoint(ChatRequest(query="规则是什么", user_id=MIA, session_id="trace-http"))
        chunks = [chunk async for chunk in response.body_iterator]
        assert meta_of([json.loads(chunk.removeprefix("data: ")) for chunk in chunks])["trace_id"] == response.headers["x-trace-id"]

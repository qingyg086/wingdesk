"""Per-turn trace context and compact JSON events without conversation content."""

import json
import logging
import sys
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from uuid import uuid4

from langchain_core.callbacks import AsyncCallbackHandler

_trace_id: ContextVar[str | None] = ContextVar("wingdesk_trace_id", default=None)
_logger = logging.getLogger("wingdesk.trace")
_logger.propagate = False
if not _logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(handler)
_logger.setLevel(logging.INFO)


def new_trace_id() -> str:
    return uuid4().hex


def current_trace_id() -> str | None:
    return _trace_id.get()


def set_trace_id(value: str):
    return _trace_id.set(value)


def reset_trace_id(token) -> None:
    _trace_id.reset(token)


def event(name: str, **fields) -> None:
    """Callers pass only bounded, non-sensitive labels and numeric metrics."""
    record = {"time": datetime.now(timezone.utc).isoformat(), "event": name,
              "trace_id": current_trace_id(), **fields}
    _logger.info(json.dumps(record, ensure_ascii=False, separators=(",", ":")))


def _usage(response) -> tuple[int | None, int | None]:
    usage = (getattr(response, "llm_output", None) or {}).get("token_usage") or {}
    if not usage:
        for group in getattr(response, "generations", []) or []:
            for generation in group:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                if usage:
                    break
    return (usage.get("input_tokens", usage.get("prompt_tokens")),
            usage.get("output_tokens", usage.get("completion_tokens")))


class TurnMetrics(AsyncCallbackHandler):
    """LangChain callback; each web request owns one instance."""

    def __init__(self, trace_id: str):
        self.trace_id = trace_id
        self.started: dict[str, tuple[float, str]] = {}
        self.input_tokens = 0
        self.output_tokens = 0
        self.model_calls = 0
        self.tool_calls = 0
        self.usage_calls = 0

    async def on_chat_model_start(self, serialized, messages, *, run_id, tags=None, metadata=None, **kwargs):
        self.started[str(run_id)] = (time.perf_counter(), self._phase(tags, metadata))

    @staticmethod
    def _phase(tags, metadata=None) -> str:
        if tags and "policy_select" in tags:
            return "policy_selection"
        if tags and "policy_verify" in tags:
            return "policy_verification"
        if tags and "policy_private_decision" in tags:
            return "policy_decision"
        if metadata and metadata.get("langgraph_node"):
            return metadata["langgraph_node"]
        for tag in tags or []:
            if tag.startswith("langgraph_node:"):
                return tag.split(":", 1)[1]
        return "agent"

    async def on_llm_end(self, response, *, run_id, tags=None, **kwargs):
        start, phase = self.started.pop(str(run_id), (time.perf_counter(), self._phase(tags)))
        input_tokens, output_tokens = _usage(response)
        self.input_tokens += input_tokens or 0
        self.output_tokens += output_tokens or 0
        self.model_calls += 1
        if input_tokens is not None and output_tokens is not None:
            self.usage_calls += 1
        event("model.finished", phase=phase, duration_ms=round((time.perf_counter() - start) * 1000),
              input_tokens=input_tokens, output_tokens=output_tokens)

    async def on_llm_error(self, error, *, run_id, tags=None, **kwargs):
        start, phase = self.started.pop(str(run_id), (time.perf_counter(), self._phase(tags)))
        event("model.failed", phase=phase, duration_ms=round((time.perf_counter() - start) * 1000),
              error_type=type(error).__name__)

    async def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        name = serialized.get("name", "unknown") if isinstance(serialized, dict) else "unknown"
        self.started[str(run_id)] = (time.perf_counter(), name)

    async def on_tool_end(self, output, *, run_id, **kwargs):
        start, name = self.started.pop(str(run_id), (time.perf_counter(), "unknown"))
        self.tool_calls += 1
        event("tool.finished", tool=name, duration_ms=round((time.perf_counter() - start) * 1000))

    async def on_tool_error(self, error, *, run_id, **kwargs):
        start, name = self.started.pop(str(run_id), (time.perf_counter(), "unknown"))
        self.tool_calls += 1
        event("tool.failed", tool=name, duration_ms=round((time.perf_counter() - start) * 1000),
              error_type=type(error).__name__)

    def summary(self) -> dict:
        return {"model_calls": self.model_calls, "tool_calls": self.tool_calls,
                "input_tokens": self.input_tokens if self.usage_calls == self.model_calls else None,
                "output_tokens": self.output_tokens if self.usage_calls == self.model_calls else None,
                "token_usage_complete": self.usage_calls == self.model_calls}

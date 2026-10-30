"""
脚本化的假大模型：按顺序返回预先写好的回复/工具调用，用来离线、确定性地测试图的机制
（路由、interrupt/resume、checkpointer、粘性路由、越权拦截），不需要 API Key，也不花钱。
模型"答得好不好"由真实模型的演示脚本（agent/scripts/demo_agents.py）检验。
"""

import itertools
from typing import Any, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import Field

_ids = itertools.count(1)


def tool_call(name: str, **args) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{next(_ids)}"}])


def say(text: str) -> AIMessage:
    return AIMessage(content=text)


class ScriptedLLM(BaseChatModel):
    script: list = Field(default_factory=list)       # Agent 每次调用模型时依次弹出一项（AIMessage 或 函数(messages)->AIMessage）
    routes: list = Field(default_factory=list)       # 路由器每次调用时依次弹出一项（RouteDecision 或 Exception）
    agent_inputs: list = Field(default_factory=list)  # 记录每次 Agent 调用时看到的消息
    router_inputs: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.agent_inputs.append(messages)
        if not self.script:
            raise AssertionError("ScriptedLLM：脚本已用完，Agent 调用模型的次数比预期多")
        item = self.script.pop(0)
        msg = item(messages) if callable(item) else item
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        """流式输出：纯文本回复逐字吐出（模拟真实模型的 token 流），工具调用整条吐出。"""
        result = self._generate(messages, stop=stop, **kwargs)
        msg = result.generations[0].message
        if msg.tool_calls or not isinstance(msg.content, str) or not msg.content:
            yield ChatGenerationChunk(message=AIMessageChunk(content=msg.content, tool_call_chunks=[
                {"name": tc["name"], "args": __import__("json").dumps(tc["args"]), "id": tc["id"], "index": i}
                for i, tc in enumerate(msg.tool_calls)]))
            return
        for ch in msg.content:
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=ch))
            if run_manager:
                await run_manager.on_llm_new_token(ch, chunk=chunk)
            yield chunk

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        def _route(messages: Any):
            self.router_inputs.append(messages)
            item = self.routes.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        return RunnableLambda(_route)

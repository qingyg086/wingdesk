"""
UserIdInjector：MCP 工具调用拦截器，防越权的第一道防线。

做法：涉及用户数据的工具，不信任模型传入的 user_id，而是从 LangGraph 的 runtime config
（configurable.user_id，由后端根据登录态设置）里取当前用户，强制写进工具参数。
配合 MCPToolRegistry 把 user_id 从模型可见的工具参数里去掉，模型根本"看不到"这个参数。

- fail-closed：拿不到 user_id 就拒绝执行。如果拿不到时放行（fail-open），config 漏传时
  模型传什么 user_id 就会查谁的数据；安全相关的逻辑，出错时要倾向于"不做"。
- 只对 USER_SCOPED_TOOLS 注入，其他工具参数保持原样。
"""

import logging
from typing import Awaitable, Callable

from langchain_mcp_adapters.interceptors import MCPToolCallRequest, MCPToolCallResult, ToolCallInterceptor
from mcp.types import CallToolResult, TextContent

log = logging.getLogger(__name__)

# 需要绑定当前登录用户的工具（MCP Server 端这些工具都有 user_id 参数）
USER_SCOPED_TOOLS = {
    "get_user_details", "get_reservation_details", "book_reservation", "update_reservation_flights",
    "update_reservation_passengers", "update_reservation_baggages", "cancel_reservation", "send_certificate",
}


def current_user_id(runtime) -> str | None:
    config = getattr(runtime, "config", None) or {}
    return (config.get("configurable") or {}).get("user_id")


class UserIdInjector(ToolCallInterceptor):
    async def __call__(
        self,
        request: MCPToolCallRequest,
        handler: Callable[[MCPToolCallRequest], Awaitable[MCPToolCallResult]],
    ) -> MCPToolCallResult:
        if request.name not in USER_SCOPED_TOOLS:
            return await handler(request)

        user_id = current_user_id(request.runtime)
        if not user_id:
            log.warning("拒绝执行 %s：上下文中没有登录用户", request.name)
            return CallToolResult(content=[TextContent(type="text", text="Error: 未登录，无法执行该操作")],
                                  isError=True)

        claimed = request.args.get("user_id")
        if claimed and claimed != user_id:
            log.warning("模型试图以 user_id=%s 调用 %s，已强制改为当前用户 %s", claimed, request.name, user_id)
        return await handler(request.override(args={**request.args, "user_id": user_id}))

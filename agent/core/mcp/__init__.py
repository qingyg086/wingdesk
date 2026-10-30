"""MCP 连接管理：全局工具注册表 + 防越权拦截器。"""

from .interceptors import USER_SCOPED_TOOLS, UserIdInjector
from .mcp_manager import MCPManager, MCPToolRegistry, mcp_registry

__all__ = ["MCPManager", "MCPToolRegistry", "mcp_registry", "UserIdInjector", "USER_SCOPED_TOOLS"]

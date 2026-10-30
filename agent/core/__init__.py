"""核心 Agent 框架组件。"""

from .mcp.mcp_manager import MCPManager, mcp_registry
from .workflow.state import AgentOutput, AgentState

__all__ = ["AgentOutput", "AgentState", "MCPManager", "mcp_registry", "AgentGraphManager", "MemoryManager"]


def __getattr__(name):
    # 延迟导入，避免 `import core.xxx` 时把整张 Agent 图和记忆系统都加载进来（也避免循环导入）
    if name == "AgentGraphManager":
        from .workflow.graph_manager import AgentGraphManager
        return AgentGraphManager
    if name == "MemoryManager":
        from .memory.memory_manager import MemoryManager
        return MemoryManager
    raise AttributeError(name)

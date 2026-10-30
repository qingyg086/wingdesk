"""多智能体系统的工作流编排。"""

from .state import AgentOutput, AgentState

__all__ = ["AgentOutput", "AgentState", "AgentGraphManager"]


def __getattr__(name):
    # 延迟导入：graph_manager 会导入全部 Agent，避免 `import core` 时就把它们全拉进来（也避免循环导入）
    if name == "AgentGraphManager":
        from .graph_manager import AgentGraphManager
        return AgentGraphManager
    raise AttributeError(name)

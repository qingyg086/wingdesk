"""
Pytest conftest.py - placed at the agent/ root so that:
1. pytest adds this directory to sys.path automatically
2. IDEs (PyCharm, VSCode) recognize agent/ as the source root
3. All absolute imports like `from core.graph import ...` resolve correctly
"""
import os
import sys
from pathlib import Path

# 测试一律用固定时钟（2024-05-15 15:00），和 τ-bench 原始数据、一致性测试的标准答案对齐；
# 不受 agent/.env 里 WINGDESK_CLOCK=live 的影响（MCP Server 子进程也会继承这个设置）
os.environ["WINGDESK_CLOCK"] = "fixed"

# Ensure agent/ is always on sys.path regardless of how the test is invoked
_root = str(Path(__file__).parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

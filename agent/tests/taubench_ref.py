"""
加载 third_party/tau_bench_airline/tools/ 里的 τ-bench 原版工具，作为一致性测试的"标准答案"。

原版工具文件第一行是 `from tau_bench.envs.tool import Tool`，而整个 tau_bench 包一导入就会拉起
litellm 等一堆依赖。这里只往 sys.modules 里塞一个最小的 Tool 基类桩，原版文件一个字不改就能加载。
"""

import copy
import importlib.util
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]                  # 项目根目录
TOOLS_DIR = ROOT / "third_party" / "tau_bench_airline" / "tools"
DATA_DIR = ROOT / "data" / "airline"


def _install_stub():
    if "tau_bench.envs.tool" in sys.modules:
        return
    tool_mod = types.ModuleType("tau_bench.envs.tool")

    class Tool:  # 原版的 Tool 只是一个带两个静态方法的抽象基类
        pass

    tool_mod.Tool = Tool
    for name in ("tau_bench", "tau_bench.envs"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["tau_bench.envs.tool"] = tool_mod


def load_reference_tools() -> dict:
    """返回 {工具名: invoke 函数}，调用方式与原版一致：invoke(data, **kwargs)。"""
    _install_stub()
    from tau_bench.envs.tool import Tool

    tools = {}
    for path in sorted(TOOLS_DIR.glob("*.py")):
        if path.name == "__init__.py":
            continue
        spec = importlib.util.spec_from_file_location(f"taubench_ref_{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cls = next(v for v in vars(mod).values()
                   if isinstance(v, type) and issubclass(v, Tool) and v is not Tool)
        tools[cls.get_info()["function"]["name"]] = cls.invoke
    return tools


_DATA_CACHE = {}


def fresh_data() -> dict:
    """原版内存数据库的一份全新副本（写工具会修改它）。"""
    if not _DATA_CACHE:
        for name in ("users", "flights", "reservations"):
            _DATA_CACHE[name] = json.loads((DATA_DIR / f"{name}.json").read_text(encoding="utf-8"))
    return copy.deepcopy(_DATA_CACHE)

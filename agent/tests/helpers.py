"""测试用小工具：用同样的参数分别调用 MySQLBackend 和 τ-bench 原版工具。"""

import copy
import json


def call_both(backend, data, ref_tools, name, **kwargs):
    """两边各拿一份参数的深拷贝（原版工具会直接修改传入的 list/dict）。"""
    mine = getattr(backend, name)(**copy.deepcopy(kwargs))
    ref = ref_tools[name](data, **copy.deepcopy(kwargs))
    return mine, ref


def same(a: str, b: str) -> bool:
    """先比字符串；不同的话，如果两边都是 JSON，再按内容比较（忽略 key 顺序）。
    为什么需要第二步：新订单的 flights 元素在原版里是 {flight_number, date, price, origin, destination}，
    从 MySQL 读回来是 {origin, destination, flight_number, date, price}，内容相同，key 顺序不同。"""
    if a == b:
        return True
    try:
        return json.loads(a) == json.loads(b)
    except (TypeError, ValueError):
        return False


def assert_same(a, b, what=""):
    assert same(a, b), f"{what}\n  MySQL : {str(a)[:600]}\n  原版  : {str(b)[:600]}"

"""
系统时钟：全项目判断"现在几点、今天几号"都从这里取，不在各处写死日期。

两种模式（环境变量 WINGDESK_CLOCK，未设置时读 agent/.env，默认 fixed）：
- fixed：固定为 2024-05-15 15:00:00（美东时间）。这是 τ-bench airline 数据的设定时间，
  测试、评测和原始数据一起使用，结果可以复现。
- live：真实的当前时间（美东时间）。配合 agent/database/demo_data.py 平移、滚动生成的演示数据使用。

时区：数据里的时间都标注为 EST，这里统一按固定的 UTC-5 计算，不做夏令时切换，保证和数据口径一致。
"""

import os
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

EST = timezone(timedelta(hours=-5), "EST")
FIXED_NOW = datetime(2024, 5, 15, 15, 0, 0)          # τ-bench airline 的设定时间（美东时间）
ENV_KEY = "WINGDESK_CLOCK"
MODES = ("fixed", "live")
_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


@lru_cache(maxsize=1)
def _mode_from_env_file() -> str | None:
    try:
        from dotenv import dotenv_values
        return dotenv_values(_ENV_FILE).get(ENV_KEY)
    except Exception:
        return None


def mode() -> str:
    """当前时钟模式。进程环境变量优先（测试、评测会强制设成 fixed），其次 agent/.env。"""
    m = (os.getenv(ENV_KEY) or _mode_from_env_file() or "fixed").strip().lower()
    if m not in MODES:
        raise ValueError(f"{ENV_KEY} 只能是 {MODES}，实际是 {m!r}")
    return m


def is_fixed() -> bool:
    return mode() == "fixed"


def now(fixed: bool | None = None) -> datetime:
    """当前时间（美东时间，不带时区信息，精确到秒）。fixed=True 时无论配置如何都返回固定时间。"""
    if fixed or (fixed is None and is_fixed()):
        return FIXED_NOW
    return datetime.now(EST).replace(tzinfo=None, microsecond=0)


def today(fixed: bool | None = None) -> date:
    return now(fixed).date()


def now_text(fixed: bool | None = None) -> str:
    """写进提示词的当前时间，例如 "2024-05-15 15:00:00（星期三，美国东部时间 EST）"。"""
    t = now(fixed)
    weekday = "一二三四五六日"[t.weekday()]
    return f"{t:%Y-%m-%d %H:%M:%S}（星期{weekday}，美国东部时间 EST）"


def day_label(days_from_today: int) -> str:
    """演示脚本用的中文日期，例如今天往后 5 天 -> "5 月 20 日"（fixed 模式下是 5 月 20 日）。"""
    d = today() + timedelta(days=days_from_today)
    return f"{d.month} 月 {d.day} 日"

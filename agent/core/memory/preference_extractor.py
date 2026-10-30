"""用大模型从对话里抽取用户的出行偏好。

- 类别固定为航空客服关心的几类，输出格式固定为"类别: 内容"，不在白名单里的类别直接丢弃；
  固定类别是为了"按类别覆盖更新"：用户以前说靠窗、现在说靠过道，新的一条替换旧的，而不是两条并存。
- 只抽取**用户自己明确说出**的长期习惯：只把用户说的话交给模型（不给助手的回复，避免把客服的建议当成用户偏好），
  并区分"习惯"和"这一次"（"这次坐商务舱"不是偏好，"我一般都坐商务舱"才是）。
- 不抽取敏感信息：卡号、证件号、密码一律不记；支付方式只记类型和尾号。
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

CATEGORIES = ["座位偏好", "舱位偏好", "常用支付方式", "常飞航线", "常用乘客", "餐食偏好", "沟通偏好"]

_PROMPT = """你是航空公司客服系统的用户画像助手。下面是某位用户在对话中说过的话（只有用户的发言）。
请从中提取这位用户**明确表达的、长期有效的**出行偏好或习惯。

只能使用以下类别（每条一行，格式为"类别: 内容"）：
{categories}

规则：
1. 只提取用户自己明确说出的习惯或偏好，例如"我一般坐靠窗""我每次都用尾号 7447 的信用卡""我经常飞纽约到西雅图"。
2. 只针对这一次行程的选择不算偏好，例如"这次订经济舱""明天去西雅图"——不要提取。
3. 不要推测，不要根据一次行为总结习惯。
4. 支付方式只记录类型和尾号（如"尾号 7447 的信用卡"），不要记录完整卡号、证件号、密码等敏感信息。
5. 同一类别只输出一条。内容只写偏好本身，不要照抄用户原话，例如：
   座位偏好: 靠窗
   常飞航线: 纽约 → 西雅图
   常用支付方式: 尾号 7447 的信用卡
6. 没有符合条件的内容时，只输出：无

用户的发言：
{conversation}

提取结果："""

_SENSITIVE = re.compile(r"\d{8,}")      # 连续 8 位以上数字：疑似卡号、证件号，整条丢弃


class PreferenceExtractor:
    def __init__(self, llm: Any, max_conversation_chars: int = 3000) -> None:
        self._llm = llm
        self._max_chars = max_conversation_chars

    async def extract(self, conversation_text: str, existing: list[str] | None = None) -> list[str]:
        """返回新的偏好列表，每条形如"座位偏好: 靠窗"。和 existing 完全相同的条目会被去掉。"""
        prompt = _PROMPT.format(categories="、".join(CATEGORIES),
                                conversation=conversation_text[-self._max_chars:])
        try:
            raw = (await self._llm.ainvoke([{"role": "user", "content": prompt}])).content.strip()
        except Exception as exc:
            logger.warning("PreferenceExtractor LLM call failed: %s", exc)
            return []
        items = self.parse(raw)
        existing_set = {e.strip() for e in existing or []}
        new = [i for i in items if i not in existing_set]
        logger.info("[EXTRACTOR] 候选 %d 条，新增 %d 条：%s", len(items), len(new), new)
        return new

    @staticmethod
    def parse(raw: str) -> list[str]:
        """解析模型输出：只保留白名单类别，每个类别只取第一条，丢弃疑似敏感信息。"""
        out, seen = [], set()
        for line in raw.splitlines():
            line = line.strip().lstrip("-•* ").replace("：", ":")
            if ":" not in line:
                continue
            cat, value = (x.strip() for x in line.split(":", 1))
            if cat in CATEGORIES and value and value != "无" and cat not in seen and not _SENSITIVE.search(value):
                seen.add(cat)
                out.append(f"{cat}: {value}")
        return out

    @staticmethod
    def category_of(item: str) -> str:
        return item.split(":", 1)[0].strip()

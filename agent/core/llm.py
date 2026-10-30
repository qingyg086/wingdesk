"""
大模型工厂：所有 Agent 统一从这里拿模型，便于集中配置，也便于测试时替换成假模型。
模型：阿里云百炼 qwen-plus（OpenAI 兼容接口）。
"""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


@lru_cache(maxsize=8)
def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    """temperature 默认 0：客服场景要的是稳定、可复现，而不是创造性。"""
    return ChatOpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model=os.getenv("MODEL", "qwen-plus"),
        base_url=os.getenv("BASE_URL", DEFAULT_BASE_URL),
        temperature=temperature,
        timeout=60,
        max_retries=2,
    )

from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str
    user_id: Optional[str] = "mia_li_3668"         # 演示账号（前端可切换 mia_li_3668 / lucas_wilson_8118）
    session_id: Optional[str] = "default_session"
    # 回应"待确认操作"卡片：前端点【确认】传 true、点【取消】传 false；不传时根据 query 文本判断（"确认""好的"等）
    confirm: Optional[bool] = Field(default=None)


class ChatResponse(BaseModel):
    status: str
    reply: str
    user_id: str
    session_id: str

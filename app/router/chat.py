from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from schemas.chat import ChatRequest
from service.chat_service import stream_chat
from core.observability import new_trace_id

router = APIRouter()

@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    处理多智能体聊天请求，并使用 SSE (Server-Sent Events) 返回流式响应。
    默认仅在问题精确命中 L1 缓存时直接返回；其余问题进入 Agent 图编排流程。
    """
    trace_id = new_trace_id()
    return StreamingResponse(
        stream_chat(request.query, request.user_id, request.session_id, request.confirm, trace_id=trace_id),
        media_type="text/event-stream", headers={"X-Trace-ID": trace_id}
    )

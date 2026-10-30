import asyncio
import logging
import os
import sys

# 将 agent 目录加入 sys.path
AGENT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent")
sys.path.insert(0, AGENT_DIR)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from router import chat
from service.chat_service import init_agent_system
from core.mcp.mcp_manager import mcp_registry
from core import clock

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("database.demo_data").setLevel(logging.INFO)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时：先连上 MCP Server（全局只拉起一次子进程），再初始化 Agent 图和记忆
    await mcp_registry.start()
    await init_agent_system()
    # 实时时钟（演示）模式：后台定时补齐未来航班、推进航班状态；固定时钟模式下数据不动
    roll_task = None
    if not clock.is_fixed():
        from database.demo_data import roll_periodically
        roll_task = asyncio.create_task(roll_periodically())
    print(f"系统时钟：{clock.mode()}，当前时间 {clock.now_text()}")
    yield
    # 关闭时：停止定时任务，关闭 MCP 会话，子进程随之退出
    if roll_task:
        roll_task.cancel()
    await mcp_registry.stop()

app = FastAPI(title="WingDesk API", description="翼枢 WingDesk 航空客服多智能体平台", lifespan=lifespan)

# 配置跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(chat.router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_main:app", host="0.0.0.0", port=5000, reload=True)

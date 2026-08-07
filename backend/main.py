"""FastAPI 服务入口"""
import sys
from pathlib import Path

# 假设当前文件位于 project/src/module/file.py
# 项目根目录为 project/
ROOT_DIR = Path(__file__).resolve().parent.parent # 根据层级调整 .parent 数量
sys.path.insert(0, str(ROOT_DIR))

import os
import uuid
import json
import time
import logging
from pathlib import Path
from collections import OrderedDict

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from backend.agent import excel_agent
from backend.tools import read_excel_structure
from backend.state import ExcelAgentState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Excel Agent API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 上传目录
UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# 文件大小限制 (100 MB)
MAX_FILE_SIZE = 100 * 1024 * 1024


# ============ 修复 #3：带 TTL + LRU 的 Session 存储 ============

class SessionStore:
    """带 TTL 和容量上限的 session 存储（Demo 级别）"""

    def __init__(self, max_size: int = 500, ttl_seconds: int = 7200):
        self._store: OrderedDict[str, dict] = OrderedDict()
        self._timestamps: dict[str, float] = {}
        self.max_size = max_size
        self.ttl = ttl_seconds

    def set(self, key: str, value: dict) -> None:
        self._cleanup_expired()
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        self._timestamps[key] = time.time()
        # LRU 淘汰
        while len(self._store) > self.max_size:
            oldest_key, _ = self._store.popitem(last=False)
            self._timestamps.pop(oldest_key, None)

    def get(self, key: str) -> dict:
        self._cleanup_expired()
        if key not in self._store:
            return {}
        self._store.move_to_end(key)
        self._timestamps[key] = time.time()
        return self._store[key]

    def _cleanup_expired(self) -> None:
        now = time.time()
        expired = [
            k for k, t in self._timestamps.items() if now - t > self.ttl
        ]
        for k in expired:
            self._store.pop(k, None)
            self._timestamps.pop(k, None)


session_contexts = SessionStore(max_size=500, ttl_seconds=7200)


# ============ 请求模型 ============

class ChatRequest(BaseModel):
    session_id: str
    message: str


# ============ 工具函数 ============

def sse_event(data: dict) -> str:
    """格式化 SSE 事件"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# ============ API 路由 ============

@app.post("/api/upload")
async def upload_excel(file: UploadFile = File(...)):
    """上传 Excel 文件（修复 #4：流式写入 + 大小校验）"""

    if not file.filename or not file.filename.endswith((".xlsx", ".xls", ".csv")):
        raise HTTPException(400, "仅支持 .xlsx / .xls / .csv 文件")

    session_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{session_id}_{file.filename}"

    # 流式写入 + 大小校验
    total_size = 0
    try:
        with open(file_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)  # 1 MB
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > MAX_FILE_SIZE:
                    raise HTTPException(400, "文件大小超过 100 MB 限制")
                f.write(chunk)
    except HTTPException:
        file_path.unlink(missing_ok=True)
        raise
    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(500, f"文件保存失败: {str(e)}")

    # 解析文件结构
    try:
        summary = read_excel_structure.invoke({"file_path": str(file_path)})
    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(500, f"文件解析失败: {str(e)}")

    # 保存 session 上下文
    session_contexts.set(session_id, {
        "file_path": str(file_path),
        "file_name": file.filename,
        "summary": summary,
    })

    logger.info(
        "[upload] session=%s, file=%s, size=%d", session_id, file.filename, total_size
    )

    return {
        "session_id": session_id,
        "file_name": file.filename,
        "summary": summary,
    }


@app.post("/api/chat")
async def chat_with_agent(request: ChatRequest):
    """与 Excel Agent 对话（SSE 流式）（修复 #1：aget_state 获取 chart_config）"""

    config = {"configurable": {"thread_id": request.session_id}}

    # 从 session 上下文中获取文件信息
    ctx = session_contexts.get(request.session_id)
    inputs: ExcelAgentState = {
        "messages": [HumanMessage(content=request.message)],
        "file_path": ctx.get("file_path"),
        "file_name": ctx.get("file_name"),
        "df_summary": ctx.get("summary"),
        "sheet_names": None,
        "analysis_result": None,
        "chart_config": None,
        "error": None,
    }

    logger.info(
        "[chat] session=%s, msg=%s", request.session_id, request.message[:80]
    )

    async def event_stream():
        try:
            async for event in excel_agent.astream_events(
                inputs, config=config, version="v2"
            ):
                kind = event.get("event")

                # 流式文本
                if kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        yield sse_event({"type": "text", "content": chunk.content})

                # 工具调用开始
                elif kind == "on_tool_start":
                    tool_name = event.get("name", "")
                    yield sse_event({
                        "type": "tool_start",
                        "content": f"🔧 正在调用: {tool_name}",
                    })

                # 工具调用结束
                elif kind == "on_tool_end":
                    output = event.get("data", {}).get("output", "")
                    yield sse_event({
                        "type": "tool_result",
                        "content": str(output)[:500],
                    })

            # ✅ 修复 #1：流结束后从 checkpointer 获取最终 state
            final_state = await excel_agent.aget_state(config)
            if final_state and final_state.values.get("chart_config"):
                yield sse_event({
                    "type": "chart",
                    "content": final_state.values["chart_config"],
                })

            yield sse_event({"type": "done"})

        except Exception as e:
            logger.error("[chat] 错误: %s", e, exc_info=True)
            yield sse_event({"type": "error", "content": f"处理出错: {str(e)}"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/chat/sync")
async def chat_sync(request: ChatRequest):
    """同步对话接口（调试用）"""

    config = {"configurable": {"thread_id": request.session_id}}
    ctx = session_contexts.get(request.session_id)
    inputs: ExcelAgentState = {
        "messages": [HumanMessage(content=request.message)],
        "file_path": ctx.get("file_path"),
        "file_name": ctx.get("file_name"),
        "df_summary": ctx.get("summary"),
        "sheet_names": None,
        "analysis_result": None,
        "chart_config": None,
        "error": None,
    }

    try:
        result = await excel_agent.ainvoke(inputs, config=config)
        final_message = result["messages"][-1]
        return {
            "session_id": request.session_id,
            "reply": final_message.content,
            "chart_config": result.get("chart_config"),
        }
    except Exception as e:
        logger.error("[chat_sync] 错误: %s", e, exc_info=True)
        raise HTTPException(500, f"Agent 执行错误: {str(e)}")


@app.get("/api/health")
async def health_check():
    return {"status": "ok"}


@app.get("/")
async def root():
    frontend_path = Path(__file__).parent.parent / "frontend" / "index.html"
    return FileResponse(str(frontend_path))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
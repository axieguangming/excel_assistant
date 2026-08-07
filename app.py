"""
FastAPI 主应用
"""
import os
import uuid
import shutil
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import sse_starlette.sse as sse

from tools import load_excel_to_store, save_df_to_excel, get_df
from agent import run_agent, run_agent_stream

app = FastAPI(title="Excel AI Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 目录初始化
UPLOAD_DIR = Path("./files/uploads")
OUTPUT_DIR = Path("./files/downloads")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")


# ============ 数据模型 ============
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    reply: str
    steps: list = []


# ============ 路由 ============

@app.get("/", response_class=HTMLResponse)
async def index():
    """返回前端页面"""
    html_path = Path("static/index.html")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/api/upload")
async def upload_excel(file: UploadFile = File(...)):
    """上传 Excel 文件"""
    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="仅支持 .xlsx / .xls 文件")

    session_id = str(uuid.uuid4())[:8]
    file_path = UPLOAD_DIR / f"{session_id}_{file.filename}"

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 加载到内存
    df = load_excel_to_store(str(file_path), session_id)

    return {
        "session_id": session_id,
        "filename": file.filename,
        "rows": len(df),
        "columns": list(df.columns),
        "preview": json.loads(df.head(20).to_json(orient="records", force_ascii=False)),
    }


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """与 AI Agent 对话"""
    if not req.session_id:
        raise HTTPException(status_code=400, detail="缺少session_id, 请先长传文件")
    try:
        reply = run_agent(
            message=req.message,
            session_id=req.session_id
        )
        print(reply)
        return {"reply": reply}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """流式对话（SSE）"""
    async def event_generator():
        try:
            for step in run_agent_stream(req.message):
                yield {"data": json.dumps(step, ensure_ascii=False)}
        except Exception as e:
            yield {"data": json.dumps({"type": "error", "content": str(e)}, ensure_ascii=False)}

    return sse.EventSourceResponse(event_generator())


@app.get("/api/preview/{session_id}")
async def preview_excel(session_id: str):
    """获取当前处理后的数据预览"""
    df = get_df(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="未找到该会话的数据")
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "data": json.loads(df.head(100).to_json(orient="records", force_ascii=False)),
    }


@app.get("/api/download/{session_id}")
async def download_excel(session_id: str):
    """下载处理后的 Excel"""
    df = get_df(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="未找到该会话的数据")

    output_path = OUTPUT_DIR / f"{session_id}_result.xlsx"
    save_df_to_excel(session_id, str(output_path))

    return FileResponse(
        path=str(output_path),
        filename=f"result_{session_id}.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
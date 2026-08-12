# app.py
import os
import json
import uuid
import time

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent import run_agent

app = FastAPI(title="Excel AI 助手")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR, OUTPUT_DIR = "uploads", "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Demo 用内存会话存储（生产环境请换 Redis）：session_id -> {df, history}
SESSIONS: dict = {}


class ChatRequest(BaseModel):
    session_id: str
    question: str


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".xlsx", ".xls", ".csv")):
        raise HTTPException(400, "仅支持 xlsx / xls / csv 文件")
    content = await file.read()

    session_id = uuid.uuid4().hex[:12]
    path = os.path.join(UPLOAD_DIR, f"{session_id}_{file.filename}")
    with open(path, "wb") as f:
        f.write(content)

    df = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)
    SESSIONS[session_id] = {"df": df, "history": []}

    return {
        "session_id": session_id,
        "filename": file.filename,
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        # split 格式便于前端直接渲染表格
        "preview": json.loads(df.head(100).to_json(orient="split", date_format="iso", force_ascii=False)),
    }


@app.post("/api/chat")
async def chat(req: ChatRequest):
    sess = SESSIONS.get(req.session_id)
    if not sess:
        raise HTTPException(404, "会话不存在，请重新上传文件")

    final = run_agent(sess["df"], req.question, sess["history"])
    resp = {"answer": final.get("final_answer", ""), "code": final.get("code", ""),
            "has_result": False}

    rdf = final.get("result_df")
    if rdf is not None and len(rdf) > 0:
        # 生成结果 Excel
        out_name = f"{req.session_id}_{int(time.time())}_{uuid.uuid4().hex[:6]}.xlsx"
        rdf.to_excel(os.path.join(OUTPUT_DIR, out_name), index=False)
        resp.update({
            "has_result": True,
            "download_url": f"/api/download/{out_name}",
            "result_preview": json.loads(rdf.head(100).to_json(orient="split", date_format="iso", force_ascii=False)),
        })

    sess["history"] += [{"role": "user", "content": req.question},
                        {"role": "assistant", "content": resp["answer"]}]
    return resp


@app.get("/api/download/{filename}")
async def download(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(404, "文件不存在")
    return FileResponse(path, filename=filename,
                        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# 前端页面
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
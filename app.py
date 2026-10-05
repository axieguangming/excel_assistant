import io
import json
import os
import re
import threading
import time
import uuid

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from agent import agent_app
from schemas import ChatRequest, SessionRequest

app = FastAPI(title="Excel AI 助手")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR, OUTPUT_DIR = "uploads", "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

SESSIONS: dict[str, dict] = {}
_AGENT_LOCK = threading.Lock()


def _run_agent_serialized(df: pd.DataFrame, question: str, history: list) -> dict:
    with _AGENT_LOCK:
        return run_agent(df, question, history)


def _session_payload(df: pd.DataFrame, filename: str = "当前工作表") -> dict:
    return {
        "session_id": uuid.uuid4().hex[:12],
        "filename": filename,
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        "preview": json.loads(df.head(100).to_json(orient="split", date_format="iso", force_ascii=False)),
    }


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
    payload = _session_payload(df, file.filename)
    payload["session_id"] = session_id
    return payload


@app.post("/api/session")
async def create_session(req: SessionRequest):
    if not req.columns:
        raise HTTPException(400, "工作表没有可用的表头数据")

    df = pd.DataFrame(req.rows, columns=req.columns)
    session_id = uuid.uuid4().hex[:12]
    SESSIONS[session_id] = {"df": df, "history": []}
    payload = _session_payload(df)
    payload["session_id"] = session_id
    return payload


def build_df_info(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    df.info(buf=buf)
    try:
        preview = df.head(8).to_markdown(index=False)
    except Exception:
        preview = df.head(8).to_string(index=False)
    return (
        f"数据形状: {df.shape[0]} 行 x {df.shape[1]} 列\n"
        f"列名: {list(df.columns)}\n\n"
        f"字段类型:\n{buf.getvalue()}"
        f"数据预览(前8行):\n{preview}"
    )


def run_agent(df: pd.DataFrame, question: str, history: list) -> dict:
    init_state = {
        "df": df,
        "df_info": build_df_info(df),
        "question": question,
        "history": history,
        "code": "",
        "error": "",
        "retries": 0,
        "result_df": None,
        "answer_text": "",
        "final_answer": "",
        "chart_file": "",
    }
    return agent_app.invoke(init_state, config={"recursion_limit": 30})


@app.post("/api/chat")
async def chat(req: ChatRequest):
    sess = SESSIONS.get(req.session_id)
    if not sess:
        raise HTTPException(404, "会话不存在，请重新上传文件")

    final = await run_in_threadpool(_run_agent_serialized, sess["df"], req.question, sess["history"])
    resp = {
        "answer": final.get("final_answer", ""),
        "code": final.get("code", ""),
        "has_result": False,
        "chart_url": final.get("chart_file", ""),
    }

    rdf = final.get("result_df")
    if rdf is not None and len(rdf) > 0:
        safe_q = re.sub(r"[\\/*?:\[\]]", "_", req.question)[:15]
        ts = str(int(time.time()))[-4:]
        sheet_name = f"结果_{safe_q}_{ts}"
        out_name = f"{req.session_id}_{int(time.time())}_{uuid.uuid4().hex[:6]}.xlsx"
        out_path = os.path.join(OUTPUT_DIR, out_name)

        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            sess["df"].to_excel(writer, sheet_name="原始数据", index=False)
            rdf.to_excel(writer, sheet_name=sheet_name, index=False)

        resp.update(
            {
                "has_result": True,
                "download_url": f"/api/download/{out_name}",
                "result_sheet_name": sheet_name,
                "result_preview": json.loads(
                    rdf.head(200).to_json(orient="split", date_format="iso", force_ascii=False)
                ),
            }
        )

    sess["history"] += [
        {"role": "user", "content": req.question},
        {"role": "assistant", "content": resp["answer"]},
    ]
    return resp


@app.get("/api/download/{filename}")
async def download(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(404, "文件不存在")
    return FileResponse(
        path,
        filename=filename,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.get("/api/chart/{filename}")
async def serve_chart(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(404, "图表不存在")
    return FileResponse(path, media_type="image/png")


app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8443"))
    certfile = os.getenv("SSL_CERTFILE")
    keyfile = os.getenv("SSL_KEYFILE")

    if certfile and keyfile and os.path.exists(certfile) and os.path.exists(keyfile):
        uvicorn.run(app, host=host, port=port, ssl_certfile=certfile, ssl_keyfile=keyfile)
    else:
        print("未设置 SSL_CERTFILE/SSL_KEYFILE，以 HTTP 启动。Excel 插件加载需要 HTTPS，请参考 addin/README 生成证书。")
        uvicorn.run(app, host=host, port=port)
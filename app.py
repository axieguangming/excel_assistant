# app.py
import os
import json
import uuid
import time
import io
import threading

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
import re as _re
from agent import agent_app
from schemas import ChatRequest, SessionRequest

app = FastAPI(title="Excel AI 助手")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR, OUTPUT_DIR = "uploads", "outputs"
# 创建上传和下载文件夹
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Demo 用内存会话存储（生产环境请换 Redis）：session_id -> {df, history}
SESSIONS: dict = {}

# matplotlib 的 pyplot 全局状态不是线程安全的，且 agent 用的是进程级共享状态，
# 因此把 agent 放到线程池里跑（避免阻塞事件循环）的同时用锁串行化。
_AGENT_LOCK = threading.Lock()


def _run_agent_serialized(df: pd.DataFrame, question: str, history: list) -> dict:
    with _AGENT_LOCK:
        return run_agent(df, question, history)

# 上传请求处理
@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    # 限制上传的文件类型
    if not file.filename.lower().endswith((".xlsx", ".xls", ".csv")):
        raise HTTPException(400, "仅支持 xlsx / xls / csv 文件")
    # 读取用户上传的文件
    content = await file.read()

    # 生成会话id，并写入upload临时文件夹
    session_id = uuid.uuid4().hex[:12]
    path = os.path.join(UPLOAD_DIR, f"{session_id}_{file.filename}")
    with open(path, "wb") as f:
        f.write(content)

    # 读取存入upload的文件
    df = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)
    # 存储会话对应的df和聊天历史
    SESSIONS[session_id] = {"df": df, "history": []}

    # 返回所需的内容给前端
    return {
        "session_id": session_id,
        "filename": file.filename,
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        # split 格式便于前端直接渲染表格
        "preview": json.loads(df.head(100).to_json(orient="split", date_format="iso", force_ascii=False)),
    }


@app.post("/api/session")
async def create_session(req: SessionRequest):
    """Excel 插件入口：用当前工作表的数据（表头 + 数据行）建立会话。"""
    if not req.columns:
        raise HTTPException(400, "工作表没有可用的表头数据")
    df = pd.DataFrame(req.rows, columns=req.columns)
    session_id = uuid.uuid4().hex[:12]
    SESSIONS[session_id] = {"df": df, "history": []}
    return {
        "session_id": session_id,
        "filename": "当前工作表",
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        "preview": json.loads(df.head(100).to_json(orient="split", date_format="iso", force_ascii=False)),
    }


def build_df_info(df: pd.DataFrame) -> str:
    """生成给 LLM 看的数据描述"""
    buf = io.StringIO()
    df.info(buf=buf)
    try:
        preview = df.head(8).to_markdown(index=False)
    except Exception:
        preview = df.head(8).to_string(index=False)
    return (f"数据形状: {df.shape[0]} 行 x {df.shape[1]} 列\n"
            f"列名: {list(df.columns)}\n\n"
            f"字段类型:\n{buf.getvalue()}\n"
            f"数据预览(前8行):\n{preview}")

def run_agent(df: pd.DataFrame, question: str, history: list) -> dict:
    """对外入口：返回最终状态"""
    init_state = {
        "df": df, "df_info": build_df_info(df), "question": question,
        "history": history, "code": "", "error": "", "retries": 0,
        "result_df": None, "answer_text": "", "final_answer": "",
        "chart_file": "",
    }
    return agent_app.invoke(init_state, config={"recursion_limit": 30})


# 聊天请求处理
# app.py  /api/chat 路由中，替换原有 resp 构建部分
@app.post("/api/chat")
async def chat(req: ChatRequest):
    # 获取本次聊天的会话id对应的会话信息：df和聊天历史
    sess = SESSIONS.get(req.session_id)
    if not sess:
        raise HTTPException(404, "会话不存在，请重新上传文件")

    # 调用和运行agent，根据df，本次问题和聊天历史进行处理
    # agent 内部是同步的 LLM + pandas 计算，放线程池执行，避免卡住事件循环
    final = await run_in_threadpool(_run_agent_serialized, sess["df"], req.question, sess["history"])
    # 返回内容
    resp = {"answer": final.get("final_answer", ""), "code": final.get("code", ""),
            "has_result": False, "chart_url": final.get("chart_file", "")}

    # 返回的df
    rdf = final.get("result_df")
    if rdf is not None and len(rdf) > 0:    # 如果有df且内容不为空
        # ✅ 新增：生成 Sheet 名称（取问题前15字符 + 时间戳后4位，避免重名）
        safe_q = _re.sub(r'[\\/*?:\[\]]', '_', req.question)[:15]
        ts = str(int(time.time()))[-4:]
        sheet_name = f"结果_{safe_q}_{ts}"

        out_name = f"{req.session_id}_{int(time.time())}_{uuid.uuid4().hex[:6]}.xlsx"
        out_path = os.path.join(OUTPUT_DIR, out_name)

        # ✅ 写入 Excel 时同时保留原始数据 Sheet + 结果 Sheet
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            sess["df"].to_excel(writer, sheet_name="原始数据", index=False)
            rdf.to_excel(writer, sheet_name=sheet_name, index=False)

        resp.update({
            "has_result": True,
            "download_url": f"/api/download/{out_name}",
            "result_sheet_name": sheet_name,   # ✅ 新增
            "result_preview": json.loads(
                rdf.head(200).to_json(orient="split", date_format="iso", force_ascii=False)
            ),
        })

    sess["history"] += [{"role": "user", "content": req.question},
                        {"role": "assistant", "content": resp["answer"]}]
    return resp

# 下载请求
@app.get("/api/download/{filename}")
async def download(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(404, "文件不存在")
    return FileResponse(path, filename=filename,
                        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# 图表图片服务
@app.get("/api/chart/{filename}")
async def serve_chart(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(404, "图表不存在")
    return FileResponse(path, media_type="image/png")


# 前端页面
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8443"))
    certfile = os.getenv("SSL_CERTFILE")
    keyfile = os.getenv("SSL_KEYFILE")

    if certfile and keyfile and os.path.exists(certfile) and os.path.exists(keyfile):
        # Excel 插件加载项要求 HTTPS
        uvicorn.run(app, host=host, port=port, ssl_certfile=certfile, ssl_keyfile=keyfile)
    else:
        print("未设置 SSL_CERTFILE/SSL_KEYFILE，以 HTTP 启动。"
              "Excel 插件加载需要 HTTPS，请参考 addin/README 生成证书。")
        uvicorn.run(app, host=host, port=port)
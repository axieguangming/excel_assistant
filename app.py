import io
import ipaddress
import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app_paths import DATA_DIR, STATIC_DIR
from agent import invoke_agent
from load_model import describe_model_error, current_provider
from local_model_runtime import (
    STATE_ERROR,
    STATE_READY,
    get_runtime,
    set_runtime,
    build_runtime,
)
from schemas import ChatRequest, SessionRequest

app = FastAPI(title="Excel AI 助手")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = str(DATA_DIR / "uploads")
OUTPUT_DIR = str(DATA_DIR / "outputs")
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
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
    filename = os.path.basename(file.filename or "")
    if not filename.lower().endswith((".xlsx", ".xls", ".csv")):
        raise HTTPException(400, "仅支持 xlsx / xls / csv 文件")

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "文件大小不能超过 50 MB。")
    session_id = uuid.uuid4().hex[:12]
    path = os.path.join(UPLOAD_DIR, f"{session_id}_{filename}")
    with open(path, "wb") as f:
        f.write(content)

    try:
        df = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path)
    except (ImportError, OSError, ValueError) as error:
        os.remove(path)
        raise HTTPException(400, f"无法读取该文件：{error}") from error
    SESSIONS[session_id] = {"df": df, "history": []}
    payload = _session_payload(df, filename)
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
        "messages": [],
        "current_df": df.copy(),
        "plans": [],
        "tool_steps": 0,
        "turns": 0,
        "code": "",
        "result_df": None,
        "final_answer": "",
        "chart_file": "",
        "user_prompt": "",
    }
    return invoke_agent(init_state)


def _ensure_local_https_certificates() -> tuple[str, str]:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    cert_dir = Path(__file__).resolve().parent / "addin" / "certs"
    cert_path, key_path = cert_dir / "cert.pem", cert_dir / "key.pem"
    cert_dir.mkdir(parents=True, exist_ok=True)

    if not cert_path.is_file() or not key_path.is_file():
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
        ])
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=5))
            .not_valid_after(now + timedelta(days=825))
            .add_extension(
                x509.SubjectAlternativeName([
                    x509.DNSName("localhost"),
                    x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                ]),
                critical=False,
            )
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(
                x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
                critical=False,
            )
            .sign(key, hashes.SHA256())
        )
        cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        key_path.write_bytes(key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ))

    certificate = x509.load_pem_x509_certificate(cert_path.read_bytes())
    thumbprint = certificate.fingerprint(hashes.SHA1()).hex().upper()
    certutil = shutil.which("certutil")
    if certutil is None:
        raise RuntimeError("找不到 Windows certutil.exe，无法自动信任本地 HTTPS 证书。")

    verified = subprocess.run(
        [certutil, "-user", "-verifystore", "Root", thumbprint],
        capture_output=True,
        text=True,
        check=False,
    )
    if verified.returncode != 0:
        installed = subprocess.run(
            [certutil, "-user", "-addstore", "-f", "Root", str(cert_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if installed.returncode != 0:
            details = (installed.stderr or installed.stdout).strip()
            raise RuntimeError(f"无法将 localhost 证书加入当前用户的受信任根存储：{details}")

    return str(cert_path), str(key_path)


@app.post("/api/chat")
async def chat(req: ChatRequest):
    sess = SESSIONS.get(req.session_id)
    if not sess:
        raise HTTPException(404, "会话不存在，请重新上传文件")

    runtime = get_runtime()
    if runtime is not None:
        state = runtime.status()["state"]
        if state in {"needs_download", "downloading", "verifying"}:
            raise HTTPException(
                503,
                "本地模型尚未准备好，请先完成模型下载（界面顶部会显示进度）。",
            )

    try:
        final = await run_in_threadpool(
            _run_agent_serialized,
            sess["df"],
            req.question,
            sess["history"],
        )
    except Exception as error:
        guidance = describe_model_error(error)
        if guidance is None:
            raise
        raise HTTPException(status_code=503, detail=guidance) from error
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


@app.get("/api/model/status")
async def model_status():
    """返回当前模型状态，供界面决定是否展示下载引导。"""
    runtime = get_runtime()
    if runtime is None:
        provider = current_provider()
        return {
            "provider": provider,
            "state": "disabled" if provider != "local" else STATE_ERROR,
            "can_download": False,
            "message": "",
        }
    return runtime.status()


@app.post("/api/model/download")
async def model_download():
    """开始下载本地模型；进度通过 /api/model/status 轮询。"""
    runtime = get_runtime()
    if runtime is None:
        raise HTTPException(400, "当前未使用本地模型，无需下载。")
    if runtime.downloader.is_installed():
        return {"started": False, "reason": "模型文件已存在。"}
    if not runtime.manifest.is_configured:
        raise HTTPException(
            400,
            "未配置模型下载地址。请手动放置模型文件，或重新安装包含下载配置的版本。",
        )
    started = runtime.start_download()
    return {"started": started}


@app.get("/api/download/{filename}")
async def download(filename: str):
    safe_filename = os.path.basename(filename)
    if safe_filename != filename:
        raise HTTPException(400, "文件名无效")
    path = os.path.join(OUTPUT_DIR, safe_filename)
    if not os.path.exists(path):
        raise HTTPException(404, "文件不存在")
    return FileResponse(
        path,
        filename=filename,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.get("/api/chart/{filename}")
async def serve_chart(filename: str):
    safe_filename = os.path.basename(filename)
    if safe_filename != filename:
        raise HTTPException(400, "文件名无效")
    path = os.path.join(OUTPUT_DIR, safe_filename)
    if not os.path.exists(path):
        raise HTTPException(404, "图表不存在")
    return FileResponse(path, media_type="image/png")


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8443"))
    certfile = os.getenv("SSL_CERTFILE")
    keyfile = os.getenv("SSL_KEYFILE")

    # 开发模式下也初始化模型运行时，便于在浏览器里走通下载与启动流程。
    set_runtime(build_runtime())

    if bool(certfile) != bool(keyfile):
        raise RuntimeError("SSL_CERTFILE 和 SSL_KEYFILE 必须同时设置。")

    if certfile and keyfile:
        uvicorn.run(
            app,
            host=host,
            port=port,
            ssl_certfile=certfile,
            ssl_keyfile=keyfile,
            timeout_graceful_shutdown=5,
        )
    elif os.name == "nt":
        certfile, keyfile = _ensure_local_https_certificates()
        print("已自动配置 localhost HTTPS 证书。")
        uvicorn.run(
            app,
            host=host,
            port=port,
            ssl_certfile=certfile,
            ssl_keyfile=keyfile,
            timeout_graceful_shutdown=5,
        )
    else:
        print("未设置 SSL_CERTFILE/SSL_KEYFILE，以 HTTP 启动。Excel 插件加载需要 HTTPS，请参考 addin/README 生成证书。")
        uvicorn.run(app, host=host, port=port, timeout_graceful_shutdown=5)
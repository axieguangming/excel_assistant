"""桌面端启动器：在后台线程跑 uvicorn，再用 pywebview 打开本地页面。

运行：python desktop.py
依赖：pip install pywebview（Windows 使用系统自带的 WebView2 内核）
"""
import os
import threading
import time

import uvicorn
import webview

from app import app

HOST = "127.0.0.1"
PORT = int(os.getenv("DESKTOP_PORT", "8000"))
URL = f"http://{HOST}:{PORT}"


def _start_server():
    config = uvicorn.Config(app, host=HOST, port=PORT, log_level="warning")
    server = uvicorn.Server(config)
    server.run()


def main():
    server_thread = threading.Thread(target=_start_server, daemon=True)
    server_thread.start()

    # 等服务就绪后再开窗口
    import urllib.request
    for _ in range(50):
        try:
            urllib.request.urlopen(URL, timeout=0.5)
            break
        except Exception:
            time.sleep(0.2)

    webview.create_window(
        "Excel AI 助手",
        URL,
        width=1280,
        height=820,
        min_size=(900, 600),
    )
    webview.start()


if __name__ == "__main__":
    main()

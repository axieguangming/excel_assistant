"""Windows desktop launcher for the Excel AI assistant."""

from __future__ import annotations

import os
from pathlib import Path
import threading
import time

import uvicorn
import webview

os.environ["EXCEL_ASSISTANT_DESKTOP"] = "1"

from app_paths import get_data_dir
from local_model_server import (
    LocalModelServer,
    LocalModelStartupError,
    load_local_server_config,
)

os.environ["EXCEL_ASSISTANT_DATA_DIR"] = str(get_data_dir())

HOST = "127.0.0.1"
STARTUP_TIMEOUT_SECONDS = 15
SHUTDOWN_TIMEOUT_SECONDS = 8


def _show_message(message: str, flags: int) -> None:
    if os.name == "nt":
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, message, "Excel AI 助手", flags)
    else:
        print(message)


def _show_error(message: str) -> None:
    _show_message(message, 0x10)


def _show_warning(message: str) -> None:
    # 允许用户继续打开界面：模型不可用时聊天会给出具体指引。
    _show_message(message, 0x30)


def _start_local_model(data_dir: Path) -> tuple[LocalModelServer | None, str | None]:
    """按配置准备本地模型服务，返回 (管理器, 警告信息)。"""
    config = load_local_server_config()
    if config is None:
        return None, None

    server = LocalModelServer(config, log_dir=data_dir / "logs")
    try:
        server.ensure_running()
    except LocalModelStartupError as error:
        return None, str(error)
    return server, None


def _run_server(server: uvicorn.Server, errors: list[Exception]) -> None:
    try:
        server.run()
    except Exception as error:
        errors.append(error)
        raise


def _stop_server(server: uvicorn.Server, thread: threading.Thread) -> None:
    server.should_exit = True
    thread.join(timeout=SHUTDOWN_TIMEOUT_SECONDS)
    if thread.is_alive():
        raise RuntimeError("本地服务未能在规定时间内关闭。")


def main() -> None:
    server: uvicorn.Server | None = None
    server_thread: threading.Thread | None = None
    local_model: LocalModelServer | None = None
    try:
        port = int(os.getenv("DESKTOP_PORT", "8000"))
        if not 1 <= port <= 65535:
            raise ValueError("DESKTOP_PORT 必须是 1 到 65535 之间的端口号。")

        data_dir = Path(os.environ["EXCEL_ASSISTANT_DATA_DIR"])
        data_dir.mkdir(parents=True, exist_ok=True)

        local_model, model_warning = _start_local_model(data_dir)
        if model_warning:
            _show_warning(model_warning)

        from app import app

        config = uvicorn.Config(
            app,
            host=HOST,
            port=port,
            log_level="warning",
            timeout_graceful_shutdown=5,
        )
        server = uvicorn.Server(config)
        errors: list[Exception] = []
        server_thread = threading.Thread(
            target=_run_server,
            args=(server, errors),
            name="excel-assistant-api",
            daemon=True,
        )
        server_thread.start()

        deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if server.started:
                break
            if not server_thread.is_alive():
                if errors:
                    raise RuntimeError(f"本地服务启动失败：{errors[0]}") from errors[0]
                raise RuntimeError(
                    f"无法启动本地服务。端口 {port} 可能已被其他程序占用。"
                )
            time.sleep(0.1)
        else:
            raise TimeoutError("本地服务在 15 秒内未就绪。")

        window = webview.create_window(
            "Excel AI 助手",
            f"http://{HOST}:{port}",
            width=1280,
            height=820,
            min_size=(900, 600),
        )
        window.events.closed += lambda: setattr(server, "should_exit", True)
        webview.start()
    except Exception as error:
        if server is not None and server_thread is not None:
            _stop_server(server, server_thread)
        _show_error(str(error))
        raise
    else:
        if server is not None and server_thread is not None:
            _stop_server(server, server_thread)
    finally:
        # 无论是正常退出还是异常退出，都要关闭由本进程启动的模型服务。
        if local_model is not None:
            local_model.stop()


if __name__ == "__main__":
    main()

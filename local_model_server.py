"""本地模型服务（llama.cpp 的 llama-server）的生命周期管理。

用途：让打包后的程序在启动时自动拉起本地模型服务，并在退出时关闭它，
这样普通用户不需要手工开命令行。

设计要点：

- 已经有一个健康的同地址服务时直接复用，不会重复启动。
- 只有显式配置 ``auto_start`` 才拉起进程；否则给出可操作的启动指引。
- 端口被其他程序占用、程序或模型文件缺失时，抛出带修复建议的错误。
- 子进程输出写入日志文件，便于排查启动失败。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import urlopen

from app_paths import get_data_dir, read_packaged_model_config

LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
DEFAULT_STARTUP_TIMEOUT = 90.0
DEFAULT_CONTEXT_SIZE = 8192
GRACEFUL_SHUTDOWN_SECONDS = 10.0
LOG_FILENAME = "local-model.log"
LOG_MAX_BYTES = 2 * 1024 * 1024
PROBE_TIMEOUT_SECONDS = 2.0
# 连接本机关闭端口时，部分环境不是立即拒绝而是等超时，
# 所以健康检查先用很短的 TCP 探测排除"端口没在监听"的情况。
FAST_CONNECT_TIMEOUT_SECONDS = 0.3

TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


class LocalModelStartupError(RuntimeError):
    """本地模型服务无法启动，消息可直接展示给用户。"""


def _as_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in TRUE_VALUES


def _as_text(value: object) -> str:
    return "" if value is None else str(value).strip()


def _as_int(value: object, default: int) -> int:
    text = _as_text(value)
    if not text:
        return default
    try:
        parsed = int(text)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _as_float(value: object, default: float) -> float:
    text = _as_text(value)
    if not text:
        return default
    try:
        parsed = float(text)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _as_extra_args(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value if str(item).strip())
    text = str(value).strip()
    if not text:
        return ()
    # 支持 "a,b" 与 "a b" 两种写法；路径含空格时请用列表形式。
    separator = "," if "," in text else None
    return tuple(part for part in text.split(separator) if part.strip())


@dataclass(frozen=True)
class LocalServerConfig:
    base_url: str
    model: str
    auto_start: bool = False
    executable: str = ""
    model_path: str = ""
    context_size: int = DEFAULT_CONTEXT_SIZE
    startup_timeout: float = DEFAULT_STARTUP_TIMEOUT
    extra_args: tuple[str, ...] = ()

    @property
    def host(self) -> str:
        return urlsplit(self.base_url).hostname or "127.0.0.1"

    @property
    def port(self) -> int:
        parsed = urlsplit(self.base_url)
        if parsed.port:
            return parsed.port
        return 443 if parsed.scheme == "https" else 80


def _resolve_path(value: str, base_dir: Path) -> str:
    """相对路径按安装目录解析，方便把程序与模型一起打包。"""
    if not value:
        return ""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def parse_local_server_config(
    raw: Mapping[str, object],
    *,
    base_dir: Path | None = None,
) -> LocalServerConfig:
    resolved_base = base_dir or Path.cwd()
    return LocalServerConfig(
        base_url=_as_text(raw.get("base_url")).rstrip("/"),
        model=_as_text(raw.get("model")),
        auto_start=_as_bool(raw.get("auto_start")),
        executable=_resolve_path(_as_text(raw.get("local_server")), resolved_base),
        model_path=_resolve_path(
            _as_text(raw.get("local_model_path")), resolved_base
        ),
        context_size=_as_int(raw.get("context_size"), DEFAULT_CONTEXT_SIZE),
        startup_timeout=_as_float(
            raw.get("startup_timeout"), DEFAULT_STARTUP_TIMEOUT
        ),
        extra_args=_as_extra_args(raw.get("extra_args")),
    )


def _base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent


def load_local_server_config() -> LocalServerConfig | None:
    """读取当前生效的本地服务配置；远程模式返回 None。"""
    if getattr(sys, "frozen", False):
        raw = read_packaged_model_config()
        provider = _as_text(raw.get("provider") or "remote").lower()
        if provider != "local":
            return None
        return parse_local_server_config(raw, base_dir=_base_dir())

    provider = _as_text(os.getenv("LLM_PROVIDER") or "remote").lower()
    if provider != "local":
        return None
    return parse_local_server_config(
        {
            "base_url": os.getenv("LOCAL_BASE_URL", ""),
            "model": os.getenv("LOCAL_MODEL", ""),
            "auto_start": os.getenv("LOCAL_AUTO_START", ""),
            "local_server": os.getenv("LOCAL_SERVER_PATH", ""),
            "local_model_path": os.getenv("LOCAL_MODEL_PATH", ""),
            "context_size": os.getenv("LOCAL_CONTEXT_SIZE", ""),
            "startup_timeout": os.getenv("LOCAL_STARTUP_TIMEOUT", ""),
            "extra_args": os.getenv("LOCAL_EXTRA_ARGS", ""),
        },
        base_dir=_base_dir(),
    )


def _http_ok(url: str, timeout: float = PROBE_TIMEOUT_SECONDS) -> bool:
    try:
        with urlopen(url, timeout=timeout) as response:
            return 200 <= int(response.status) < 300
    except (URLError, OSError, ValueError):
        return False


def port_in_use(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def probe_endpoint(base_url: str, timeout: float = PROBE_TIMEOUT_SECONDS) -> bool:
    """判断该地址上是否已有一个可用的 OpenAI 兼容服务。

    先做一次很短的 TCP 探测：端口没在监听时立刻返回，避免每个
    健康检查都等满 HTTP 超时。
    """
    root = base_url.rstrip("/")
    if not root:
        return False
    parsed = urlsplit(root)
    if not parsed.scheme or not parsed.netloc:
        return False

    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if not host or not port_in_use(host, port, timeout=FAST_CONNECT_TIMEOUT_SECONDS):
        return False

    candidates = [root + "/models", f"{parsed.scheme}://{parsed.netloc}/health"]
    return any(_http_ok(url, timeout) for url in candidates)


class LocalModelServer:
    """负责启动、等待就绪与关闭本地模型服务进程。"""

    def __init__(
        self,
        config: LocalServerConfig,
        *,
        log_dir: str | Path | None = None,
        popen: Callable[..., subprocess.Popen] = subprocess.Popen,
        probe: Callable[[str], bool] = probe_endpoint,
        port_check: Callable[[str, int], bool] = port_in_use,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.config = config
        self.log_dir = Path(log_dir) if log_dir is not None else get_data_dir() / "logs"
        self._popen = popen
        self._probe = probe
        self._port_check = port_check
        self._clock = clock
        self._sleep = sleep
        self._process: subprocess.Popen | None = None
        self._log_handle: object | None = None
        self.owns_process = False

    @property
    def log_path(self) -> Path:
        return self.log_dir / LOG_FILENAME

    def is_healthy(self) -> bool:
        return bool(self.config.base_url) and self._probe(self.config.base_url)

    def ensure_running(self) -> bool:
        """确保服务可用。

        返回 True 表示进程由本对象启动，需要调用 ``stop`` 关闭。
        返回 False 表示复用了已有服务。
        """
        if self.is_healthy():
            return False

        if not self.config.auto_start:
            raise LocalModelStartupError(self._manual_start_hint())

        self._validate_paths()

        if self._port_check(self.config.host, self.config.port):
            raise LocalModelStartupError(
                f"端口 {self.config.port} 已被其他程序占用，但该地址不是可用的模型服务。"
                f"请修改配置中的 LOCAL_BASE_URL 端口，或关闭占用该端口的程序。"
            )

        self._spawn()
        try:
            self._wait_until_ready()
        except LocalModelStartupError:
            self.stop()
            raise
        self.owns_process = True
        return True

    def stop(self) -> None:
        process, self._process = self._process, None
        self.owns_process = False
        try:
            if process is None or process.poll() is not None:
                return
            process.terminate()
            try:
                process.wait(timeout=GRACEFUL_SHUTDOWN_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            self._close_log()

    def __enter__(self) -> "LocalModelServer":
        self.ensure_running()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    def _manual_start_hint(self) -> str:
        executable = Path(self.config.executable).name if self.config.executable else "llama-server"
        lines = [
            "未检测到本地模型服务，且未启用自动启动。",
            f"请先启动本地模型服务并监听 {self.config.base_url}，",
            f"或在配置中设置 auto_start 为 true 并提供 llama-server 与模型文件路径。",
        ]
        if self.config.model_path:
            lines.append(f"当前模型文件：{self.config.model_path}")
        else:
            lines.append(f"例如：{executable} -m <模型文件> --host {self.config.host} --port {self.config.port}")
        return "\n".join(lines)

    def _validate_paths(self) -> None:
        if not self.config.executable:
            raise LocalModelStartupError(
                "已启用本地模型自动启动，但没有配置 llama-server 程序路径。"
                "请在模型配置中设置 local_server，或设置环境变量 LOCAL_SERVER_PATH。"
            )
        if not Path(self.config.executable).is_file():
            raise LocalModelStartupError(
                f"找不到本地模型服务程序：{self.config.executable}。"
                "请重新安装程序，或修改配置中的 local_server 路径。"
            )
        if not self.config.model_path:
            raise LocalModelStartupError(
                "已启用本地模型自动启动，但没有配置模型文件路径。"
                "请在模型配置中设置 local_model_path，或设置环境变量 LOCAL_MODEL_PATH。"
            )
        if not Path(self.config.model_path).is_file():
            raise LocalModelStartupError(
                f"找不到模型文件：{self.config.model_path}。"
                "请确认模型文件已下载并放在该路径，或修改配置中的 local_model_path。"
            )
        if self.config.host not in LOOPBACK_HOSTS:
            raise LocalModelStartupError(
                "本地模型服务只允许监听本机回环地址（127.0.0.1 或 localhost）。"
            )

    def _build_command(self) -> list[str]:
        return [
            self.config.executable,
            "-m",
            self.config.model_path,
            "--host",
            self.config.host,
            "--port",
            str(self.config.port),
            "-c",
            str(self.config.context_size),
            # 启用模型自带的聊天模板，工具调用需要它。
            "--jinja",
            *self.config.extra_args,
        ]

    def _creation_flags(self) -> int:
        if os.name != "nt":
            return 0
        return int(getattr(subprocess, "CREATE_NO_WINDOW", 0))

    def _open_log(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_path
        try:
            if path.is_file() and path.stat().st_size > LOG_MAX_BYTES:
                path.unlink()
        except OSError:
            pass
        handle = open(path, "ab")
        handle.write(
            f"\n--- 启动于 {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n".encode("utf-8")
        )
        handle.flush()
        return handle

    def _close_log(self) -> None:
        handle, self._log_handle = self._log_handle, None
        if handle is None:
            return
        try:
            handle.close()
        except OSError:
            pass

    def _spawn(self) -> None:
        command = self._build_command()
        self._log_handle = self._open_log()
        try:
            self._process = self._popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                cwd=str(Path(self.config.executable).parent),
                creationflags=self._creation_flags(),
            )
        except OSError as error:
            self._close_log()
            raise LocalModelStartupError(
                f"无法启动本地模型服务：{error}。请确认程序路径与文件权限。"
            ) from error

    def _wait_until_ready(self) -> None:
        deadline = self._clock() + self.config.startup_timeout
        while self._clock() < deadline:
            process = self._process
            if process is not None and process.poll() is not None:
                raise LocalModelStartupError(
                    f"本地模型服务启动后立即退出（退出码 {process.returncode}）。"
                    f"详情见日志：{self.log_path}"
                )
            if self.is_healthy():
                return
            self._sleep(0.5)

        raise LocalModelStartupError(
            f"本地模型服务在 {self.config.startup_timeout:.0f} 秒内未就绪。"
            f"首次加载大模型可能较慢，可调大 startup_timeout。详情见日志：{self.log_path}"
        )

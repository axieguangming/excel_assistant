"""把本地模型的分发与启动串成一个可供界面查询的运行状态。

界面通过 ``/api/model/status`` 轮询状态，通过 ``/api/model/download`` 触发下载；
下载完成后再由这里启动本地模型服务。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import threading
from typing import Any, Callable

from app_paths import get_data_dir
from local_model_server import (
    LocalModelServer,
    LocalModelStartupError,
    LocalServerConfig,
    load_local_server_config,
    load_raw_local_config,
)
from model_download import (
    DownloadStatus,
    ModelDownloader,
    ModelManifest,
    manifest_from_config,
)

STATE_REMOTE = "remote"
STATE_READY = "ready"
STATE_NEEDS_DOWNLOAD = "needs_download"
STATE_DOWNLOADING = "downloading"
STATE_VERIFYING = "verifying"
STATE_STARTING = "starting"
STATE_ERROR = "error"

MODELS_SUBDIR = "models"


def resolve_model_path(
    config: LocalServerConfig,
    manifest: ModelManifest,
    data_dir: Path,
) -> Path:
    """确定模型文件的最终位置。

    配置里给了路径就用它（可随安装包分发）；否则放到用户数据目录，
    避免卸载时连同已下载的模型一起删除。
    """
    if config.model_path:
        return Path(config.model_path)
    return data_dir / MODELS_SUBDIR / manifest.effective_filename


class LocalModelRuntime:
    """本地模型的状态机：检查 → 下载 → 启动。"""

    def __init__(
        self,
        config: LocalServerConfig,
        manifest: ModelManifest,
        *,
        data_dir: str | Path | None = None,
        downloader: ModelDownloader | None = None,
        server: LocalModelServer | None = None,
        server_factory: Callable[[LocalServerConfig, Path], LocalModelServer] | None = None,
    ):
        self.config = config
        self.manifest = manifest
        self.data_dir = Path(data_dir) if data_dir is not None else get_data_dir()
        self.model_path = resolve_model_path(config, manifest, self.data_dir)

        self._lock = threading.Lock()
        self._server_error = ""
        self._last_error = ""
        self._starting = False
        # 已下载完但还没轮到启动：用于消除"下载完成但服务未起"的错误闪烁。
        self._needs_start = False

        self.downloader = downloader or ModelDownloader(manifest, self.model_path)
        # 用实际模型路径重建配置，确保 llama-server 拿到正确参数。
        effective_config = replace(config, model_path=str(self.model_path))
        log_dir = self.data_dir / "logs"
        if server is not None:
            self.server = server
        elif server_factory is not None:
            self.server = server_factory(effective_config, log_dir)
        else:
            self.server = LocalModelServer(effective_config, log_dir=log_dir)

    # ---- 查询 -------------------------------------------------------

    @property
    def needs_download(self) -> bool:
        if self.downloader.is_installed():
            return False
        return self.manifest.is_configured

    def can_download(self) -> bool:
        return self.manifest.is_configured and not self.downloader.is_installed()

    def _download_status(self) -> DownloadStatus:
        status = self.downloader.status
        if status.state == "missing" and self.downloader.is_installed():
            return DownloadStatus(
                state="ready",
                downloaded=self.model_path.stat().st_size,
                total=self.manifest.size,
            )
        return status

    def status(self) -> dict[str, Any]:
        download = self._download_status()
        state: str
        message = ""

        with self._lock:
            starting = self._starting
            needs_start = self._needs_start
            server_error = self._server_error
            last_error = self._last_error

        if download.state in {"downloading", "verifying"}:
            state = download.state
        elif download.state == "error":
            state = STATE_ERROR
            message = download.error
        elif self.server.is_healthy():
            # 健康检查优先于 starting：服务已可用时不必再显示"启动中"。
            state = STATE_READY
        elif starting or needs_start:
            state = STATE_STARTING
        elif self.downloader.is_installed():
            state = STATE_ERROR
            message = server_error or "本地模型服务未运行。"
        elif self.manifest.is_configured:
            state = STATE_NEEDS_DOWNLOAD
            message = "首次使用需要下载本地模型。"
        elif self.config.auto_start:
            state = STATE_ERROR
            message = last_error or (
                f"找不到本地模型文件：{self.model_path}，且未配置下载地址。"
            )
        else:
            state = STATE_ERROR
            message = server_error or last_error or (
                "本地模型服务未运行。请先启动服务，或在配置中启用自动启动。"
            )

        payload = download.as_dict()
        payload.update(
            {
                "provider": "local",
                "state": state,
                "message": message,
                "model": self.config.model,
                "base_url": self.config.base_url,
                "model_filename": self.model_path.name,
                "can_download": self.can_download(),
                "model_size": self.manifest.size,
            }
        )
        return payload

    # ---- 动作 -------------------------------------------------------

    def ensure_ready(self) -> tuple[bool, str | None]:
        """尝试让服务可用；返回 (是否就绪, 警告信息)。"""
        if self.server.is_healthy():
            return True, None

        if self.config.auto_start and not self.downloader.is_installed():
            if self.manifest.is_configured:
                # 交给界面引导下载，不算错误。
                return False, None
            message = (
                f"找不到本地模型文件：{self.model_path}，"
                "且未配置下载地址。请放置模型文件或重新安装包含下载配置的版本。"
            )
            with self._lock:
                self._last_error = message
            return False, message

        return self._start_server()

    def _start_server(self) -> tuple[bool, str | None]:
        with self._lock:
            if self._starting:
                return False, None
            self._starting = True
            self._needs_start = False
        try:
            self.server.ensure_running()
        except LocalModelStartupError as error:
            with self._lock:
                self._server_error = str(error)
                self._last_error = str(error)
            return False, str(error)
        except Exception as error:
            # 任何意外异常都不能让界面永远停在"启动中"。
            message = f"启动本地模型服务时发生意外错误：{error}"
            with self._lock:
                self._server_error = message
                self._last_error = message
            return False, message
        else:
            with self._lock:
                self._server_error = ""
            return True, None
        finally:
            with self._lock:
                self._starting = False

    def start_download(self) -> bool:
        """开始后台下载；已在下载或已完成时返回 False。"""
        if self.downloader.is_installed():
            return False
        if not self.manifest.is_configured:
            with self._lock:
                self._last_error = (
                    "未配置模型下载地址，请手动放置模型文件或联系发布者。"
                )
            return False
        started = self.downloader.start()
        if started:
            # 从下载开始就标记"待启动"，避免下载完成到启动之间闪现错误状态。
            with self._lock:
                self._needs_start = True
            threading.Thread(
                target=self._finish_after_download,
                name="model-setup",
                daemon=True,
            ).start()
        return started

    def _finish_after_download(self) -> None:
        try:
            self.downloader.wait()
            status = self.downloader.status
            if status.state != "ready":
                with self._lock:
                    self._last_error = status.error or "模型下载失败。"
                return
            self._start_server()
        except Exception as error:
            # 保证线程异常不会把状态永久留在 downloading/starting。
            with self._lock:
                self._last_error = f"准备本地模型时出错：{error}"
        finally:
            with self._lock:
                self._needs_start = False

    def stop(self) -> None:
        self.server.stop()


_runtime: LocalModelRuntime | None = None
_lock = threading.Lock()


def build_runtime(
    *,
    data_dir: str | Path | None = None,
    config: LocalServerConfig | None = None,
    manifest: ModelManifest | None = None,
) -> LocalModelRuntime | None:
    """按当前配置构造运行时；远程模式返回 None。"""
    resolved_config = config or load_local_server_config()
    if resolved_config is None:
        return None
    resolved_manifest = manifest
    if resolved_manifest is None:
        resolved_manifest = manifest_from_config(load_raw_local_config())
    return LocalModelRuntime(
        resolved_config,
        resolved_manifest,
        data_dir=data_dir,
    )


def set_runtime(runtime: LocalModelRuntime | None) -> None:
    global _runtime
    with _lock:
        _runtime = runtime


def get_runtime() -> LocalModelRuntime | None:
    return _runtime

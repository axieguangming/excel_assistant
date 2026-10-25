"""本地模型文件的分发：按清单下载、断点续传、完整性校验。

设计要点：

- 下载到 ``<目标>.part``，校验通过后才原子重命名，避免半个文件被当成可用模型。
- 支持 HTTP Range 续传；服务端不支持时自动从头开始。
- 提供线程安全的进度状态，供界面轮询。
- 校验失败会删除文件并报错，而不是留下一个损坏的模型。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import threading
import time
from typing import Any, BinaryIO, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

CHUNK_SIZE = 1024 * 1024
# 模型下载完成后还要解压/写盘，预留一点余量，避免刚好占满磁盘。
MIN_FREE_MARGIN_BYTES = 256 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 30.0
# 进度上报的最短间隔，避免界面被高频更新拖慢。
PROGRESS_INTERVAL_SECONDS = 0.2

DownloadState = Literal["missing", "downloading", "verifying", "ready", "error"]


@dataclass(frozen=True)
class ModelManifest:
    """模型文件的来源与校验信息。"""

    url: str
    sha256: str = ""
    size: int = 0
    filename: str = ""

    @property
    def is_configured(self) -> bool:
        return bool(self.url.strip())

    @property
    def effective_filename(self) -> str:
        if self.filename.strip():
            return self.filename.strip()
        name = Path(urlsplit(self.url).path).name
        return name or "model.gguf"


@dataclass
class DownloadStatus:
    state: DownloadState = "missing"
    downloaded: int = 0
    total: int = 0
    error: str = ""
    updated_at: float = 0.0

    @property
    def percent(self) -> float:
        if self.total <= 0:
            return 0.0
        return min(100.0, self.downloaded * 100.0 / self.total)

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "downloaded_bytes": self.downloaded,
            "total_bytes": self.total,
            "percent": round(self.percent, 1),
            "error": self.error,
        }


def _as_text(value: object) -> str:
    return "" if value is None else str(value).strip()


def _as_int(value: object, default: int = 0) -> int:
    text = _as_text(value)
    if not text:
        return default
    try:
        parsed = int(text)
    except ValueError:
        return default
    return parsed if parsed >= 0 else default


def manifest_from_config(raw: Mapping[str, object]) -> ModelManifest:
    """从模型配置读取下载清单；未配置 URL 时返回空清单。"""
    return ModelManifest(
        url=_as_text(raw.get("model_url")),
        sha256=_as_text(raw.get("model_sha256")).lower(),
        size=_as_int(raw.get("model_size")),
        filename=_as_text(raw.get("model_filename")),
    )


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ModelDownloader:
    """下载并校验单个模型文件。"""

    def __init__(
        self,
        manifest: ModelManifest,
        target_path: str | Path,
        *,
        open_url: Callable[..., BinaryIO] = urlopen,
        clock: Callable[[], float] = time.monotonic,
        chunk_size: int = CHUNK_SIZE,
    ):
        self.manifest = manifest
        self.target_path = Path(target_path)
        self._open_url = open_url
        self._clock = clock
        self._chunk_size = chunk_size
        self._lock = threading.Lock()
        self._status = DownloadStatus()
        self._thread: threading.Thread | None = None

    @property
    def partial_path(self) -> Path:
        return self.target_path.with_suffix(self.target_path.suffix + ".part")

    @property
    def status(self) -> DownloadStatus:
        with self._lock:
            return DownloadStatus(
                state=self._status.state,
                downloaded=self._status.downloaded,
                total=self._status.total,
                error=self._status.error,
                updated_at=self._status.updated_at,
            )

    def _set_status(self, **changes: Any) -> None:
        with self._lock:
            for key, value in changes.items():
                setattr(self._status, key, value)
            self._status.updated_at = self._clock()

    def is_installed(self) -> bool:
        """文件已存在且大小符合预期（未知大小时只要求文件存在）。"""
        path = self.target_path
        if not path.is_file():
            return False
        if self.manifest.size > 0:
            return path.stat().st_size == self.manifest.size
        return path.stat().st_size > 0

    def verify(self) -> tuple[bool, str]:
        """校验已安装文件的完整性。返回 (是否通过, 原因)。"""
        if not self.target_path.is_file():
            return False, "模型文件不存在"
        if self.manifest.size > 0:
            actual = self.target_path.stat().st_size
            if actual != self.manifest.size:
                return False, f"模型文件大小不符：期望 {self.manifest.size}，实际 {actual}"
        if not self.manifest.sha256:
            # 没有校验值时只能信任大小检查。
            return True, ""
        actual_digest = _sha256_of(self.target_path)
        if actual_digest != self.manifest.sha256:
            return False, "模型文件校验失败（SHA-256 不匹配），可能已损坏"
        return True, ""

    def refresh_state(self) -> DownloadStatus:
        """未开始下载时，根据磁盘现状推断初始状态。"""
        with self._lock:
            busy = self._status.state in {"downloading", "verifying"}
        if busy:
            return self.status

        if self.is_installed():
            self._set_status(
                state="ready",
                downloaded=self.target_path.stat().st_size,
                total=self.manifest.size,
                error="",
            )
        else:
            partial = self.partial_path.stat().st_size if self.partial_path.is_file() else 0
            self._set_status(
                state="missing",
                downloaded=partial,
                total=self.manifest.size,
                error="",
            )
        return self.status

    def start(self) -> bool:
        """在后台线程开始下载。已有任务在跑时返回 False。"""
        with self._lock:
            if self._status.state in {"downloading", "verifying"}:
                return False
            if self._thread is not None and self._thread.is_alive():
                return False
            self._thread = threading.Thread(
                target=self._run, name="model-download", daemon=True
            )
            self._thread.start()
            return True

    def wait(self, timeout: float | None = None) -> None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def download(self) -> Path:
        """同步下载并校验。失败时抛出 ``ModelDownloadError``。"""
        if not self.manifest.is_configured:
            raise ModelDownloadError(
                "未配置模型下载地址，无法自动获取模型文件。"
            )
        if self.is_installed():
            ok, _ = self.verify()
            if ok:
                self._set_status(
                    state="ready",
                    downloaded=self.target_path.stat().st_size,
                    total=self.manifest.size,
                    error="",
                )
                return self.target_path
            # 已有文件损坏：删掉重下。
            self._remove_quietly(self.target_path)
            self._remove_quietly(self.partial_path)

        self.target_path.parent.mkdir(parents=True, exist_ok=True)
        self._check_disk_space()
        self._download_to_partial()
        self._verify_partial()
        os.replace(self.partial_path, self.target_path)
        size = self.target_path.stat().st_size
        self._set_status(state="ready", downloaded=size, total=size, error="")
        return self.target_path

    def _run(self) -> None:
        try:
            self.download()
        except ModelDownloadError as error:
            self._set_status(state="error", error=str(error))
        except Exception as error:  # pragma: no cover - 兜底，避免线程静默退出
            self._set_status(state="error", error=f"模型下载失败：{error}")

    def _remove_quietly(self, path: Path) -> None:
        try:
            if path.is_file():
                path.unlink()
        except OSError:
            pass

    def _check_disk_space(self) -> None:
        required = self.manifest.size + MIN_FREE_MARGIN_BYTES if self.manifest.size else MIN_FREE_MARGIN_BYTES
        try:
            free = shutil.disk_usage(self.target_path.parent).free
        except OSError:
            return
        if free < required:
            raise ModelDownloadError(
                f"磁盘空间不足：需要约 {required / 1024**3:.1f} GB，"
                f"当前可用 {free / 1024**3:.1f} GB。"
            )

    def _download_to_partial(self) -> None:
        offset = self.partial_path.stat().st_size if self.partial_path.is_file() else 0
        if self.manifest.size and offset > self.manifest.size:
            # 残留文件比目标还大，不可信，重来。
            self._remove_quietly(self.partial_path)
            offset = 0

        self._set_status(
            state="downloading",
            downloaded=offset,
            total=self.manifest.size,
            error="",
        )

        try:
            response = self._request(offset)
        except ModelDownloadError:
            raise
        except (URLError, OSError) as error:
            raise ModelDownloadError(f"无法连接模型下载地址：{error}") from error

        with response:
            total = self._resolve_total(response, offset)
            mode = "ab" if offset else "wb"
            self._set_status(total=total, downloaded=offset)
            last_update = 0.0
            with self.partial_path.open(mode) as handle:
                while True:
                    chunk = response.read(self._chunk_size)
                    if not chunk:
                        break
                    handle.write(chunk)
                    offset += len(chunk)
                    now = self._clock()
                    if now - last_update >= PROGRESS_INTERVAL_SECONDS:
                        last_update = now
                        self._set_status(downloaded=offset)
            handle_size = self.partial_path.stat().st_size
            self._set_status(downloaded=handle_size)

        actual = self.partial_path.stat().st_size
        if self.manifest.size and actual != self.manifest.size:
            raise ModelDownloadError(
                f"下载不完整：期望 {self.manifest.size} 字节，实际 {actual} 字节。请重试。"
            )

    def _request(self, offset: int):
        headers = {"User-Agent": "ExcelAssistant/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = Request(self.manifest.url, headers=headers)
        try:
            return self._open_url(request, timeout=DEFAULT_TIMEOUT_SECONDS)
        except HTTPError as error:
            if error.code == 416 and offset:
                # 服务端认为范围无效：放弃续传，从头下载。
                self._remove_quietly(self.partial_path)
                return self._request(0)
            raise ModelDownloadError(
                f"下载模型失败：服务器返回 HTTP {error.code}。"
            ) from error

    def _resolve_total(self, response: Any, offset: int) -> int:
        length = self._header_length(response)
        status = getattr(response, "status", None)
        if status == 206:
            # 续传：Content-Length 只是剩余部分。
            return offset + length if length else self.manifest.size
        if length:
            return length
        return self.manifest.size

    @staticmethod
    def _header_length(response: Any) -> int:
        headers = getattr(response, "headers", None)
        if headers is None:
            return 0
        try:
            raw = headers.get("Content-Length")
        except AttributeError:
            return 0
        if not raw:
            return 0
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 0

    def _verify_partial(self) -> None:
        self._set_status(state="verifying")
        if not self.manifest.sha256:
            return
        digest = _sha256_of(self.partial_path)
        if digest != self.manifest.sha256:
            self._remove_quietly(self.partial_path)
            raise ModelDownloadError(
                "模型文件校验失败（SHA-256 不匹配），请检查下载地址或重试。"
            )


class ModelDownloadError(RuntimeError):
    """下载或校验失败，消息可直接展示给用户。"""

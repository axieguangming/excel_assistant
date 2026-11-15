"""进程内加载微调后的模型（transformers 后端）。

为什么有这个模块
----------------
原先桌面版把模型转成 GGUF，由 llama-server 子进程通过 HTTP 提供服务。实测同一份
权重、同样 20 条样本：

    transformers（进程内，bf16）   17/20 = 85%
    llama.cpp（bf16 GGUF）         13/20 = 65%

差异样本 4 条，llama.cpp 独赢 0 条。已排除的原因：量化（Q4 与 bf16 GGUF 同为
13/20）、logits 数值差异（同一 prompt 两边 top-5 logprob 最大偏差 0.1、top1 完全
相同）、tools schema、chat template、分词。剩下的解释是多轮 agent 循环中行为
逐步偏移。既然单步 logits 一致而整链行为不同，改 llama.cpp 的参数不解决问题，
换回 transformers 是唯一可靠的路径。

代价（实测于 CPU，1.544B 参数）：
    transformers   6.95 tok/s → 约 6.5 秒/次工具调用、约 16 秒/任务
    llama.cpp      约 12 tok/s → 约 3.8 秒/次调用、约 9 秒/任务
用 GPU 时 transformers 会快很多，远优于 llama.cpp 的 CPU 后端。

与 `finetune.local_llm.LocalChatModel` 的关系
--------------------------------------------
本模块只负责**加载**并把它包装成带状态机的运行时，供界面查询进度。消息序列化与
工具调用解析复用 `LocalChatModel`，因此训练/推理格式与评测完全一致（它内部同样
走 `tool_schema.describe_tool`，与训练数据的 schema 逐字节相同）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import sys
import threading
import time
from typing import Any

from app_paths import read_packaged_model_config

# 与 `local_model_runtime` 中的状态常量保持一致，前端按这些值分支。
STATE_READY = "ready"
STATE_STARTING = "starting"
STATE_ERROR = "error"
STATE_DISABLED = "disabled"

DEFAULT_MAX_NEW_TOKENS = 256
# 加载 3 GB 权重 + 建 CUDA context，留足余量。
DEFAULT_LOAD_TIMEOUT_SECONDS = 180.0

# 判断「模型目录是否是有效的 HF 模型」的依据。merged 目录里有这些文件。
REQUIRED_MODEL_FILES = ("config.json",)


class EmbeddedModelError(RuntimeError):
    passthrough = False


@dataclass(frozen=True)
class EmbeddedModelConfig:
    """进程内模型的配置。"""

    model_dir: str
    model: str = "excel-assistant"
    device: str = "auto"  # auto | cuda | cpu
    dtype: str = "bfloat16"
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    startup_timeout: float = DEFAULT_LOAD_TIMEOUT_SECONDS

    @property
    def is_valid(self) -> bool:
        return self.model_dir != ""


def _as_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _as_int(value: object, default: int) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _as_float(value: object, default: float) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent


def _resolve_path(value: str, base_dir: Path) -> str:
    if not value:
        return ""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def parse_embedded_config(
    raw: dict[str, Any],
    *,
    base_dir: Path | None = None,
) -> EmbeddedModelConfig:
    resolved_base = base_dir or Path.cwd()
    return EmbeddedModelConfig(
        model_dir=_resolve_path(_as_text(raw.get("model_dir")), resolved_base),
        model=_as_text(raw.get("model")) or "excel-assistant",
        device=_as_text(raw.get("device")).lower() or "auto",
        dtype=_as_text(raw.get("dtype")).lower() or "bfloat16",
        max_new_tokens=_as_int(raw.get("max_new_tokens"), DEFAULT_MAX_NEW_TOKENS),
        startup_timeout=_as_float(
            raw.get("startup_timeout"), DEFAULT_LOAD_TIMEOUT_SECONDS
        ),
    )


def local_backend(raw: dict[str, Any] | None = None) -> str:
    """返回本地模式的后端名：`transformers` 或 `llama_server`。

    默认 `llama_server` 是为了兼容旧安装包与旧 `.env`；新的发行配置会显式写
    `"backend": "transformers"`。
    """
    if raw is None:
        raw = load_raw_config()
    value = _as_text(raw.get("backend")).lower()
    return value or "llama_server"


def load_raw_config() -> dict[str, Any]:
    """读取当前生效的本地模型原始配置；远程模式返回空字典。"""
    if getattr(sys, "frozen", False):
        try:
            raw = read_packaged_model_config()
        except RuntimeError:
            return {}
        provider = _as_text(raw.get("provider") or "remote").lower()
        return dict(raw) if provider == "local" else {}

    provider = _as_text(os.getenv("LLM_PROVIDER") or "remote").lower()
    if provider != "local":
        return {}
    return {
        "backend": os.getenv("LOCAL_BACKEND", ""),
        "model_dir": os.getenv("LOCAL_MODEL_DIR", ""),
        "model": os.getenv("LOCAL_MODEL", ""),
        "device": os.getenv("LOCAL_DEVICE", ""),
        "dtype": os.getenv("LOCAL_DTYPE", ""),
        "max_new_tokens": os.getenv("LOCAL_MAX_NEW_TOKENS", ""),
        "startup_timeout": os.getenv("LOCAL_STARTUP_TIMEOUT", ""),
    }


def load_embedded_config() -> EmbeddedModelConfig | None:
    """读取进程内模型配置；远程模式或非 transformers 后端返回 None。"""
    raw = load_raw_config()
    if not raw:
        return None
    if local_backend(raw) != "transformers":
        return None
    return parse_embedded_config(raw, base_dir=_base_dir())


def model_dir_looks_valid(model_dir: str) -> bool:
    path = Path(model_dir) if model_dir else None
    if path is None or not path.is_dir():
        return False
    return all((path / name).is_file() for name in REQUIRED_MODEL_FILES)


def _directory_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def _resolve_device(requested: str, torch: Any) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise EmbeddedModelError(
                "配置要求使用 GPU，但当前环境检测不到可用的 CUDA 设备。"
                "请安装 GPU 驱动，或把配置中的 device 改为 auto / cpu。"
            )
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dtype(requested: str, device: str, torch: Any) -> Any:
    if device == "cuda":
        # bf16 需要 Ampere 及以上；不支持的卡退到 fp16，避免加载即报错。
        if requested == "bfloat16" and not torch.cuda.is_bf16_supported():
            return torch.float16
        return {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
            requested, torch.bfloat16
        )
    # CPU 上 fp16 很慢且部分算子不支持，统一用 fp32/bf16 中的可用项。
    if requested == "float32":
        return torch.float32
    return torch.bfloat16


def build_chat_model(config: EmbeddedModelConfig):
    """加载权重并返回可直接交给 Agent 的聊天模型。

    加载的结果会被缓存：Agent 每次调用 `init_llm()` 都返回同一实例，
    避免重复占用数 GB 显存。
    """
    with _CACHE_LOCK:
        cached = _CACHE.get(config.model_dir)
        if cached is not None:
            return cached
        model = _load(config)
        _CACHE[config.model_dir] = model
        return model


_CACHE: dict[str, Any] = {}
_CACHE_LOCK = threading.Lock()


def _load(config: EmbeddedModelConfig):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from finetune.local_llm import LocalChatModel

    if not model_dir_looks_valid(config.model_dir):
        raise EmbeddedModelError(
            f"找不到本地模型目录或文件不完整：{config.model_dir}。"
            "请确认安装包内的 models 目录已解压完整。"
        )

    device = _resolve_device(config.device, torch)
    dtype = _resolve_dtype(config.dtype, device, torch)

    tokenizer = AutoTokenizer.from_pretrained(config.model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_dir,
        dtype=dtype,
    )
    model.eval()
    # 先按 CPU 加载再搬到目标设备：这样不依赖 accelerate，也不会在多卡机器上
    # 把单模型拆开。加载期占一份主机内存，模型量级（约 3 GB）可以接受。
    if device != "cpu":
        model.to(device)

    return LocalChatModel(
        model,
        tokenizer,
        max_new_tokens=config.max_new_tokens,
        model_name=config.model,
    )


class _StubDownloader:
    """进程内模式的模型随包分发，不支持下载。

    保留同名接口是为了让 `app.py` 的下载端点无需分支。
    """

    def __init__(self, model_dir: str):
        self._model_dir = model_dir

    def is_installed(self) -> bool:
        return model_dir_looks_valid(self._model_dir)


class _StubManifest:
    def __init__(self, model_dir: str):
        self._model_dir = model_dir
        self.size = _directory_size(Path(model_dir)) if model_dir else 0

    @property
    def is_configured(self) -> bool:
        return False


class EmbeddedModelRuntime:
    """进程内模型的状态机：检查 → 后台加载 → 就绪。

    与 `local_model_runtime.LocalModelRuntime` 保持相同的对外接口
    （`status()` / `ensure_ready()` / `stop()` / `downloader` / `manifest`），
    这样 `app.py` 与 `desktop.py` 不需要知道后端是子进程还是进程内。
    """

    def __init__(
        self,
        config: EmbeddedModelConfig,
        *,
        loader=None,
    ):
        self.config = config
        self.downloader = _StubDownloader(config.model_dir)
        self.manifest = _StubManifest(config.model_dir)

        self._loader = loader or build_chat_model
        self._lock = threading.Lock()
        self._model: Any = None
        self._loading = False
        self._loaded = False
        self._error = ""
        self._elapsed = 0.0
        self._thread: threading.Thread | None = None

    # ---- 查询 -------------------------------------------------------

    @property
    def model_loaded(self) -> bool:
        with self._lock:
            return self._loaded and self._model is not None

    def get_model(self):
        """已加载的模型实例；未就绪时返回 None。"""
        with self._lock:
            return self._model

    def status(self) -> dict[str, Any]:
        with self._lock:
            loading = self._loading
            loaded = self._loaded
            error = self._error
            elapsed = self._elapsed
            model = self._model

        valid = model_dir_looks_valid(self.config.model_dir)
        total = self.manifest.size

        if error:
            state = STATE_ERROR
            message = error
        elif loaded and model is not None:
            state = STATE_READY
            device = getattr(model, "device", "") or ""
            message = f"模型已就绪（{device or 'unknown'}）。" if device else ""
        elif loading:
            state = STATE_STARTING
            message = "正在加载本地模型，首次启动需要几秒到几十秒。"
        elif not valid:
            state = STATE_ERROR
            message = (
                f"找不到本地模型目录或文件不完整：{self.config.model_dir}。"
                "请确认安装包内的 models 目录已解压完整。"
            )
        else:
            # 文件齐全但尚未开始加载：按「启动中」上报，避免界面闪一下红色错误。
            # 正常流程里 desktop 启动时会立刻调用 ensure_ready()，这个窗口很短。
            state = STATE_STARTING
            message = "本地模型待加载。"

        return {
            "provider": "local",
            "state": state,
            "message": message,
            "model": self.config.model,
            "backend": "transformers",
            "device": getattr(model, "device", "") if model is not None else "",
            # 前端用这两个字段画进度条；进程内加载没有下载阶段，按完成计。
            "downloaded_bytes": total if loaded else 0,
            "total_bytes": total,
            "percent": 100.0 if loaded else 0.0,
            "error": error,
            "can_download": False,
            "model_filename": Path(self.config.model_dir).name
            if self.config.model_dir
            else "",
            "model_size": total,
            "load_seconds": round(elapsed, 1),
        }

    # ---- 动作 -------------------------------------------------------

    def ensure_ready(self) -> tuple[bool, str | None]:
        """启动后台加载并立即返回；状态通过 `status()` 查询。

        加载 3 GB 权重需要数秒到数十秒，放在启动流程里同步等待会让窗口迟迟不
        出现。这里改为后台线程，界面显示「正在加载」。
        """
        with self._lock:
            if self._loaded and self._model is not None:
                return True, None
            if self._error:
                return False, self._error
            if not model_dir_looks_valid(self.config.model_dir):
                message = (
                    f"找不到本地模型目录或文件不完整：{self.config.model_dir}。"
                    "请确认安装包内的 models 目录已解压完整。"
                )
                self._error = message
                return False, message
            if self._loading:
                return False, None
            self._loading = True
            self._error = ""

        self._thread = threading.Thread(
            target=self._load_worker,
            name="embedded-model-loader",
            daemon=True,
        )
        self._thread.start()
        return False, None

    def _load_worker(self) -> None:
        started = time.monotonic()
        try:
            model = self._loader(self.config)
        except Exception as error:  # noqa: BLE001 - 不能让状态永远停在加载中
            message = f"加载本地模型失败：{error}"
            with self._lock:
                self._error = message
                self._loading = False
                self._elapsed = time.monotonic() - started
            return
        with self._lock:
            self._model = model
            self._loaded = True
            self._loading = False
            self._error = ""
            self._elapsed = time.monotonic() - started

    def wait_until_ready(self, timeout: float | None = None) -> bool:
        """等待加载完成，供脚本与测试使用；界面走轮询不需要它。"""
        deadline = time.monotonic() + (
            timeout if timeout is not None else self.config.startup_timeout
        )
        while time.monotonic() < deadline:
            with self._lock:
                if self._loaded and self._model is not None:
                    return True
                if self._error:
                    return False
                if not self._loading:
                    return False
            time.sleep(0.1)
        return False

    def start_download(self) -> bool:
        # 模型随包分发，没有下载阶段。
        return False

    def stop(self) -> None:
        """释放显存。

        Agent 侧还持有模型引用，因此这里只做显存回收提示，不强行置空，
        避免停止瞬间的请求拿到半销毁的对象。
        """
        with self._lock:
            model = self._model
        if model is None:
            return
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - 清理失败不应影响退出
            pass

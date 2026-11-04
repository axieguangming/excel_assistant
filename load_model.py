import json
import os
import sys
from urllib.parse import urlsplit

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.exceptions import ModelConnectionError

from app_paths import get_packaged_model_config

""" LLM 提供方配置：remote（云端 API）/ local（本地 Ollama、llama.cpp 等 OpenAI 兼容服务）"""

REMOTE_PROVIDER = "remote"
LOCAL_PROVIDER = "local"
SUPPORTED_PROVIDERS = (REMOTE_PROVIDER, LOCAL_PROVIDER)

DEFAULT_REMOTE_MODEL = "deepseek-flash"
DEFAULT_LOCAL_MODEL = "qwen2.5-coder:1.5b"
DEFAULT_LOCAL_BASE_URL = "http://localhost:11434/v1"
# 本地服务不校验密钥，占位值即可；llama.cpp 与 Ollama 都接受任意非空字符串。
LOCAL_API_KEY_PLACEHOLDER = "local"
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


class ModelConfigurationError(RuntimeError):
    pass


def _is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def _load_packaged_config() -> dict[str, str]:
    try:
        return get_packaged_model_config()
    except RuntimeError as error:
        raise ModelConfigurationError(str(error)) from error


def _normalize_provider(value: str | None, *, source: str) -> str:
    provider = (value or REMOTE_PROVIDER).strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise ModelConfigurationError(
            f"{source}中的 provider 无效：{provider}。只支持 remote 或 local。"
        )
    return provider


def _validate_base_url(
    base_url: str,
    *,
    provider: str,
    require_loopback: bool = False,
) -> None:
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ModelConfigurationError("模型 API 地址必须是有效的 http(s) 地址。")
    if (
        parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ModelConfigurationError(
            "模型 API 地址不能包含凭据、查询参数或片段。"
        )
    if provider == REMOTE_PROVIDER:
        if parsed.scheme != "https":
            raise ModelConfigurationError("远程模型 API 地址必须使用 HTTPS。")
        return
    if require_loopback and parsed.hostname not in LOOPBACK_HOSTS:
        # 安装包里的本地配置只允许回环地址，避免把数据发往任意 HTTP 端点。
        raise ModelConfigurationError(
            "本地模型 API 地址必须是本机回环地址（127.0.0.1 或 localhost）。"
        )


# 远程 API（DeepSeek 等 OpenAI 兼容接口）
def _load_remote_config(package_config: dict[str, str] | None = None):
    if package_config is not None:
        model_name = package_config.get("model", "")
        api_key = package_config.get("api_key", "")
        base_url = package_config.get("base_url", "")
    else:
        model_name = os.getenv("DEEPSEEK_MODEL", DEFAULT_REMOTE_MODEL).strip()
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        base_url = os.getenv("DEEPSEEK_BASE_URL", "").strip()
        api_json = os.getenv("deepseek_api")
        if api_json:
            try:
                api = json.loads(api_json)
            except json.JSONDecodeError as error:
                raise ModelConfigurationError(
                    "deepseek_api 环境变量必须是包含 api_key 和 base_url 的 JSON。"
                ) from error
            if not isinstance(api, dict):
                raise ModelConfigurationError(
                    "deepseek_api 环境变量必须是包含 api_key 和 base_url 的 JSON 对象。"
                )
            api_key = api_key or str(api.get("api_key", "")).strip()
            base_url = base_url or str(api.get("base_url", "")).strip()
    if not api_key or not base_url or not model_name:
        raise ModelConfigurationError(
            "远程模型尚未配置。请设置 DEEPSEEK_API_KEY 和 DEEPSEEK_BASE_URL。"
        )
    _validate_base_url(base_url, provider=REMOTE_PROVIDER)
    return {
        "model": model_name,
        "model_provider": "openai",
        "temperature": 0,
        "api_key": api_key,
        "base_url": base_url,
        # thinking 开关是 DeepSeek 专有参数，只发给远程
        "extra_body": {"thinking": {"type": "disabled"}},
    }


# 本地模型（Ollama / llama.cpp server 等 OpenAI 兼容接口）
def _load_local_config(
    package_config: dict[str, str] | None = None,
    *,
    require_loopback: bool = False,
):
    if package_config is not None:
        model_name = package_config.get("model", "")
        base_url = package_config.get("base_url", "")
        api_key = package_config.get("api_key", "") or LOCAL_API_KEY_PLACEHOLDER
    else:
        model_name = os.getenv("LOCAL_MODEL", DEFAULT_LOCAL_MODEL).strip()
        base_url = os.getenv("LOCAL_BASE_URL", DEFAULT_LOCAL_BASE_URL).strip()
        api_key = os.getenv("LOCAL_API_KEY", "").strip() or LOCAL_API_KEY_PLACEHOLDER

    if not model_name or not base_url:
        raise ModelConfigurationError(
            "本地模型尚未配置。请设置 LOCAL_MODEL 和 LOCAL_BASE_URL，"
            "或在安装包的 llm_config.json 中写入这两项。"
        )
    _validate_base_url(
        base_url,
        provider=LOCAL_PROVIDER,
        require_loopback=require_loopback,
    )
    return {
        "model": model_name,
        "model_provider": "openai",
        "temperature": 0,
        "api_key": api_key,
        "base_url": base_url,
    }


def describe_model_error(
    error: BaseException,
    *,
    provider: str | None = None,
) -> str | None:
    """把模型连接失败翻译成用户能看懂的提示；其他错误返回 None。

    打包后的本地模型模式下，最常见的问题是本地服务没有启动。
    """
    if isinstance(error, ModelConfigurationError):
        return str(error)

    # langchain_core 为连接失败提供了统一基类，openai 等提供方的异常都继承它。
    offline_types: tuple[type[BaseException], ...] = (
        ModelConnectionError,
        ConnectionError,
        TimeoutError,
    )
    offline_markers = (
        "APIConnectionError",
        "APITimeoutError",
        "OpenAIConnectionError",
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "RemoteProtocolError",
    )
    is_connection_failure = isinstance(error, offline_types) or (
        type(error).__name__ in offline_markers
    )
    if not is_connection_failure:
        if isinstance(error, OSError):
            return "无法连接模型服务。请检查网络，或确认本地模型服务正在运行。"
        return None

    resolved = provider
    if resolved is None:
        try:
            resolved = current_provider()
        except ModelConfigurationError:
            resolved = REMOTE_PROVIDER

    if resolved == LOCAL_PROVIDER:
        return (
            "无法连接本地模型服务。请确认本地模型服务已启动并监听配置的地址，然后重试。"
        )
    return "无法连接模型服务。请检查网络连接后重试。"


def current_provider() -> str:
    """返回当前生效的提供方，用于错误提示等展示逻辑。"""
    if _is_frozen():
        try:
            package_config = get_packaged_model_config()
        except RuntimeError:
            return REMOTE_PROVIDER
        raw = package_config.get("provider")
    else:
        raw = os.getenv("LLM_PROVIDER")
    try:
        return _normalize_provider(raw, source="模型配置")
    except ModelConfigurationError:
        return REMOTE_PROVIDER


def init_llm():
    if _is_frozen():
        package_config = _load_packaged_config()
        provider = _normalize_provider(
            package_config.get("provider"), source="安装包模型配置"
        )
        config = (
            _load_remote_config(package_config)
            if provider == REMOTE_PROVIDER
            else _load_local_config(package_config, require_loopback=True)
        )
        return init_chat_model(**config)

    load_dotenv()
    provider = _normalize_provider(
        os.getenv("LLM_PROVIDER"), source="LLM_PROVIDER 环境变量"
    )
    config = (
        _load_local_config() if provider == LOCAL_PROVIDER else _load_remote_config()
    )
    return init_chat_model(**config)

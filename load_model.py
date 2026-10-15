import json
import os
import sys
from urllib.parse import urlsplit

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from app_paths import get_packaged_model_config

""" LLM 提供方配置：remote（云端 API）/ local（本地 Ollama 等 OpenAI 兼容服务）"""


class ModelConfigurationError(RuntimeError):
    pass


# 远程 API（DeepSeek 等 OpenAI 兼容接口）
def _load_remote_config():
    if getattr(sys, "frozen", False):
        try:
            package_config = get_packaged_model_config()
        except RuntimeError as error:
            raise ModelConfigurationError(str(error)) from error
        model_name = package_config["model"]
        api_key = package_config["api_key"]
        base_url = package_config["base_url"]
    else:
        model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash").strip()
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
    parsed_base_url = urlsplit(base_url)
    if parsed_base_url.scheme != "https" or not parsed_base_url.netloc:
        raise ModelConfigurationError("远程模型 API 地址必须使用 HTTPS。")
    if (
        parsed_base_url.username
        or parsed_base_url.password
        or parsed_base_url.query
        or parsed_base_url.fragment
    ):
        raise ModelConfigurationError(
            "远程模型 API 地址不能包含凭据、查询参数或片段。"
        )
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
def _load_local_config():
    model_name = os.getenv("LOCAL_MODEL", "qwen2.5-coder:1.5b")
    base_url = os.getenv("LOCAL_BASE_URL", "http://localhost:11434/v1")
    return {
        "model": model_name,
        "model_provider": "openai",
        "temperature": 0,
        "api_key": "ollama",          # 本地服务不校验，占位即可
        "base_url": base_url,
    }

def init_llm():
    frozen = getattr(sys, "frozen", False)
    if not frozen:
        load_dotenv()
    provider = "remote" if frozen else os.getenv("LLM_PROVIDER", "remote").strip().lower()
    if provider == "local":
        config = _load_local_config()
    elif provider == "remote":
        config = _load_remote_config()
    else:
        raise ModelConfigurationError(
            "LLM_PROVIDER 只支持 remote 或 local。"
        )
    return init_chat_model(**config)

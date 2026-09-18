from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
import os
import json

""" LLM 提供方配置：remote（云端 API）/ local（本地 Ollama 等 OpenAI 兼容服务）"""

# 远程 API（DeepSeek 等 OpenAI 兼容接口）
def _load_remote_config():
    model_name = "deepseek-v4-flash"
    api_json = os.getenv("deepseek_api")
    if not api_json:
        raise RuntimeError("未找到 deepseek_api 环境变量，请在 .env 中配置。")
    api = json.loads(api_json)
    return {
        "model": model_name,
        "model_provider": "openai",
        "temperature": 0,
        "api_key": api["api_key"],
        "base_url": api["base_url"],
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
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "remote").strip().lower()
    config = _load_local_config() if provider == "local" else _load_remote_config()
    return init_chat_model(**config)

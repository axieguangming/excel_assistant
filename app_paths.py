from __future__ import annotations

import json
import os
from pathlib import Path
import sys


def get_resource_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent


def get_data_dir() -> Path:
    configured = os.getenv("EXCEL_ASSISTANT_DATA_DIR")
    if configured:
        return Path(configured).expanduser()
    if getattr(sys, "frozen", False) or os.getenv("EXCEL_ASSISTANT_DESKTOP") == "1":
        local_app_data = os.getenv("LOCALAPPDATA")
        base = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
        return base / "ExcelAssistant"
    return get_resource_dir()


RESOURCE_DIR = get_resource_dir()
DATA_DIR = get_data_dir()
STATIC_DIR = RESOURCE_DIR / "static"


def get_packaged_model_config() -> dict[str, str]:
    config_path = Path(sys.executable).parent / "llm_config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"无法读取安装包中的模型配置文件 {config_path}：{error}") from error
    if not isinstance(config, dict):
        raise RuntimeError("安装包中的模型配置文件格式无效。")
    return {
        key: str(config.get(key, "")).strip()
        for key in ("api_key", "base_url", "model")
    }

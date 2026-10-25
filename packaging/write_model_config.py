from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit

from dotenv import dotenv_values

REMOTE_PROVIDER = "remote"
LOCAL_PROVIDER = "local"
SUPPORTED_PROVIDERS = (REMOTE_PROVIDER, LOCAL_PROVIDER)

DEFAULT_LOCAL_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_LOCAL_MODEL = "qwen2.5-1.5b-instruct"
LOCAL_API_KEY_PLACEHOLDER = "local"
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def merge_build_environment(
    file_values: Mapping[str, str | None],
    process_environment: Mapping[str, str],
) -> dict[str, str]:
    environment = {
        key: value
        for key, value in file_values.items()
        if value is not None
    }
    environment.update(
        {
            key: value
            for key, value in process_environment.items()
            if value.strip()
        }
    )
    return environment


def _resolve_provider(environment: Mapping[str, str]) -> str:
    raw = (
        environment.get("EXCEL_ASSISTANT_LLM_PROVIDER")
        or environment.get("LLM_PROVIDER")
        or REMOTE_PROVIDER
    )
    provider = raw.strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise SystemExit(
            f"LLM_PROVIDER must be one of {', '.join(SUPPORTED_PROVIDERS)}, "
            f"got {provider!r}."
        )
    return provider


def _checked_url(base_url: str, *, provider: str) -> str:
    parsed = urlsplit(base_url)
    if not base_url or parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise SystemExit(f"{provider} base URL must be a valid http(s) URL.")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise SystemExit(
            "Base URL must not contain credentials, query parameters, or a fragment."
        )
    if provider == REMOTE_PROVIDER and parsed.scheme != "https":
        raise SystemExit("Remote model base URL must use HTTPS.")
    if provider == LOCAL_PROVIDER and parsed.hostname not in LOOPBACK_HOSTS:
        # 本地模型只应指向本机；安装包内不应出现任意 HTTP 端点。
        raise SystemExit(
            "Local model base URL must be a loopback address "
            "(127.0.0.1 or localhost)."
        )
    return base_url.rstrip("/")


def _build_local_config(environment: Mapping[str, str]) -> dict[str, object]:
    base_url = (
        environment.get("LOCAL_BASE_URL", "").strip() or DEFAULT_LOCAL_BASE_URL
    )
    model = environment.get("LOCAL_MODEL", "").strip() or DEFAULT_LOCAL_MODEL
    api_key = environment.get("LOCAL_API_KEY", "").strip() or LOCAL_API_KEY_PLACEHOLDER
    auto_start = environment.get("LOCAL_AUTO_START", "").strip().lower() in TRUE_VALUES
    if not model:
        raise SystemExit("LOCAL_MODEL must not be empty.")

    config: dict[str, object] = {
        "provider": LOCAL_PROVIDER,
        "api_key": api_key,
        "base_url": _checked_url(base_url, provider=LOCAL_PROVIDER),
        "model": model,
        "auto_start": auto_start,
    }

    # 以下字段只在显式配置时写入，避免把空路径塞进安装包。
    for environment_key, config_key in (
        ("LOCAL_SERVER_PATH", "local_server"),
        ("LOCAL_MODEL_PATH", "local_model_path"),
        ("LOCAL_MODEL_URL", "model_url"),
        ("LOCAL_MODEL_SHA256", "model_sha256"),
        ("LOCAL_MODEL_FILENAME", "model_filename"),
    ):
        value = environment.get(environment_key, "").strip()
        if value:
            if config_key == "model_sha256":
                value = value.lower()
            config[config_key] = value

    model_size = environment.get("LOCAL_MODEL_SIZE", "").strip()
    if model_size:
        config["model_size"] = _checked_positive_int("LOCAL_MODEL_SIZE", model_size)

    # 首次启动下载：允许构建机没有模型文件，但要有下载地址。
    model_url = str(config.get("model_url", ""))
    if model_url:
        parsed_url = urlsplit(model_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise SystemExit("LOCAL_MODEL_URL must be a valid http(s) URL.")
        if parsed_url.scheme != "https" and parsed_url.hostname not in LOOPBACK_HOSTS:
            # 明文传输可被篡改；只对本机镜像（测试用）放行。
            raise SystemExit(
                "LOCAL_MODEL_URL must use HTTPS unless it points at a loopback "
                "address used for local testing."
            )

    has_model_file = bool(config.get("local_model_path"))
    if auto_start and not (config.get("local_server") and (has_model_file or model_url)):
        raise SystemExit(
            "LOCAL_AUTO_START requires LOCAL_SERVER_PATH and either "
            "LOCAL_MODEL_PATH (bundled model) or LOCAL_MODEL_URL "
            "(first-run download) so the packaged app knows what to launch."
        )

    if model_url and not config.get("model_sha256"):
        # 没有校验值就无法保证下载完整，这里只提示，不阻断（本地搭的镜像可能没有）。
        print(
            "WARNING: LOCAL_MODEL_SHA256 is empty; the downloaded model will "
            "only be size-checked.",
            file=sys.stderr,
        )

    context_size = environment.get("LOCAL_CONTEXT_SIZE", "").strip()
    if context_size:
        config["context_size"] = _checked_positive_int("LOCAL_CONTEXT_SIZE", context_size)

    startup_timeout = environment.get("LOCAL_STARTUP_TIMEOUT", "").strip()
    if startup_timeout:
        config["startup_timeout"] = _checked_positive_float(
            "LOCAL_STARTUP_TIMEOUT", startup_timeout
        )

    extra_args = environment.get("LOCAL_EXTRA_ARGS", "").strip()
    if extra_args:
        config["extra_args"] = [
            part.strip() for part in extra_args.split(",") if part.strip()
        ]

    return config


def _checked_positive_int(name: str, value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise SystemExit(f"{name} must be an integer, got {value!r}.") from error
    if parsed <= 0:
        raise SystemExit(f"{name} must be greater than 0.")
    return parsed


def _checked_positive_float(name: str, value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise SystemExit(f"{name} must be a number, got {value!r}.") from error
    if parsed <= 0:
        raise SystemExit(f"{name} must be greater than 0.")
    return parsed


def _build_remote_config(environment: Mapping[str, str]) -> dict[str, object]:
    api_key = environment.get("DEEPSEEK_API_KEY", "").strip()
    base_url = environment.get("DEEPSEEK_BASE_URL", "").strip()
    model = environment.get("DEEPSEEK_MODEL", "deepseek-v4-flash").strip()

    legacy_config = environment.get("deepseek_api", "")
    if legacy_config:
        try:
            legacy = json.loads(legacy_config)
        except json.JSONDecodeError as error:
            raise SystemExit("The deepseek_api setting in .env is not valid JSON.") from error
        if not isinstance(legacy, dict):
            raise SystemExit("The deepseek_api setting in .env must be a JSON object.")
        api_key = api_key or str(legacy.get("api_key", "")).strip()
        base_url = base_url or str(legacy.get("base_url", "")).strip()

    parsed = urlsplit(base_url)
    if not api_key or not model or parsed.scheme != "https" or not parsed.netloc:
        raise SystemExit(
            "Configure DEEPSEEK_API_KEY and DEEPSEEK_BASE_URL (HTTPS) in the build "
            "environment or untracked .env file."
        )
    return {
        "provider": REMOTE_PROVIDER,
        "api_key": api_key,
        "base_url": _checked_url(base_url, provider=REMOTE_PROVIDER),
        "model": model,
    }


def build_model_config(environment: Mapping[str, str]) -> dict[str, object]:
    provider = _resolve_provider(environment)
    if provider == LOCAL_PROVIDER:
        return _build_local_config(environment)
    return _build_remote_config(environment)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: write_model_config.py <output-path>")

    file_values = dotenv_values(Path(__file__).resolve().parent.parent / ".env")
    environment = merge_build_environment(file_values, os.environ)
    config = build_model_config(environment)
    output_path = Path(sys.argv[1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(config, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

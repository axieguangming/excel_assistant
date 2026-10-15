from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit

from dotenv import dotenv_values


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


def build_model_config(environment: dict[str, str]) -> dict[str, str]:
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
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise SystemExit(
            "DEEPSEEK_BASE_URL must not contain credentials, query parameters, or a fragment."
        )

    return {
        "api_key": api_key,
        "base_url": base_url.rstrip("/"),
        "model": model,
    }


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

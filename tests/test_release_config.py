import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from load_model import ModelConfigurationError, _load_remote_config, init_llm

MODEL_CONFIG_WRITER = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "packaging" / "write_model_config.py")
)


class ReleaseModelConfigurationTests(unittest.TestCase):
    def test_remote_model_secret_must_be_configured(self):
        with (
            patch("load_model.load_dotenv"),
            patch.dict(os.environ, {}, clear=True),
        ):
            with self.assertRaisesRegex(ModelConfigurationError, "模型尚未配置"):
                _load_remote_config()

    def test_remote_model_endpoint_requires_https(self):
        with patch.dict(
            os.environ,
            {
                "DEEPSEEK_API_KEY": "test-key",
                "DEEPSEEK_BASE_URL": "http://model.example",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ModelConfigurationError, "HTTPS"):
                _load_remote_config()

    def test_frozen_release_always_uses_bundled_remote_model(self):
        with (
            patch("load_model.sys.frozen", True, create=True),
            patch.dict(os.environ, {"LLM_PROVIDER": "local"}),
            patch(
                "load_model.get_packaged_model_config",
                return_value={
                    "api_key": "test-build-key",
                    "base_url": "https://api.example.test/v1",
                    "model": "test-model",
                },
            ),
            patch("load_model.init_chat_model") as init_chat_model,
        ):
            init_llm()

        self.assertEqual(init_chat_model.call_args.kwargs["model"], "test-model")
        self.assertEqual(
            init_chat_model.call_args.kwargs["base_url"],
            "https://api.example.test/v1",
        )
        self.assertEqual(
            init_chat_model.call_args.kwargs["api_key"],
            "test-build-key",
        )

    def test_release_config_writer_requires_credentials(self):
        with self.assertRaisesRegex(SystemExit, "Configure DEEPSEEK_API_KEY"):
            MODEL_CONFIG_WRITER["build_model_config"]({})

    def test_release_config_writer_uses_dotenv_when_inherited_values_are_blank(self):
        environment = MODEL_CONFIG_WRITER["merge_build_environment"](
            {
                "deepseek_api": (
                    '{"api_key":"test-file-key",'
                    '"base_url":"https://api.example.test/v1"}'
                ),
                "DEEPSEEK_MODEL": "file-model",
            },
            {
                "deepseek_api": "",
                "DEEPSEEK_API_KEY": "",
                "DEEPSEEK_BASE_URL": "",
            },
        )

        self.assertEqual(
            MODEL_CONFIG_WRITER["build_model_config"](environment),
            {
                "api_key": "test-file-key",
                "base_url": "https://api.example.test/v1",
                "model": "file-model",
            },
        )

    def test_release_config_writer_prefers_non_empty_process_values(self):
        environment = MODEL_CONFIG_WRITER["merge_build_environment"](
            {
                "DEEPSEEK_API_KEY": "test-file-key",
                "DEEPSEEK_BASE_URL": "https://file.example.test/v1",
            },
            {
                "DEEPSEEK_API_KEY": "test-process-key",
                "DEEPSEEK_BASE_URL": "https://process.example.test/v1",
            },
        )

        self.assertEqual(environment["DEEPSEEK_API_KEY"], "test-process-key")
        self.assertEqual(
            environment["DEEPSEEK_BASE_URL"],
            "https://process.example.test/v1",
        )

    def test_release_config_writer_embeds_explicit_build_credentials(self):
        script = Path(__file__).resolve().parents[1] / "packaging" / "write_model_config.py"
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "llm_config.json"
            environment = os.environ.copy()
            environment["DEEPSEEK_API_KEY"] = "test-build-key"
            environment["DEEPSEEK_BASE_URL"] = "https://api.example.test/v1"
            environment["DEEPSEEK_MODEL"] = "test-model"
            result = subprocess.run(
                [sys.executable, str(script), str(output_path)],
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("test-build-key", result.stdout + result.stderr)
            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8")),
                {
                    "api_key": "test-build-key",
                    "base_url": "https://api.example.test/v1",
                    "model": "test-model",
                },
            )


if __name__ == "__main__":
    unittest.main()

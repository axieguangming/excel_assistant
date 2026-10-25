import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from load_model import (
    ModelConfigurationError,
    _load_remote_config,
    describe_model_error,
    init_llm,
)

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


class FrozenLocalModelTests(unittest.TestCase):
    """打包程序必须能按安装包内的配置使用本地模型。"""

    local_config = {
        "provider": "local",
        "api_key": "",
        "base_url": "http://127.0.0.1:8080/v1",
        "model": "qwen2.5-1.5b-instruct",
    }

    def test_frozen_release_uses_bundled_local_model(self):
        with (
            patch("load_model.sys.frozen", True, create=True),
            patch(
                "load_model.get_packaged_model_config",
                return_value=dict(self.local_config),
            ),
            patch("load_model.init_chat_model") as init_chat_model,
        ):
            init_llm()

        kwargs = init_chat_model.call_args.kwargs
        self.assertEqual(kwargs["model"], "qwen2.5-1.5b-instruct")
        self.assertEqual(kwargs["base_url"], "http://127.0.0.1:8080/v1")
        # 本地服务不校验密钥，但 OpenAI 客户端要求非空。
        self.assertTrue(kwargs["api_key"])
        self.assertEqual(kwargs["temperature"], 0)
        # thinking 是远程模式专用的 DeepSeek 参数，不应发给本地服务。
        self.assertNotIn("extra_body", kwargs)

    def test_frozen_local_rejects_non_loopback_endpoint(self):
        config = dict(self.local_config, base_url="http://192.168.1.20:8080/v1")
        with (
            patch("load_model.sys.frozen", True, create=True),
            patch("load_model.get_packaged_model_config", return_value=config),
            patch("load_model.init_chat_model") as init_chat_model,
            self.assertRaisesRegex(ModelConfigurationError, "回环地址"),
        ):
            init_llm()

        init_chat_model.assert_not_called()

    def test_frozen_local_requires_model_and_url(self):
        for missing in ("model", "base_url"):
            with self.subTest(missing=missing):
                config = dict(self.local_config)
                config[missing] = ""
                with (
                    patch("load_model.sys.frozen", True, create=True),
                    patch(
                        "load_model.get_packaged_model_config", return_value=config
                    ),
                    self.assertRaisesRegex(ModelConfigurationError, "本地模型尚未配置"),
                ):
                    init_llm()

    def test_frozen_rejects_unknown_provider(self):
        config = dict(self.local_config, provider="ollama")
        with (
            patch("load_model.sys.frozen", True, create=True),
            patch("load_model.get_packaged_model_config", return_value=config),
            self.assertRaisesRegex(ModelConfigurationError, "provider 无效"),
        ):
            init_llm()

    def test_frozen_local_accepts_localhost_and_ipv6_loopback(self):
        for base_url in (
            "http://localhost:8080/v1",
            "http://127.0.0.1:11434/v1",
            "http://[::1]:8080/v1",
        ):
            with self.subTest(base_url=base_url):
                config = dict(self.local_config, base_url=base_url)
                with (
                    patch("load_model.sys.frozen", True, create=True),
                    patch(
                        "load_model.get_packaged_model_config", return_value=config
                    ),
                    patch("load_model.init_chat_model") as init_chat_model,
                ):
                    init_llm()

                self.assertEqual(
                    init_chat_model.call_args.kwargs["base_url"], base_url
                )


class ModelErrorDescriptionTests(unittest.TestCase):
    def test_connection_errors_get_actionable_message(self):
        class APIConnectionError(Exception):
            pass

        message = describe_model_error(APIConnectionError("Connection error."), provider="local")

        self.assertIsNotNone(message)
        self.assertIn("本地模型服务", message)

    def test_real_langchain_connection_error_is_recognised(self):
        from langchain_core.exceptions import ModelConnectionError

        class ConcreteConnectionError(ModelConnectionError):
            pass

        message = describe_model_error(ConcreteConnectionError("boom"), provider="local")

        self.assertIn("本地模型服务", message)

    def test_remote_provider_gets_network_message(self):
        message = describe_model_error(ConnectionError("refused"), provider="remote")

        self.assertIn("网络", message)
        self.assertNotIn("本地模型服务", message)

    def test_os_errors_get_generic_message(self):
        message = describe_model_error(ConnectionRefusedError("refused"), provider="local")

        self.assertIn("本地模型服务", message)

    def test_unrelated_errors_are_not_translated(self):
        self.assertIsNone(describe_model_error(KeyError("missing")))

    def test_configuration_errors_keep_their_message(self):
        message = describe_model_error(ModelConfigurationError("配置有问题"))

        self.assertEqual(message, "配置有问题")

    def test_provider_is_inferred_when_not_passed(self):
        with (
            patch("load_model.sys.frozen", True, create=True),
            patch(
                "load_model.get_packaged_model_config",
                return_value={"provider": "local", "base_url": "http://127.0.0.1:8080/v1"},
            ),
        ):
            message = describe_model_error(ConnectionError("refused"))

        self.assertIn("本地模型服务", message)


class ReleaseConfigWriterTests(unittest.TestCase):
    def test_release_config_writer_requires_credentials(self):
        with self.assertRaisesRegex(SystemExit, "Configure DEEPSEEK_API_KEY"):
            MODEL_CONFIG_WRITER["build_model_config"]({})

    def test_release_config_writer_builds_local_config_without_a_key(self):
        config = MODEL_CONFIG_WRITER["build_model_config"](
            {
                "LLM_PROVIDER": "local",
                "LOCAL_BASE_URL": "http://127.0.0.1:8080/v1/",
                "LOCAL_MODEL": "qwen2.5-1.5b-instruct",
            }
        )

        self.assertEqual(
            config,
            {
                "provider": "local",
                "api_key": "local",
                "base_url": "http://127.0.0.1:8080/v1",
                "model": "qwen2.5-1.5b-instruct",
                "auto_start": False,
            },
        )

    def test_release_config_writer_local_defaults(self):
        config = MODEL_CONFIG_WRITER["build_model_config"](
            {"EXCEL_ASSISTANT_LLM_PROVIDER": "local"}
        )

        self.assertEqual(config["provider"], "local")
        self.assertEqual(config["base_url"], "http://127.0.0.1:8080/v1")
        self.assertEqual(config["model"], "qwen2.5-1.5b-instruct")
        self.assertTrue(config["api_key"])

    def test_release_config_writer_local_rejects_non_loopback(self):
        with self.assertRaisesRegex(SystemExit, "loopback"):
            MODEL_CONFIG_WRITER["build_model_config"](
                {
                    "LLM_PROVIDER": "local",
                    "LOCAL_BASE_URL": "http://192.168.1.20:8080/v1",
                }
            )

    def test_release_config_writer_rejects_unknown_provider(self):
        with self.assertRaisesRegex(SystemExit, "must be one of"):
            MODEL_CONFIG_WRITER["build_model_config"]({"LLM_PROVIDER": "ollama"})

    def test_release_config_writer_local_config_loads_in_frozen_mode(self):
        """构建产物必须能被打包程序直接消费，无需手工改字段。"""
        config = MODEL_CONFIG_WRITER["build_model_config"](
            {
                "LLM_PROVIDER": "local",
                "LOCAL_BASE_URL": "http://127.0.0.1:8080/v1",
                "LOCAL_MODEL": "qwen2.5-1.5b-instruct",
            }
        )
        with (
            patch("load_model.sys.frozen", True, create=True),
            patch("load_model.get_packaged_model_config", return_value=config),
            patch("load_model.init_chat_model") as init_chat_model,
        ):
            init_llm()

        kwargs = init_chat_model.call_args.kwargs
        self.assertEqual(kwargs["model"], "qwen2.5-1.5b-instruct")
        self.assertEqual(kwargs["base_url"], "http://127.0.0.1:8080/v1")

    def test_release_config_writer_supports_first_run_download(self):
        config = MODEL_CONFIG_WRITER["build_model_config"](
            {
                "LLM_PROVIDER": "local",
                "LOCAL_BASE_URL": "http://127.0.0.1:8080/v1",
                "LOCAL_AUTO_START": "1",
                "LOCAL_SERVER_PATH": r"llama\llama-server.exe",
                "LOCAL_MODEL_URL": "https://models.example.test/qwen.gguf",
                "LOCAL_MODEL_SHA256": "ABC123",
                "LOCAL_MODEL_SIZE": "1048576000",
                "LOCAL_MODEL_FILENAME": "qwen.gguf",
            }
        )

        self.assertEqual(config["model_url"], "https://models.example.test/qwen.gguf")
        self.assertEqual(config["model_sha256"], "abc123")
        self.assertEqual(config["model_size"], 1048576000)
        self.assertEqual(config["model_filename"], "qwen.gguf")
        self.assertNotIn("local_model_path", config)

    def test_release_config_writer_rejects_insecure_model_url(self):
        with self.assertRaisesRegex(SystemExit, "must use HTTPS"):
            MODEL_CONFIG_WRITER["build_model_config"](
                {
                    "LLM_PROVIDER": "local",
                    "LOCAL_AUTO_START": "1",
                    "LOCAL_SERVER_PATH": "llama-server.exe",
                    "LOCAL_MODEL_URL": "http://models.example.test/qwen.gguf",
                }
            )

    def test_release_config_writer_allows_loopback_http_model_url(self):
        config = MODEL_CONFIG_WRITER["build_model_config"](
            {
                "LLM_PROVIDER": "local",
                "LOCAL_AUTO_START": "1",
                "LOCAL_SERVER_PATH": "llama-server.exe",
                "LOCAL_MODEL_URL": "http://127.0.0.1:8095/qwen.gguf",
            }
        )

        self.assertEqual(config["model_url"], "http://127.0.0.1:8095/qwen.gguf")

    def test_auto_start_requires_a_model_source(self):
        with self.assertRaisesRegex(SystemExit, "LOCAL_AUTO_START requires"):
            MODEL_CONFIG_WRITER["build_model_config"](
                {
                    "LLM_PROVIDER": "local",
                    "LOCAL_AUTO_START": "1",
                    "LOCAL_SERVER_PATH": "llama-server.exe",
                }
            )

    def test_download_config_is_readable_by_the_runtime(self):
        """下载清单必须能被运行时正确解析，字段名不能对不上。"""
        from model_download import manifest_from_config

        config = MODEL_CONFIG_WRITER["build_model_config"](
            {
                "LLM_PROVIDER": "local",
                "LOCAL_BASE_URL": "http://127.0.0.1:8080/v1",
                "LOCAL_AUTO_START": "1",
                "LOCAL_SERVER_PATH": r"llama\llama-server.exe",
                "LOCAL_MODEL_URL": "https://models.example.test/qwen.gguf",
                "LOCAL_MODEL_SIZE": "1048576000",
            }
        )

        manifest = manifest_from_config(config)
        self.assertTrue(manifest.is_configured)
        self.assertEqual(manifest.size, 1048576000)
        self.assertEqual(manifest.effective_filename, "qwen.gguf")

    def test_release_config_writer_rejects_invalid_size(self):
        with self.assertRaises(SystemExit):
            MODEL_CONFIG_WRITER["build_model_config"](
                {
                    "LLM_PROVIDER": "local",
                    "LOCAL_MODEL_URL": "https://models.example.test/qwen.gguf",
                    "LOCAL_MODEL_SIZE": "huge",
                }
            )

    def test_release_config_writer_embeds_auto_start_settings(self):
        config = MODEL_CONFIG_WRITER["build_model_config"](
            {
                "LLM_PROVIDER": "local",
                "LOCAL_BASE_URL": "http://127.0.0.1:8080/v1",
                "LOCAL_MODEL": "qwen2.5-1.5b-instruct",
                "LOCAL_AUTO_START": "true",
                "LOCAL_SERVER_PATH": r"llama\llama-server.exe",
                "LOCAL_MODEL_PATH": r"models\model.gguf",
                "LOCAL_CONTEXT_SIZE": "4096",
                "LOCAL_STARTUP_TIMEOUT": "120",
                "LOCAL_EXTRA_ARGS": "--threads,4",
            }
        )

        self.assertTrue(config["auto_start"])
        self.assertEqual(config["local_server"], r"llama\llama-server.exe")
        self.assertEqual(config["local_model_path"], r"models\model.gguf")
        self.assertEqual(config["context_size"], 4096)
        self.assertEqual(config["startup_timeout"], 120.0)
        self.assertEqual(config["extra_args"], ["--threads", "4"])

    def test_release_config_writer_requires_paths_when_auto_start_is_enabled(self):
        for missing in ("LOCAL_SERVER_PATH", "LOCAL_MODEL_PATH"):
            with self.subTest(missing=missing):
                environment = {
                    "LLM_PROVIDER": "local",
                    "LOCAL_AUTO_START": "1",
                    "LOCAL_SERVER_PATH": "llama-server.exe",
                    "LOCAL_MODEL_PATH": "model.gguf",
                }
                environment.pop(missing)
                with self.assertRaisesRegex(SystemExit, "LOCAL_AUTO_START requires"):
                    MODEL_CONFIG_WRITER["build_model_config"](environment)

    def test_release_config_writer_rejects_invalid_numbers(self):
        for key, value in (
            ("LOCAL_CONTEXT_SIZE", "many"),
            ("LOCAL_CONTEXT_SIZE", "0"),
            ("LOCAL_STARTUP_TIMEOUT", "abc"),
            ("LOCAL_STARTUP_TIMEOUT", "-1"),
        ):
            with self.subTest(key=key, value=value):
                with self.assertRaises(SystemExit):
                    MODEL_CONFIG_WRITER["build_model_config"](
                        {
                            "LLM_PROVIDER": "local",
                            key: value,
                        }
                    )

    def test_written_config_can_be_consumed_by_the_server_manager(self):
        """构建产物必须能被本地服务管理器直接解析，无需手工补字段。"""
        import local_model_server

        config = MODEL_CONFIG_WRITER["build_model_config"](
            {
                "LLM_PROVIDER": "local",
                "LOCAL_BASE_URL": "http://127.0.0.1:8080/v1",
                "LOCAL_MODEL": "qwen2.5-1.5b-instruct",
                "LOCAL_AUTO_START": "1",
                "LOCAL_SERVER_PATH": r"llama\llama-server.exe",
                "LOCAL_MODEL_PATH": r"models\model.gguf",
                "LOCAL_CONTEXT_SIZE": "4096",
            }
        )

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            parsed = local_model_server.parse_local_server_config(config, base_dir=base)

            self.assertTrue(parsed.auto_start)
            self.assertEqual(parsed.base_url, "http://127.0.0.1:8080/v1")
            self.assertEqual(parsed.context_size, 4096)
            self.assertTrue(parsed.executable.endswith("llama-server.exe"))
            self.assertTrue(parsed.model_path.endswith("model.gguf"))
            self.assertEqual(Path(parsed.executable).parent, base / "llama")

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
                "provider": "remote",
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
                    "provider": "remote",
                    "api_key": "test-build-key",
                    "base_url": "https://api.example.test/v1",
                    "model": "test-model",
                },
            )


if __name__ == "__main__":
    unittest.main()

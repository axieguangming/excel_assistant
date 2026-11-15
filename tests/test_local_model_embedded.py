"""进程内 transformers 后端的配置解析与状态机测试。

背景：桌面版原来用 llama-server 子进程，实测 20 条样本 65%；同一份权重在进程内
用 transformers 加载是 85%，且差异单向（llama.cpp 独赢 0 条）。这些测试锁住新
后端的关键行为，避免以后改配置格式时悄悄退化。
"""

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from local_model_embedded import (  # noqa: E402
    EmbeddedModelConfig,
    EmbeddedModelRuntime,
    local_backend,
    model_dir_looks_valid,
    parse_embedded_config,
)


class ConfigParsingTests(unittest.TestCase):
    def test_backend_defaults_to_llama_server(self):
        """旧配置（没有 backend 字段）必须继续走 llama_server。"""
        self.assertEqual(local_backend({}), "llama_server")

    def test_backend_reads_explicit_value(self):
        self.assertEqual(
            local_backend({"backend": "Transformers"}), "transformers"
        )

    def test_relative_model_dir_resolves_against_base(self):
        with tempfile.TemporaryDirectory() as base:
            config = parse_embedded_config(
                {"model_dir": "models/embedded"}, base_dir=Path(base)
            )
            self.assertEqual(
                Path(config.model_dir), Path(base).resolve() / "models" / "embedded"
            )

    def test_device_must_be_known(self):
        """未知设备名不应静默降级，否则用户以为在用 GPU。"""
        config = parse_embedded_config({"model_dir": "x", "device": "tpu"})
        # 解析层保留原值，由 _resolve_device 拒绝；这里确认没有被悄悄改成 auto
        self.assertEqual(config.device, "tpu")

    def test_max_new_tokens_rejects_non_positive(self):
        config = parse_embedded_config({"model_dir": "x", "max_new_tokens": "0"})
        self.assertEqual(config.max_new_tokens, 256)

    def test_defaults_are_gpu_friendly(self):
        config = parse_embedded_config({"model_dir": "x"})
        self.assertEqual(config.device, "auto")
        self.assertEqual(config.dtype, "bfloat16")


class ModelDirValidationTests(unittest.TestCase):
    def test_requires_config_json(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(model_dir_looks_valid(directory))
            (Path(directory) / "config.json").write_text("{}", encoding="utf-8")
            self.assertTrue(model_dir_looks_valid(directory))

    def test_missing_directory_is_invalid(self):
        self.assertFalse(model_dir_looks_valid("Z:/definitely/not/here"))

    def test_empty_path_is_invalid(self):
        self.assertFalse(model_dir_looks_valid(""))


class RuntimeStateTests(unittest.TestCase):
    def _config(self, directory: str) -> EmbeddedModelConfig:
        return EmbeddedModelConfig(model_dir=directory, model="test")

    def test_missing_model_dir_reports_error(self):
        runtime = EmbeddedModelRuntime(self._config("Z:/nope"))
        ok, warning = runtime.ensure_ready()
        self.assertFalse(ok)
        self.assertIn("找不到本地模型目录", warning or "")
        self.assertEqual(runtime.status()["state"], "error")

    def test_valid_dir_before_ensure_ready_is_starting(self):
        """文件齐全但还没开始加载时不能报成红色错误。"""
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "config.json").write_text("{}", encoding="utf-8")
            runtime = EmbeddedModelRuntime(self._config(directory))
            self.assertEqual(runtime.status()["state"], "starting")

    def test_loader_failure_surfaces_as_error(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "config.json").write_text("{}", encoding="utf-8")

            def boom(_config):
                raise RuntimeError("CUDA out of memory")

            runtime = EmbeddedModelRuntime(self._config(directory), loader=boom)
            ok, _ = runtime.ensure_ready()
            self.assertFalse(ok)  # 后台加载，先返回 False
            # 加载失败时 wait_until_ready 返回 False（表示「没就绪」），
            # 具体原因从 status() 读。
            self.assertFalse(runtime.wait_until_ready(timeout=5))
            status = runtime.status()
            self.assertEqual(status["state"], "error")
            self.assertIn("CUDA out of memory", status["message"])

    def test_successful_load_reaches_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "config.json").write_text("{}", encoding="utf-8")

            class FakeModel:
                device = "cuda:0"

            runtime = EmbeddedModelRuntime(
                self._config(directory), loader=lambda _config: FakeModel()
            )
            runtime.ensure_ready()
            self.assertTrue(runtime.wait_until_ready(timeout=5))
            status = runtime.status()
            self.assertEqual(status["state"], "ready")
            self.assertEqual(status["device"], "cuda:0")
            self.assertEqual(status["backend"], "transformers")
            self.assertEqual(status["percent"], 100.0)

    def test_download_is_not_supported(self):
        """模型随包分发，下载端点应安全地返回「不能下载」。"""
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "config.json").write_text("{}", encoding="utf-8")
            runtime = EmbeddedModelRuntime(self._config(directory))
            self.assertFalse(runtime.manifest.is_configured)
            self.assertFalse(runtime.start_download())
            self.assertFalse(runtime.status()["can_download"])


class BuildRuntimeDispatchTests(unittest.TestCase):
    def test_transformers_backend_returns_embedded_runtime(self):
        from local_model_runtime import build_runtime

        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "config.json").write_text("{}", encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "LLM_PROVIDER": "local",
                    "LOCAL_BACKEND": "transformers",
                    "LOCAL_MODEL_DIR": directory,
                    "LOCAL_MODEL": "test",
                },
                clear=False,
            ):
                runtime = build_runtime()
        self.assertIsInstance(runtime, EmbeddedModelRuntime)

    def test_remote_provider_returns_none(self):
        from local_model_runtime import build_runtime

        with patch.dict(os.environ, {"LLM_PROVIDER": "remote"}, clear=False):
            self.assertIsNone(build_runtime())


class InitLlmDispatchTests(unittest.TestCase):
    def test_local_transformers_backend_returns_chat_model(self):
        """init_llm 在本地 + transformers 时应返回进程内模型，而不是 OpenAI 客户端。"""
        from finetune.local_llm import LocalChatModel
        from load_model import init_llm

        fake = LocalChatModel(model=object(), tokenizer=object())
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "config.json").write_text("{}", encoding="utf-8")
            with (
                patch.dict(
                    os.environ,
                    {
                        "LLM_PROVIDER": "local",
                        "LOCAL_BACKEND": "transformers",
                        "LOCAL_MODEL_DIR": directory,
                    },
                    clear=False,
                ),
                patch(
                    "local_model_embedded.build_chat_model", return_value=fake
                ) as builder,
            ):
                result = init_llm()
        self.assertIs(result, fake)
        self.assertTrue(builder.called)


if __name__ == "__main__":
    unittest.main()

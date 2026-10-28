"""校验训练脚本的模型来源检查。

重点是：GGUF 必须被明确拒绝，而不是让用户等一次隐式下载。
"""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_train_module():
    spec = importlib.util.spec_from_file_location(
        "train_lora_under_test", ROOT / "finetune" / "train_lora.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ResolveModelSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_train_module()

    def test_gguf_is_rejected_with_explanation(self):
        gguf = ROOT / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
        if not gguf.is_file():
            self.skipTest("本机没有 GGUF 文件")

        with self.assertRaises(SystemExit) as captured:
            self.module.resolve_model_source(str(gguf))

        message = str(captured.exception)
        self.assertIn("GGUF", message)
        self.assertIn("safetensors", message)
        self.assertIn("不能用于微调", message)

    def test_gguf_extension_is_rejected_even_if_missing(self):
        with self.assertRaises(SystemExit) as captured:
            self.module.resolve_model_source("models/not-there.gguf")

        self.assertIn("GGUF", str(captured.exception))

    def test_repo_id_warns_about_download(self):
        source, warning = self.module.resolve_model_source("Qwen/Qwen2.5-1.5B-Instruct")

        self.assertEqual(source, "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertIsNotNone(warning)
        self.assertIn("下载", warning)

    def test_local_directory_without_weights_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text("{}", encoding="utf-8")

            with self.assertRaises(SystemExit) as captured:
                self.module.resolve_model_source(directory)

        self.assertIn("safetensors", str(captured.exception))

    def test_local_directory_with_safetensors_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "model-00001-of-00002.safetensors").write_bytes(b"x")

            source, warning = self.module.resolve_model_source(directory)

        self.assertEqual(Path(source), Path(directory))
        self.assertIsNone(warning)

    def test_local_directory_with_bin_weights_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "pytorch_model.bin").write_bytes(b"x")

            source, warning = self.module.resolve_model_source(directory)

        self.assertEqual(Path(source), Path(directory))
        self.assertIsNone(warning)

    def test_plain_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "config.json")
            path.write_text("{}", encoding="utf-8")

            with self.assertRaises(SystemExit) as captured:
                self.module.resolve_model_source(str(path))

        self.assertIn("需要是模型目录", str(captured.exception))


class LazyImportTests(unittest.TestCase):
    """没有训练依赖时，--help 与参数校验仍应可用。"""

    @classmethod
    def setUpClass(cls):
        cls.module = load_train_module()

    def test_help_works_without_importing_heavy_modules(self):
        import subprocess

        # Windows 控制台默认 GBK，用 bytes 捕获后容错解码，避免编码错误掩盖真实失败。
        result = subprocess.run(
            [sys.executable, str(ROOT / "finetune" / "train_lora.py"), "--help"],
            capture_output=True,
            timeout=180,
        )
        stdout = result.stdout.decode("utf-8", errors="replace")
        stderr = result.stderr.decode("utf-8", errors="replace")

        self.assertEqual(result.returncode, 0, stderr)
        self.assertIn("--model", stdout)
        self.assertIn("gguf", stdout.lower())

    def test_collator_pads_labels_with_ignore_index(self):
        collator = self.module.TrajectoryCollator(pad_token_id=0)
        batch = collator(
            [
                {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1, 2, 3]},
                {"input_ids": [4], "attention_mask": [1], "labels": [4]},
            ]
        )

        labels = batch["labels"].tolist()
        self.assertEqual(labels[0], [1, 2, 3])
        # 短样本补到 3 位，padding 位置必须是 -100，否则会被当成 token 0 计入损失
        self.assertEqual(labels[1], [4, -100, -100])
        self.assertEqual(batch["attention_mask"].tolist()[1], [1, 0, 0])


if __name__ == "__main__":
    unittest.main()

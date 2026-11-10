"""DPO 训练脚本的纯逻辑单元测试（不加载模型）。

重点覆盖两处曾导致实际故障的地方：
1. 补全掩码必须与标签位置对齐，否则 logprob 会错位
2. 适配器必须保存到目录根位，`load_model` 才能加载
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_module():
    spec = importlib.util.spec_from_file_location(
        "train_dpo_under_test", ROOT / "finetune" / "train_dpo.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTokenizer:
    """每个消息固定 4 个 token，便于精确断言。"""

    def apply_chat_template(self, messages, **kwargs):
        return {"input_ids": list(range(len(messages) * 4))}


class PairEncodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def _pair(self, prompt_messages: int = 2) -> dict:
        prompt = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "a", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        ][:prompt_messages]
        return {
            "pair_id": "p1",
            "kind": "success_repeat",
            "tools": [],
            "prompt": prompt,
            "chosen": {"role": "assistant", "content": "", "tool_calls": [{"id": "x", "function": {"name": "b", "arguments": "{}"}}]},
            "rejected": {"role": "assistant", "content": "", "tool_calls": [{"id": "y", "function": {"name": "a", "arguments": "{}"}}]},
        }

    def test_completion_is_suffix_after_prompt(self):
        dataset = self.module.PairDataset([self._pair(4)], FakeTokenizer(), max_length=4096)
        self.assertEqual(len(dataset.items), 1)
        item = dataset.items[0]
        self.assertEqual(len(item["prompt_ids"]), 16)  # 4 条消息 × 4
        # 5 条消息 = 20 token，补全 4 token
        self.assertEqual(len(item["chosen_ids"]), 4)
        self.assertEqual(len(item["rejected_ids"]), 4)

    def test_overlong_pair_is_skipped(self):
        dataset = self.module.PairDataset([self._pair(4)], FakeTokenizer(), max_length=18)
        self.assertEqual(len(dataset.items), 0)
        self.assertEqual(dataset.skipped, 1)

    def test_empty_completion_is_skipped(self):
        pair = self._pair(4)
        tokenizer = FakeTokenizer()

        # 构造一个 prompt 与 prompt+chosen 等长的情形（模板没有新增 token）
        class NoopTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return {"input_ids": list(range(16))}

        dataset = self.module.PairDataset([pair], NoopTokenizer(), max_length=4096)
        self.assertEqual(len(dataset.items), 0)


class SavePolicyAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_nested_adapter_files_are_promoted(self):
        """peft 写到 <out>/policy/ 下的文件必须被移动到 <out>/ 根位。"""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "adapter"
            out.mkdir()
            nested = out / "policy"
            nested.mkdir()
            (nested / "adapter_config.json").write_text("{}", encoding="utf-8")
            (nested / "adapter_model.safetensors").write_bytes(b"x")

            class FakeModel:
                def set_adapter(self, name):
                    self.name = name

                def save_pretrained(self, directory, selected_adapters=None):
                    # 模拟 peft 的多适配器行为：写到子目录
                    target = Path(directory) / "policy"
                    target.mkdir(parents=True, exist_ok=True)
                    (target / "adapter_config.json").write_text("{}", encoding="utf-8")

            self.module.save_policy_adapter(FakeModel(), out)
            self.assertTrue((out / "adapter_config.json").is_file())
            self.assertTrue((out / "adapter_model.safetensors").is_file())
            self.assertFalse((out / "policy").exists())


class CompletionMaskMathTests(unittest.TestCase):
    """不需要模型：直接验证「位置 i 的 logits 预测 i+1」这一移位关系。"""

    def test_shifted_mask_matches_targets(self):
        import torch

        input_ids = torch.tensor([[10, 11, 12, 13, 14]])
        completion = torch.tensor([[False, False, False, True, True]])
        keep = torch.zeros_like(completion)
        keep[:, :-1] = completion[:, 1:]
        # 补全 token 是 13,14（位置 3、4），由位置 2、3 预测
        self.assertEqual(keep.tolist(), [[False, False, True, True, False]])
        targets = input_ids[:, 1:][completion[:, 1:]]
        self.assertEqual(targets.tolist(), [13, 14])


if __name__ == "__main__":
    unittest.main()

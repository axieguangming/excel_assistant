"""打包应用启动 llama-server 时，采样参数必须对齐贪心解码。

背景：llama-server 的默认值是 temperature 0.7 / top_k 20 / top_p 0.8 /
min_p 0.05 / repeat_penalty 1.1。即使客户端在请求里传 temperature=0，
其余采样器仍按默认值生效。其中 repeat_penalty 1.1 危害最大——工具调用会
反复输出 "name"、"arguments"、工具名这类相同短语，penalty 压低这些 token
的概率，模型就会改口、跳步，或陷入「同一工具调用 9 次直到轮数上限」的循环。

实测（20 条样本、bf16 GGUF）：默认采样 12–14/20 且同配置重复测量会漂移；
中和后稳定 13–14/20 且逐条 20/20 可复现。
"""

import unittest

from local_model_server import LocalModelServer, parse_local_server_config

BASE = {
    "base_url": "http://127.0.0.1:8080/v1",
    "model": "test-model",
    "local_server": "llama-server.exe",
    "local_model_path": "model.gguf",
    "context_size": 8192,
}


class BuildCommandSamplingTests(unittest.TestCase):
    def _command(self, **overrides):
        raw = dict(BASE)
        raw.update(overrides)
        config = parse_local_server_config(raw)
        server = LocalModelServer(config, port_check=lambda host, port: False)
        return server._build_command()

    def _value_of(self, command, flag):
        index = command.index(flag)
        return command[index + 1]

    def test_temperature_is_zero(self):
        self.assertEqual(self._value_of(self._command(), "--temp"), "0")

    def test_top_k_is_one(self):
        self.assertEqual(self._value_of(self._command(), "--top-k"), "1")

    def test_top_p_disabled(self):
        self.assertEqual(self._value_of(self._command(), "--top-p"), "1.0")

    def test_min_p_disabled(self):
        self.assertEqual(self._value_of(self._command(), "--min-p"), "0.0")

    def test_repeat_penalty_neutral(self):
        """这是循环失败的元凶，必须是 1.0（等于不惩罚）。"""
        self.assertEqual(self._value_of(self._command(), "--repeat-penalty"), "1.0")

    def test_single_slot(self):
        """多槽位的 KV 前缀复用会把不同请求拼在一起。"""
        self.assertEqual(self._value_of(self._command(), "-np"), "1")

    def test_jinja_still_enabled(self):
        """工具调用依赖模型自带的聊天模板，不能丢。"""
        self.assertIn("--jinja", self._command())

    def test_user_extra_args_win(self):
        """extra_args 放在最后，用户仍然可以覆盖任何默认值。"""
        command = self._command(extra_args=["-t", "8"])
        self.assertEqual(command[-2:], ["-t", "8"])
        # 默认值仍然在前，只是被后面的同名参数覆盖
        self.assertIn("--temp", command)

    def test_executable_and_model_first(self):
        command = self._command()
        self.assertTrue(command[0].endswith("llama-server.exe"))
        self.assertEqual(command[1], "-m")
        self.assertTrue(command[2].endswith("model.gguf"))


if __name__ == "__main__":
    unittest.main()

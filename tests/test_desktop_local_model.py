import json
import os
import socket
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

FAKE_SERVER = textwrap.dedent(
    """
    import sys
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    args = sys.argv[1:]
    host = args[args.index("--host") + 1]
    port = int(args[args.index("--port") + 1])

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *extra):
            pass

        def do_GET(self):
            body = b'{"data":[]}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    ThreadingHTTPServer((host, port), Handler).serve_forever()
    """
)

# 在子进程里替换 _build_command，用真实的 Python 假服务代替 llama-server，
# 这样既能验证 desktop 的接线，又不需要真的模型文件。
DRIVER = textwrap.dedent(
    """
    import json, os, sys
    from pathlib import Path

    import local_model_server

    def fake_build_command(self):
        return [
            sys.executable, "-u", os.environ["FAKE_SERVER_SCRIPT"],
            "--host", self.config.host, "--port", str(self.config.port),
        ]

    local_model_server.LocalModelServer._build_command = fake_build_command

    import desktop

    runtime, warning = desktop._start_local_model(Path(os.environ["TEST_DATA_DIR"]))
    payload = {
        "has_runtime": runtime is not None,
        "warning": warning,
    }
    if runtime is not None:
        payload.update(
            {
                "state": runtime.status()["state"],
                "healthy": runtime.server.is_healthy(),
                "owns_process": runtime.server.owns_process,
                "has_child": runtime.server._process is not None,
            }
        )
        runtime.stop()
        payload["stopped_healthy"] = runtime.server.is_healthy()
    print("RESULT:" + json.dumps(payload, ensure_ascii=False))
    """
)


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class DesktopLocalModelWiringTests(unittest.TestCase):
    """验证 desktop 启动器与本地模型管理器的真实接线。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.script = self.root / "fake_server.py"
        self.script.write_text(FAKE_SERVER, encoding="utf-8")
        (self.root / "llama-server.exe").write_bytes(b"")
        (self.root / "model.gguf").write_bytes(b"gguf")

    def tearDown(self):
        self._tmp.cleanup()

    def _run_driver(self, env_overrides: dict[str, str]) -> dict:
        env = os.environ.copy()
        env.update(
            {
                "PYTHONPATH": str(REPO_ROOT),
                "PYTHONIOENCODING": "utf-8",
                "FAKE_SERVER_SCRIPT": str(self.script),
                "TEST_DATA_DIR": str(self.root / "data"),
            }
        )
        env.update(env_overrides)

        completed = subprocess.run(
            [sys.executable, "-c", DRIVER],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=120,
            check=False,
        )

        self.assertEqual(
            completed.returncode,
            0,
            f"driver failed:\nstdout={completed.stdout}\nstderr={completed.stderr}",
        )
        for line in completed.stdout.splitlines():
            if line.startswith("RESULT:"):
                return json.loads(line[len("RESULT:") :])
        self.fail(f"driver produced no result:\n{completed.stdout}\n{completed.stderr}")

    def test_desktop_starts_and_stops_the_managed_server(self):
        port = free_port()
        result = self._run_driver(
            {
                "LLM_PROVIDER": "local",
                "LOCAL_BASE_URL": f"http://127.0.0.1:{port}/v1",
                "LOCAL_MODEL": "qwen2.5-1.5b-instruct",
                "LOCAL_AUTO_START": "1",
                "LOCAL_SERVER_PATH": str(self.root / "llama-server.exe"),
                "LOCAL_MODEL_PATH": str(self.root / "model.gguf"),
                "LOCAL_STARTUP_TIMEOUT": "30",
            }
        )

        self.assertIsNone(result["warning"], result["warning"])
        self.assertTrue(result["has_runtime"])
        self.assertTrue(result["has_child"])
        self.assertTrue(result["healthy"])
        self.assertTrue(result["owns_process"])
        self.assertEqual(result["state"], "ready")
        # stop() 之后服务应已下线，证明确实关闭了子进程。
        self.assertFalse(result["stopped_healthy"])
        self.assertTrue((self.root / "data" / "logs" / "local-model.log").is_file())

    def test_desktop_reports_manual_mode_without_auto_start(self):
        port = free_port()
        result = self._run_driver(
            {
                "LLM_PROVIDER": "local",
                "LOCAL_BASE_URL": f"http://127.0.0.1:{port}/v1",
                "LOCAL_MODEL": "qwen2.5-1.5b-instruct",
                "LOCAL_AUTO_START": "0",
            }
        )

        self.assertTrue(result["has_runtime"])
        self.assertFalse(result["healthy"])
        self.assertEqual(result["state"], "error")
        self.assertIn(f"127.0.0.1:{port}", result["warning"])
        self.assertIn("未启用自动启动", result["warning"])

    def test_desktop_reports_missing_model_file(self):
        port = free_port()
        result = self._run_driver(
            {
                "LLM_PROVIDER": "local",
                "LOCAL_BASE_URL": f"http://127.0.0.1:{port}/v1",
                "LOCAL_MODEL": "qwen2.5-1.5b-instruct",
                "LOCAL_AUTO_START": "1",
                "LOCAL_SERVER_PATH": str(self.root / "llama-server.exe"),
                "LOCAL_MODEL_PATH": str(self.root / "missing.gguf"),
            }
        )

        self.assertTrue(result["has_runtime"])
        self.assertFalse(result["healthy"])
        self.assertEqual(result["state"], "error")
        self.assertIn("找不到本地模型文件", result["warning"])

    def test_missing_model_with_download_url_does_not_warn(self):
        """首次启动且配置了下载地址时，界面负责引导下载，不应弹警告。"""
        port = free_port()
        result = self._run_driver(
            {
                "LLM_PROVIDER": "local",
                "LOCAL_BASE_URL": f"http://127.0.0.1:{port}/v1",
                "LOCAL_MODEL": "qwen2.5-1.5b-instruct",
                "LOCAL_AUTO_START": "1",
                "LOCAL_SERVER_PATH": str(self.root / "llama-server.exe"),
                "LOCAL_MODEL_URL": "https://models.test/model.gguf",
            }
        )

        self.assertTrue(result["has_runtime"])
        self.assertIsNone(result["warning"])
        self.assertEqual(result["state"], "needs_download")

    def test_desktop_does_nothing_in_remote_mode(self):
        result = self._run_driver({"LLM_PROVIDER": "remote"})

        self.assertFalse(result["has_runtime"])
        self.assertIsNone(result["warning"])


if __name__ == "__main__":
    unittest.main()

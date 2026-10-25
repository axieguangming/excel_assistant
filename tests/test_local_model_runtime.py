import hashlib
import io
import tempfile
import threading
import time
import unittest
from pathlib import Path

from local_model_runtime import (
    STATE_ERROR,
    STATE_NEEDS_DOWNLOAD,
    STATE_READY,
    LocalModelRuntime,
    resolve_model_path,
    set_runtime,
    get_runtime,
    build_runtime,
)
from local_model_server import LocalModelServer, LocalModelStartupError, LocalServerConfig
from model_download import ModelDownloader, ModelManifest

PAYLOAD = b"GGUF" + b"\x00" * 2048


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FakeResponse:
    def __init__(self, data: bytes, status: int = 200):
        self._buffer = io.BytesIO(data)
        self.status = status
        self.headers = {"Content-Length": str(len(data))}

    def read(self, size: int = -1) -> bytes:
        return self._buffer.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class StubServer(LocalModelServer):
    """可控的假服务：不真的拉起进程。"""

    def __init__(self, config, *, healthy=False, fail_with=None):
        super().__init__(config, log_dir=Path("."), port_check=lambda h, p: False)
        self._healthy = healthy
        self._fail_with = fail_with
        self.started = 0
        self.stopped = 0

    def is_healthy(self) -> bool:
        return self._healthy

    def ensure_running(self) -> bool:
        if self._fail_with:
            raise LocalModelStartupError(self._fail_with)
        self.started += 1
        self._healthy = True
        return True

    def stop(self) -> None:
        self.stopped += 1
        self._healthy = False


def base_config(**overrides) -> LocalServerConfig:
    values = {
        "base_url": "http://127.0.0.1:8080/v1",
        "model": "qwen2.5-1.5b-instruct",
        "auto_start": True,
        "executable": "llama-server.exe",
    }
    values.update(overrides)
    return LocalServerConfig(**values)


class ResolveModelPathTests(unittest.TestCase):
    def test_explicit_path_wins(self):
        config = base_config(model_path=r"D:\models\m.gguf")
        manifest = ModelManifest(url="https://x.test/other.gguf")

        path = resolve_model_path(config, manifest, Path(r"C:\data"))

        self.assertEqual(path, Path(r"D:\models\m.gguf"))

    def test_falls_back_to_data_dir_models(self):
        config = base_config(model_path="")
        manifest = ModelManifest(url="https://x.test/qwen.gguf")

        path = resolve_model_path(config, manifest, Path(r"C:\data"))

        self.assertEqual(path, Path(r"C:\data") / "models" / "qwen.gguf")


class RuntimeStatusTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.manifest = ModelManifest(
            url="https://x.test/model.gguf",
            sha256=sha256_of(PAYLOAD),
            size=len(PAYLOAD),
        )

    def tearDown(self):
        self._tmp.cleanup()

    def _runtime(self, *, server=None, installed=False, manifest=None):
        manifest = manifest or self.manifest
        model_path = self.root / "models" / "model.gguf"
        if installed:
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model_path.write_bytes(PAYLOAD)
        config = base_config(model_path=str(model_path))
        downloader = ModelDownloader(
            manifest,
            model_path,
            open_url=lambda request, timeout=None: FakeResponse(PAYLOAD),
        )
        return LocalModelRuntime(
            config,
            manifest,
            data_dir=self.root,
            downloader=downloader,
            server=server if server is not None else StubServer(config),
        )

    def test_missing_model_reports_needs_download(self):
        runtime = self._runtime()

        status = runtime.status()

        self.assertEqual(status["state"], STATE_NEEDS_DOWNLOAD)
        self.assertTrue(status["can_download"])
        self.assertIn("下载", status["message"])

    def test_missing_model_without_manifest_reports_error(self):
        runtime = self._runtime(manifest=ModelManifest(url=""))

        status = runtime.status()

        self.assertEqual(status["state"], STATE_ERROR)
        self.assertFalse(status["can_download"])

    def test_installed_and_healthy_reports_ready(self):
        server = StubServer(base_config(), healthy=True)
        runtime = self._runtime(server=server, installed=True)

        status = runtime.status()

        self.assertEqual(status["state"], STATE_READY)
        self.assertFalse(status["can_download"])

    def test_installed_but_unhealthy_reports_error_with_server_message(self):
        server = StubServer(base_config(), fail_with="端口被占用")
        runtime = self._runtime(server=server, installed=True)
        runtime.ensure_ready()

        status = runtime.status()

        self.assertEqual(status["state"], STATE_ERROR)
        self.assertIn("端口被占用", status["message"])

    def test_status_reports_model_size_and_filename(self):
        runtime = self._runtime()

        status = runtime.status()

        self.assertEqual(status["model_size"], len(PAYLOAD))
        self.assertEqual(status["model_filename"], "model.gguf")
        self.assertEqual(status["provider"], "local")

    def test_ensure_ready_skips_when_model_missing(self):
        server = StubServer(base_config())
        runtime = self._runtime(server=server)

        ready, warning = runtime.ensure_ready()

        self.assertFalse(ready)
        self.assertIsNone(warning)
        self.assertEqual(server.started, 0)

    def test_ensure_ready_starts_server_when_model_present(self):
        server = StubServer(base_config())
        runtime = self._runtime(server=server, installed=True)

        ready, warning = runtime.ensure_ready()

        self.assertTrue(ready)
        self.assertIsNone(warning)
        self.assertEqual(server.started, 1)

    def test_ensure_ready_returns_warning_on_failure(self):
        server = StubServer(base_config(), fail_with="模型文件损坏")
        runtime = self._runtime(server=server, installed=True)

        ready, warning = runtime.ensure_ready()

        self.assertFalse(ready)
        self.assertEqual(warning, "模型文件损坏")

    def test_ensure_ready_does_not_restart_healthy_server(self):
        server = StubServer(base_config(), healthy=True)
        runtime = self._runtime(server=server, installed=True)

        ready, warning = runtime.ensure_ready()

        self.assertTrue(ready)
        self.assertIsNone(warning)
        self.assertEqual(server.started, 0)

    def test_start_download_rejects_when_already_installed(self):
        runtime = self._runtime(installed=True)

        self.assertFalse(runtime.start_download())

    def test_start_download_without_manifest_fails_with_message(self):
        runtime = self._runtime(manifest=ModelManifest(url=""))

        self.assertFalse(runtime.start_download())
        self.assertEqual(runtime.status()["state"], STATE_ERROR)

    def test_start_download_then_auto_start_server(self):
        gate = threading.Event()

        def open_url(request, timeout=None):
            gate.wait(5)
            return FakeResponse(PAYLOAD)

        server = StubServer(base_config())
        runtime = self._runtime(server=server)
        runtime.downloader = ModelDownloader(
            self.manifest, runtime.model_path, open_url=open_url
        )

        self.assertTrue(runtime.start_download())
        # 下载进行中：状态应为 downloading 或已经在校验。
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if runtime.downloader.status.state in {"downloading", "verifying"}:
                break
            time.sleep(0.02)

        gate.set()

        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if runtime.status()["state"] == STATE_READY:
                break
            time.sleep(0.02)

        status = runtime.status()
        self.assertEqual(status["state"], STATE_READY)
        self.assertEqual(server.started, 1)

    def test_download_failure_surfaces_error_message(self):
        def open_url(request, timeout=None):
            raise OSError("no network")

        runtime = self._runtime()
        runtime.downloader = ModelDownloader(
            self.manifest, runtime.model_path, open_url=open_url
        )

        runtime.start_download()

        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if runtime.status()["state"] == STATE_ERROR:
                break
            time.sleep(0.02)

        self.assertEqual(runtime.status()["state"], STATE_ERROR)

    def test_unexpected_start_error_does_not_hang_in_starting(self):
        """启动过程中的意外异常必须变成 error，而不是永远显示"启动中"。"""

        class ExplodingServer(StubServer):
            def ensure_running(self) -> bool:
                raise OSError("unexpected")

        server = ExplodingServer(base_config())
        runtime = self._runtime(server=server, installed=True)

        ready, warning = runtime.ensure_ready()

        self.assertFalse(ready)
        self.assertIn("意外错误", warning or "")
        self.assertEqual(runtime.status()["state"], STATE_ERROR)

    def test_stop_delegates_to_server(self):
        server = StubServer(base_config(), healthy=True)
        runtime = self._runtime(server=server, installed=True)

        runtime.stop()

        self.assertEqual(server.stopped, 1)

    def test_server_receives_resolved_model_path(self):
        runtime = self._runtime()

        self.assertEqual(str(runtime.model_path), runtime.server.config.model_path)


class EndToEndDownloadTests(unittest.TestCase):
    """用真实 HTTP 服务与真实子进程验证 下载 → 校验 → 启动 全链路。"""

    LLAMA_SCRIPT = (
        "import sys\n"
        "from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer\n"
        "a = sys.argv[1:]\n"
        "host = a[a.index('--host') + 1]\n"
        "port = int(a[a.index('--port') + 1])\n"
        "class H(BaseHTTPRequestHandler):\n"
        "    def log_message(self, *x): pass\n"
        "    def do_GET(self):\n"
        "        b = b'{\"data\":[]}'\n"
        "        self.send_response(200)\n"
        "        self.send_header('Content-Length', str(len(b)))\n"
        "        self.end_headers()\n"
        "        self.wfile.write(b)\n"
        "ThreadingHTTPServer((host, port), H).serve_forever()\n"
    )

    def setUp(self):
        import http.server
        import socket
        import socketserver
        import sys
        import threading

        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.payload = b"GGUF" + bytes(range(256)) * 8000

        www = self.root / "www"
        www.mkdir()
        (www / "m.gguf").write_bytes(self.payload)

        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(www), **kwargs)

            def log_message(self, *extra):
                pass

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            file_port = probe.getsockname()[1]

        self.files = socketserver.ThreadingTCPServer(("127.0.0.1", file_port), Handler)
        self.addCleanup(self.files.server_close)
        self.addCleanup(self.files.shutdown)
        threading.Thread(target=self.files.serve_forever, daemon=True).start()
        self.file_url = f"http://127.0.0.1:{file_port}/m.gguf"

        self.llama_script = self.root / "fake_llama.py"
        self.llama_script.write_text(self.LLAMA_SCRIPT, encoding="utf-8")
        (self.root / "llama-server.exe").write_bytes(b"")

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.llama_port = probe.getsockname()[1]

        self._sys = sys

    def tearDown(self):
        self._tmp.cleanup()

    def test_download_verify_and_start(self):
        import time

        payload = self.payload
        manifest = ModelManifest(
            url=self.file_url,
            sha256=sha256_of(payload),
            size=len(payload),
            filename="m.gguf",
        )
        config = LocalServerConfig(
            base_url=f"http://127.0.0.1:{self.llama_port}/v1",
            model="qwen2.5-1.5b-instruct",
            auto_start=True,
            executable=str(self.root / "llama-server.exe"),
            startup_timeout=30,
        )
        data_dir = self.root / "data"
        script = self.llama_script
        sys_module = self._sys

        class FakeLlama(LocalModelServer):
            def _build_command(self):
                return [
                    sys_module.executable,
                    "-u",
                    str(script),
                    "--host",
                    self.config.host,
                    "--port",
                    str(self.config.port),
                ]

        runtime = LocalModelRuntime(
            config,
            manifest,
            data_dir=data_dir,
            server_factory=lambda effective, log_dir: FakeLlama(
                effective, log_dir=log_dir
            ),
        )

        self.assertEqual(runtime.status()["state"], STATE_NEEDS_DOWNLOAD)
        try:
            runtime.start_download()
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                state = runtime.status()["state"]
                if state in {STATE_READY, STATE_ERROR}:
                    break
                time.sleep(0.05)

            status = runtime.status()
            self.assertEqual(status["state"], STATE_READY, status.get("message"))
            self.assertEqual(runtime.model_path.read_bytes(), payload)
            self.assertFalse(runtime.downloader.partial_path.exists())
            self.assertTrue(runtime.server.is_healthy())
        finally:
            runtime.stop()

        time.sleep(0.5)
        self.assertFalse(runtime.server.is_healthy())
        # 模型文件保留，重启后不需要重新下载。
        self.assertTrue(runtime.model_path.is_file())

    def test_second_launch_does_not_redownload(self):
        payload = self.payload
        manifest = ModelManifest(
            url=self.file_url,
            sha256=sha256_of(payload),
            size=len(payload),
            filename="m.gguf",
        )
        config = base_config()
        data_dir = self.root / "data"
        model_path = data_dir / "models" / "m.gguf"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(payload)

        runtime = LocalModelRuntime(config, manifest, data_dir=data_dir)

        status = runtime.status()
        self.assertFalse(status["can_download"])
        self.assertFalse(runtime.start_download())


class RegistryTests(unittest.TestCase):
    def tearDown(self):
        set_runtime(None)

    def test_set_and_get_runtime(self):
        sentinel = object()
        set_runtime(sentinel)

        self.assertIs(get_runtime(), sentinel)

    def test_build_runtime_returns_none_for_remote(self):
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {"LLM_PROVIDER": "remote"}, clear=True):
            self.assertIsNone(build_runtime())


if __name__ == "__main__":
    unittest.main()

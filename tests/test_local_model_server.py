import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from local_model_server import (
    DEFAULT_CONTEXT_SIZE,
    DEFAULT_STARTUP_TIMEOUT,
    LocalModelServer,
    LocalModelStartupError,
    load_local_server_config,
    parse_local_server_config,
    port_in_use,
    probe_endpoint,
)


class FakeProcess:
    def __init__(self, *, returncode=None):
        self.returncode = returncode
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        if self.returncode is None:
            self.returncode = 0

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9


class StubbornProcess(FakeProcess):
    """终止后仍不退出，用于验证会升级为 kill。"""

    def wait(self, timeout=None):
        if self.killed:
            self.returncode = -9
            return -9
        raise subprocess.TimeoutExpired(cmd="llama-server", timeout=timeout)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def make_executable(directory: Path, name: str = "llama-server.exe") -> Path:
    path = directory / name
    path.write_bytes(b"")
    return path


def make_model(directory: Path) -> Path:
    path = directory / "model.gguf"
    path.write_bytes(b"gguf")
    return path


class ConfigParsingTests(unittest.TestCase):
    def test_defaults_are_applied(self):
        config = parse_local_server_config(
            {"base_url": "http://127.0.0.1:8080/v1", "model": "qwen2.5-1.5b-instruct"}
        )

        self.assertFalse(config.auto_start)
        self.assertEqual(config.context_size, DEFAULT_CONTEXT_SIZE)
        self.assertEqual(config.startup_timeout, DEFAULT_STARTUP_TIMEOUT)
        self.assertEqual(config.host, "127.0.0.1")
        self.assertEqual(config.port, 8080)

    def test_auto_start_accepts_common_truthy_values(self):
        for raw in (True, "true", "1", "yes", "on", "TRUE"):
            with self.subTest(raw=raw):
                config = parse_local_server_config(
                    {"base_url": "http://127.0.0.1:8080/v1", "auto_start": raw}
                )
                self.assertTrue(config.auto_start)

    def test_auto_start_rejects_falsey_values(self):
        for raw in (False, "false", "0", "", None):
            with self.subTest(raw=raw):
                config = parse_local_server_config(
                    {"base_url": "http://127.0.0.1:8080/v1", "auto_start": raw}
                )
                self.assertFalse(config.auto_start)

    def test_relative_paths_resolve_against_install_dir(self):
        config = parse_local_server_config(
            {
                "base_url": "http://127.0.0.1:8080/v1",
                "local_server": r"llama\llama-server.exe",
                "local_model_path": "models/model.gguf",
            },
            base_dir=Path("C:/Program Files/ExcelAssistant"),
        )

        self.assertTrue(Path(config.executable).is_absolute())
        self.assertTrue(Path(config.model_path).is_absolute())
        self.assertIn("llama-server.exe", config.executable)
        self.assertIn("model.gguf", config.model_path)

    def test_absolute_paths_are_kept(self):
        config = parse_local_server_config(
            {
                "base_url": "http://127.0.0.1:8080/v1",
                "local_model_path": r"D:\models\m.gguf",
            },
            base_dir=Path("C:/elsewhere"),
        )

        self.assertEqual(config.model_path, r"D:\models\m.gguf")

    def test_extra_args_support_list_and_comma_string(self):
        as_list = parse_local_server_config(
            {"base_url": "http://127.0.0.1:8080/v1", "extra_args": ["--threads", "4"]}
        )
        as_text = parse_local_server_config(
            {"base_url": "http://127.0.0.1:8080/v1", "extra_args": "--threads,4"}
        )

        self.assertEqual(as_list.extra_args, ("--threads", "4"))
        self.assertEqual(as_text.extra_args, ("--threads", "4"))

    def test_invalid_numbers_fall_back_to_defaults(self):
        config = parse_local_server_config(
            {
                "base_url": "http://127.0.0.1:8080/v1",
                "context_size": "many",
                "startup_timeout": "-5",
            }
        )

        self.assertEqual(config.context_size, DEFAULT_CONTEXT_SIZE)
        self.assertEqual(config.startup_timeout, DEFAULT_STARTUP_TIMEOUT)

    def test_port_defaults_follow_scheme(self):
        self.assertEqual(
            parse_local_server_config({"base_url": "http://127.0.0.1/v1"}).port, 80
        )
        self.assertEqual(
            parse_local_server_config({"base_url": "https://host/v1"}).port, 443
        )


class LoadConfigTests(unittest.TestCase):
    def test_remote_mode_returns_none(self):
        with (
            patch("local_model_server.sys.frozen", True, create=True),
            patch(
                "local_model_server.read_packaged_model_config",
                return_value={"provider": "remote", "base_url": "https://api.test/v1"},
            ),
        ):
            self.assertIsNone(load_local_server_config())

    def test_missing_provider_is_treated_as_remote(self):
        with (
            patch("local_model_server.sys.frozen", True, create=True),
            patch(
                "local_model_server.read_packaged_model_config",
                return_value={"base_url": "https://api.test/v1"},
            ),
        ):
            self.assertIsNone(load_local_server_config())

    def test_frozen_local_config_is_parsed(self):
        with (
            patch("local_model_server.sys.frozen", True, create=True),
            patch("local_model_server.sys.executable", r"C:\App\ExcelAssistant.exe"),
            patch(
                "local_model_server.read_packaged_model_config",
                return_value={
                    "provider": "local",
                    "base_url": "http://127.0.0.1:8080/v1",
                    "model": "qwen2.5-1.5b-instruct",
                    "auto_start": True,
                    "local_server": r"llama\llama-server.exe",
                },
            ),
        ):
            config = load_local_server_config()

        assert config is not None
        self.assertTrue(config.auto_start)
        self.assertTrue(Path(config.executable).is_absolute())
        self.assertIn("llama-server.exe", config.executable)

    def test_development_mode_reads_environment(self):
        with patch.dict(
            "os.environ",
            {
                "LLM_PROVIDER": "local",
                "LOCAL_BASE_URL": "http://127.0.0.1:8080/v1",
                "LOCAL_MODEL": "qwen2.5-1.5b-instruct",
                "LOCAL_AUTO_START": "1",
                "LOCAL_SERVER_PATH": "llama-server",
                "LOCAL_MODEL_PATH": "model.gguf",
            },
            clear=True,
        ):
            config = load_local_server_config()

        assert config is not None
        self.assertTrue(config.auto_start)
        self.assertEqual(config.model, "qwen2.5-1.5b-instruct")

    def test_development_remote_mode_returns_none(self):
        with patch.dict("os.environ", {"LLM_PROVIDER": "remote"}, clear=True):
            self.assertIsNone(load_local_server_config())


class EnsureRunningTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.log_dir = self.root / "logs"

    def tearDown(self):
        self._tmp.cleanup()

    def _config(self, **overrides):
        base = {
            "base_url": "http://127.0.0.1:8080/v1",
            "model": "qwen2.5-1.5b-instruct",
        }
        base.update(overrides)
        return parse_local_server_config(base, base_dir=self.root)

    def test_reuses_an_already_healthy_service(self):
        spawned = []

        def popen(*args, **kwargs):
            spawned.append(args)
            return FakeProcess()

        server = LocalModelServer(
            self._config(auto_start=True),
            log_dir=self.log_dir,
            popen=popen,
            probe=lambda url: True,
            port_check=lambda host, port: False,
        )

        self.assertFalse(server.ensure_running())
        self.assertEqual(spawned, [])

    def test_manual_mode_reports_bind_address(self):
        server = LocalModelServer(
            self._config(),
            log_dir=self.log_dir,
            probe=lambda url: False,
            port_check=lambda host, port: False,
        )

        with self.assertRaises(LocalModelStartupError) as captured:
            server.ensure_running()

        message = str(captured.exception)
        self.assertIn("http://127.0.0.1:8080/v1", message)
        self.assertIn("未启用自动启动", message)

    def test_missing_executable_is_reported(self):
        server = LocalModelServer(
            self._config(auto_start=True, local_model_path="model.gguf"),
            log_dir=self.log_dir,
            probe=lambda url: False,
            port_check=lambda host, port: False,
        )

        with self.assertRaisesRegex(LocalModelStartupError, "没有配置 llama-server"):
            server.ensure_running()

    def test_nonexistent_executable_is_reported(self):
        server = LocalModelServer(
            self._config(
                auto_start=True,
                local_server="missing/llama-server.exe",
                local_model_path="model.gguf",
            ),
            log_dir=self.log_dir,
            probe=lambda url: False,
            port_check=lambda host, port: False,
        )

        with self.assertRaisesRegex(LocalModelStartupError, "找不到本地模型服务程序"):
            server.ensure_running()

    def test_missing_model_path_is_reported(self):
        executable = make_executable(self.root)
        server = LocalModelServer(
            self._config(auto_start=True, local_server=str(executable)),
            log_dir=self.log_dir,
            probe=lambda url: False,
            port_check=lambda host, port: False,
        )

        with self.assertRaisesRegex(LocalModelStartupError, "没有配置模型文件"):
            server.ensure_running()

    def test_nonexistent_model_file_is_reported(self):
        executable = make_executable(self.root)
        server = LocalModelServer(
            self._config(
                auto_start=True,
                local_server=str(executable),
                local_model_path="missing.gguf",
            ),
            log_dir=self.log_dir,
            probe=lambda url: False,
            port_check=lambda host, port: False,
        )

        with self.assertRaisesRegex(LocalModelStartupError, "找不到模型文件"):
            server.ensure_running()

    def test_occupied_port_is_reported(self):
        executable = make_executable(self.root)
        model = make_model(self.root)
        server = LocalModelServer(
            self._config(
                auto_start=True,
                local_server=str(executable),
                local_model_path=str(model),
            ),
            log_dir=self.log_dir,
            probe=lambda url: False,
            port_check=lambda host, port: True,
        )

        with self.assertRaisesRegex(LocalModelStartupError, "已被其他程序占用"):
            server.ensure_running()

    def test_non_loopback_host_is_rejected(self):
        executable = make_executable(self.root)
        model = make_model(self.root)
        server = LocalModelServer(
            self._config(
                base_url="http://192.168.1.9:8080/v1",
                auto_start=True,
                local_server=str(executable),
                local_model_path=str(model),
            ),
            log_dir=self.log_dir,
            probe=lambda url: False,
            port_check=lambda host, port: False,
        )

        with self.assertRaisesRegex(LocalModelStartupError, "回环地址"):
            server.ensure_running()

    def test_successful_start_uses_expected_arguments(self):
        executable = make_executable(self.root)
        model = make_model(self.root)
        captured = {}

        def popen(command, **kwargs):
            captured["command"] = command
            captured["kwargs"] = kwargs
            return FakeProcess()

        probes = {"count": 0}

        def probe(url):
            probes["count"] += 1
            return probes["count"] > 1

        server = LocalModelServer(
            self._config(
                auto_start=True,
                local_server=str(executable),
                local_model_path=str(model),
                context_size=4096,
                extra_args=["--threads", "4"],
            ),
            log_dir=self.log_dir,
            popen=popen,
            probe=probe,
            port_check=lambda host, port: False,
        )

        self.assertTrue(server.ensure_running())

        command = captured["command"]
        self.assertEqual(command[0], str(executable))
        self.assertEqual(command[command.index("-m") + 1], str(model))
        self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
        self.assertEqual(command[command.index("--port") + 1], "8080")
        self.assertEqual(command[command.index("-c") + 1], "4096")
        self.assertIn("--jinja", command)
        self.assertEqual(command[-2:], ["--threads", "4"])
        self.assertFalse(captured["kwargs"]["stderr"] is subprocess.PIPE)
        self.assertTrue(server.log_path.is_file())

        server.stop()

    def test_immediate_exit_reports_code_and_log(self):
        executable = make_executable(self.root)
        model = make_model(self.root)
        server = LocalModelServer(
            self._config(
                auto_start=True,
                local_server=str(executable),
                local_model_path=str(model),
            ),
            log_dir=self.log_dir,
            popen=lambda command, **kwargs: FakeProcess(returncode=3),
            probe=lambda url: False,
            port_check=lambda host, port: False,
        )

        with self.assertRaises(LocalModelStartupError) as captured:
            server.ensure_running()

        message = str(captured.exception)
        self.assertIn("退出码 3", message)
        self.assertIn("local-model.log", message)

    def test_timeout_reports_duration(self):
        executable = make_executable(self.root)
        model = make_model(self.root)
        clock = FakeClock()
        server = LocalModelServer(
            self._config(
                auto_start=True,
                local_server=str(executable),
                local_model_path=str(model),
                startup_timeout=2,
            ),
            log_dir=self.log_dir,
            popen=lambda command, **kwargs: FakeProcess(),
            probe=lambda url: False,
            port_check=lambda host, port: False,
            clock=clock,
            sleep=clock.sleep,
        )

        with self.assertRaises(LocalModelStartupError) as captured:
            server.ensure_running()

        message = str(captured.exception)
        self.assertIn("2 秒内未就绪", message)
        self.assertIn("startup_timeout", message)

    def test_stop_terminates_the_process_it_started(self):
        executable = make_executable(self.root)
        model = make_model(self.root)
        process = FakeProcess()
        probes = {"count": 0}

        def probe(url):
            probes["count"] += 1
            return probes["count"] > 1

        server = LocalModelServer(
            self._config(
                auto_start=True,
                local_server=str(executable),
                local_model_path=str(model),
            ),
            log_dir=self.log_dir,
            popen=lambda command, **kwargs: process,
            probe=probe,
            port_check=lambda host, port: False,
        )

        server.ensure_running()
        server.stop()

        self.assertTrue(process.terminated)

    def test_stop_escalates_to_kill(self):
        executable = make_executable(self.root)
        model = make_model(self.root)
        process = StubbornProcess()
        probes = {"count": 0}

        def probe(url):
            probes["count"] += 1
            return probes["count"] > 1

        server = LocalModelServer(
            self._config(
                auto_start=True,
                local_server=str(executable),
                local_model_path=str(model),
            ),
            log_dir=self.log_dir,
            popen=lambda command, **kwargs: process,
            probe=probe,
            port_check=lambda host, port: False,
        )

        server.ensure_running()
        server.stop()

        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)

    def test_stop_is_idempotent_and_safe_when_never_started(self):
        server = LocalModelServer(
            self._config(), log_dir=self.log_dir, port_check=lambda host, port: False
        )

        server.stop()
        server.stop()

    def test_failed_start_cleans_up_the_process(self):
        executable = make_executable(self.root)
        model = make_model(self.root)
        process = FakeProcess()
        clock = FakeClock()

        server = LocalModelServer(
            self._config(
                auto_start=True,
                local_server=str(executable),
                local_model_path=str(model),
                startup_timeout=1,
            ),
            log_dir=self.log_dir,
            popen=lambda command, **kwargs: process,
            probe=lambda url: False,
            port_check=lambda host, port: False,
            clock=clock,
            sleep=clock.sleep,
        )

        with self.assertRaises(LocalModelStartupError):
            server.ensure_running()

        self.assertTrue(process.terminated)

    def test_context_manager_stops_on_exit(self):
        executable = make_executable(self.root)
        model = make_model(self.root)
        process = FakeProcess()
        probes = {"count": 0}

        def probe(url):
            probes["count"] += 1
            return probes["count"] > 1

        server = LocalModelServer(
            self._config(
                auto_start=True,
                local_server=str(executable),
                local_model_path=str(model),
            ),
            log_dir=self.log_dir,
            popen=lambda command, **kwargs: process,
            probe=probe,
            port_check=lambda host, port: False,
        )

        with server:
            self.assertTrue(server.owns_process)

        self.assertTrue(process.terminated)
        self.assertFalse(server.owns_process)


class RealProcessLifecycleTests(unittest.TestCase):
    """用真实子进程验证 启动 → 就绪探测 → 关闭 的完整链路。"""

    SCRIPT = (
        "import sys\n"
        "from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer\n"
        "args = sys.argv[1:]\n"
        "host = args[args.index('--host') + 1]\n"
        "port = int(args[args.index('--port') + 1])\n"
        "class H(BaseHTTPRequestHandler):\n"
        "    def log_message(self, *a): pass\n"
        "    def do_GET(self):\n"
        "        body = b'{\"data\":[]}' if self.path.endswith('/models') else b'{\"status\":\"ok\"}'\n"
        "        self.send_response(200)\n"
        "        self.send_header('Content-Type', 'application/json')\n"
        "        self.send_header('Content-Length', str(len(body)))\n"
        "        self.end_headers()\n"
        "        self.wfile.write(body)\n"
        "ThreadingHTTPServer((host, port), H).serve_forever()\n"
    )

    def _free_port(self) -> int:
        import socket

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            return int(probe.getsockname()[1])

    def test_start_and_stop_a_real_subprocess(self):
        import sys

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "fake_server.py"
            script.write_text(self.SCRIPT, encoding="utf-8")
            make_executable(root)
            make_model(root)
            port = self._free_port()

            class FakeLlamaServer(LocalModelServer):
                def _build_command(self):
                    return [
                        sys.executable,
                        "-u",
                        str(script),
                        "--host",
                        self.config.host,
                        "--port",
                        str(self.config.port),
                    ]

            config = parse_local_server_config(
                {
                    "base_url": f"http://127.0.0.1:{port}/v1",
                    "model": "qwen2.5-1.5b-instruct",
                    "auto_start": True,
                    "local_server": "llama-server.exe",
                    "local_model_path": "model.gguf",
                    "startup_timeout": "20",
                },
                base_dir=root,
            )
            server = FakeLlamaServer(config, log_dir=root / "logs")

            self.assertFalse(server.is_healthy())
            try:
                self.assertTrue(server.ensure_running())
                self.assertTrue(server.is_healthy())
                process = server._process
                assert process is not None
                self.assertIsNone(process.poll())

                # 已有健康服务时应复用而不是再起一个。
                second = FakeLlamaServer(config, log_dir=root / "logs")
                self.assertFalse(second.ensure_running())
                self.assertIsNone(second._process)
            finally:
                server.stop()

            time.sleep(0.5)
            self.assertIsNotNone(process.poll())
            self.assertFalse(server.is_healthy())
            self.assertTrue(server.log_path.is_file())


class ProbeTests(unittest.TestCase):
    def test_invalid_url_is_not_healthy(self):
        self.assertFalse(probe_endpoint(""))
        self.assertFalse(probe_endpoint("not-a-url"))

    def test_closed_port_short_circuits_the_http_probe(self):
        """端口没在监听时不应再等 HTTP 超时，否则每次健康检查都很慢。"""
        with (
            patch("local_model_server.port_in_use", return_value=False),
            patch("local_model_server._http_ok") as http_ok,
        ):
            self.assertFalse(probe_endpoint("http://127.0.0.1:8080/v1"))

        http_ok.assert_not_called()

    def test_listening_port_is_probed_over_http(self):
        with (
            patch("local_model_server.port_in_use", return_value=True),
            patch("local_model_server._http_ok", return_value=True) as http_ok,
        ):
            self.assertTrue(probe_endpoint("http://127.0.0.1:8080/v1"))

        self.assertTrue(http_ok.called)
        self.assertIn("/v1/models", http_ok.call_args.args[0])

    def test_health_endpoint_is_used_as_fallback(self):
        responses = {"/v1/models": False, "/health": True}

        def fake_http(url, timeout=0.0):
            return any(url.endswith(suffix) and ok for suffix, ok in responses.items())

        with (
            patch("local_model_server.port_in_use", return_value=True),
            patch("local_model_server._http_ok", side_effect=fake_http),
        ):
            self.assertTrue(probe_endpoint("http://127.0.0.1:8080/v1"))

    def test_port_in_use_detects_a_listening_socket(self):
        import socket

        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port = listener.getsockname()[1]

            self.assertTrue(port_in_use("127.0.0.1", port))

    def test_port_in_use_returns_false_for_closed_port(self):
        import socket

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]

        self.assertFalse(port_in_use("127.0.0.1", port, timeout=0.2))


if __name__ == "__main__":
    unittest.main()

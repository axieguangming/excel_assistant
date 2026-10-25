import hashlib
import io
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from model_download import (
    DownloadStatus,
    ModelDownloader,
    ModelDownloadError,
    ModelManifest,
    manifest_from_config,
)

PAYLOAD = b"GGUF" + bytes(range(256)) * 40


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FakeResponse:
    """模拟 urllib 响应，支持分块读取与 Content-Length。"""

    def __init__(self, data: bytes, *, status: int = 200, length: int | None = None):
        self._buffer = io.BytesIO(data)
        self.status = status
        self.headers = {}
        if length is not None:
            self.headers["Content-Length"] = str(length)

    def read(self, size: int = -1) -> bytes:
        return self._buffer.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class ManifestTests(unittest.TestCase):
    def test_filename_derived_from_url(self):
        manifest = manifest_from_config({"model_url": "https://x.test/a/b/model.gguf"})

        self.assertEqual(manifest.effective_filename, "model.gguf")
        self.assertTrue(manifest.is_configured)

    def test_explicit_filename_wins(self):
        manifest = manifest_from_config(
            {"model_url": "https://x.test/a/model.gguf", "model_filename": "custom.gguf"}
        )

        self.assertEqual(manifest.effective_filename, "custom.gguf")

    def test_empty_config_is_not_configured(self):
        manifest = manifest_from_config({})

        self.assertFalse(manifest.is_configured)
        self.assertEqual(manifest.effective_filename, "model.gguf")

    def test_values_are_coerced_from_strings(self):
        manifest = manifest_from_config(
            {"model_url": "https://x.test/m.gguf", "model_size": "12345"}
        )

        self.assertEqual(manifest.size, 12345)

    def test_invalid_size_falls_back_to_zero(self):
        manifest = manifest_from_config(
            {"model_url": "https://x.test/m.gguf", "model_size": "big"}
        )

        self.assertEqual(manifest.size, 0)


class DownloadStatusTests(unittest.TestCase):
    def test_percent_is_clamped_and_handles_unknown_total(self):
        self.assertEqual(DownloadStatus(downloaded=5, total=10).percent, 50.0)
        self.assertEqual(DownloadStatus(downloaded=20, total=10).percent, 100.0)
        self.assertEqual(DownloadStatus(downloaded=5, total=0).percent, 0.0)

    def test_dict_shape(self):
        payload = DownloadStatus(state="downloading", downloaded=1, total=2).as_dict()

        self.assertEqual(payload["state"], "downloading")
        self.assertEqual(payload["percent"], 50.0)


class DownloaderTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.target = self.root / "models" / "model.gguf"

    def tearDown(self):
        self._tmp.cleanup()

    def _downloader(self, manifest=None, responses=None, **kwargs):
        manifest = manifest or ModelManifest(
            url="https://x.test/model.gguf",
            sha256=sha256_of(PAYLOAD),
            size=len(PAYLOAD),
        )
        responses = list(responses) if responses else [
            FakeResponse(PAYLOAD, length=len(PAYLOAD))
        ]
        calls = []

        def open_url(request, timeout=None):
            calls.append(request)
            return responses.pop(0)

        downloader = ModelDownloader(
            manifest, self.target, open_url=open_url, **kwargs
        )
        downloader.calls = calls
        return downloader

    def test_download_writes_file_and_reports_ready(self):
        downloader = self._downloader()

        path = downloader.download()

        self.assertEqual(path, self.target)
        self.assertEqual(path.read_bytes(), PAYLOAD)
        self.assertEqual(downloader.status.state, "ready")
        self.assertFalse(downloader.partial_path.exists())

    def test_existing_valid_file_is_reused_without_network(self):
        self.target.parent.mkdir(parents=True, exist_ok=True)
        self.target.write_bytes(PAYLOAD)
        downloader = self._downloader()
        downloader._open_url = lambda *a, **k: self.fail("不应发起网络请求")

        path = downloader.download()

        self.assertEqual(path.read_bytes(), PAYLOAD)
        self.assertEqual(downloader.status.state, "ready")

    def test_corrupted_existing_file_is_replaced(self):
        self.target.parent.mkdir(parents=True, exist_ok=True)
        self.target.write_bytes(PAYLOAD[:10])
        downloader = self._downloader()

        downloader.download()

        self.assertEqual(self.target.read_bytes(), PAYLOAD)

    def test_sha256_mismatch_deletes_partial_and_raises(self):
        manifest = ModelManifest(
            url="https://x.test/model.gguf",
            sha256="0" * 64,
            size=len(PAYLOAD),
        )
        downloader = self._downloader(manifest)

        with self.assertRaisesRegex(ModelDownloadError, "校验失败"):
            downloader.download()

        self.assertFalse(downloader.partial_path.exists())
        self.assertFalse(self.target.exists())

    def test_size_mismatch_is_reported(self):
        manifest = ModelManifest(url="https://x.test/model.gguf", size=len(PAYLOAD) + 5)
        downloader = self._downloader(manifest)

        with self.assertRaisesRegex(ModelDownloadError, "下载不完整"):
            downloader.download()

        self.assertFalse(self.target.exists())

    def test_resume_sends_range_header_and_appends(self):
        self.target.parent.mkdir(parents=True, exist_ok=True)
        partial = self.target.with_suffix(".gguf.part")
        head = PAYLOAD[:100]
        partial.write_bytes(head)

        response = FakeResponse(PAYLOAD[100:], status=206, length=len(PAYLOAD) - 100)
        downloader = self._downloader(responses=[response])

        downloader.download()

        self.assertEqual(self.target.read_bytes(), PAYLOAD)
        self.assertEqual(downloader.calls[0].get_header("Range"), f"bytes={len(head)}-")

    def test_no_range_header_for_fresh_download(self):
        downloader = self._downloader()

        downloader.download()

        self.assertIsNone(downloader.calls[0].get_header("Range"))

    def test_range_not_satisfiable_restarts_from_scratch(self):
        self.target.parent.mkdir(parents=True, exist_ok=True)
        self.target.with_suffix(".gguf.part").write_bytes(b"stale")

        from urllib.error import HTTPError

        error = HTTPError("https://x.test/model.gguf", 416, "range", {}, None)
        calls = [error, FakeResponse(PAYLOAD, length=len(PAYLOAD))]

        def open_url(request, timeout=None):
            item = calls.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        manifest = ModelManifest(
            url="https://x.test/model.gguf",
            sha256=sha256_of(PAYLOAD),
            size=len(PAYLOAD),
        )
        downloader = ModelDownloader(manifest, self.target, open_url=open_url)

        downloader.download()

        self.assertEqual(self.target.read_bytes(), PAYLOAD)

    def test_http_error_is_reported_as_download_error(self):
        from urllib.error import HTTPError

        def open_url(request, timeout=None):
            raise HTTPError("https://x.test/model.gguf", 404, "not found", {}, None)

        manifest = ModelManifest(url="https://x.test/model.gguf")
        downloader = ModelDownloader(manifest, self.target, open_url=open_url)

        with self.assertRaisesRegex(ModelDownloadError, "HTTP 404"):
            downloader.download()

    def test_network_error_is_reported(self):
        from urllib.error import URLError

        def open_url(request, timeout=None):
            raise URLError("no route")

        manifest = ModelManifest(url="https://x.test/model.gguf")
        downloader = ModelDownloader(manifest, self.target, open_url=open_url)

        with self.assertRaisesRegex(ModelDownloadError, "无法连接"):
            downloader.download()

    def test_missing_url_raises(self):
        downloader = ModelDownloader(ModelManifest(url=""), self.target)

        with self.assertRaisesRegex(ModelDownloadError, "未配置模型下载地址"):
            downloader.download()

    def test_insufficient_disk_space_is_reported(self):
        downloader = self._downloader()
        with patch("model_download.shutil.disk_usage") as usage:
            usage.return_value = type("U", (), {"free": 1024})()
            with self.assertRaisesRegex(ModelDownloadError, "磁盘空间不足"):
                downloader.download()

    def test_is_installed_uses_size_when_known(self):
        downloader = self._downloader()
        self.assertFalse(downloader.is_installed())

        self.target.parent.mkdir(parents=True, exist_ok=True)
        self.target.write_bytes(PAYLOAD[:5])

        self.assertFalse(downloader.is_installed())

    def test_verify_without_sha256_only_checks_size(self):
        manifest = ModelManifest(url="https://x.test/m.gguf", size=len(PAYLOAD))
        downloader = ModelDownloader(manifest, self.target)
        self.target.parent.mkdir(parents=True, exist_ok=True)
        self.target.write_bytes(PAYLOAD)

        ok, reason = downloader.verify()

        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_verify_reports_size_mismatch(self):
        manifest = ModelManifest(url="https://x.test/m.gguf", size=1)
        downloader = ModelDownloader(manifest, self.target)
        self.target.parent.mkdir(parents=True, exist_ok=True)
        self.target.write_bytes(PAYLOAD)

        ok, reason = downloader.verify()

        self.assertFalse(ok)
        self.assertIn("大小不符", reason)

    def test_verify_reports_missing_file(self):
        downloader = self._downloader()

        ok, reason = downloader.verify()

        self.assertFalse(ok)
        self.assertIn("不存在", reason)

    def test_refresh_state_reports_partial_progress(self):
        self.target.parent.mkdir(parents=True, exist_ok=True)
        self.target.with_suffix(".gguf.part").write_bytes(PAYLOAD[:7])
        downloader = self._downloader()

        status = downloader.refresh_state()

        self.assertEqual(status.state, "missing")
        self.assertEqual(status.downloaded, 7)

    def test_refresh_state_reports_ready_for_installed_file(self):
        self.target.parent.mkdir(parents=True, exist_ok=True)
        self.target.write_bytes(PAYLOAD)
        downloader = self._downloader()

        status = downloader.refresh_state()

        self.assertEqual(status.state, "ready")

    def test_background_start_runs_once(self):
        gate = threading.Event()

        def open_url(request, timeout=None):
            gate.wait(5)
            return FakeResponse(PAYLOAD, length=len(PAYLOAD))

        manifest = ModelManifest(
            url="https://x.test/model.gguf",
            sha256=sha256_of(PAYLOAD),
            size=len(PAYLOAD),
        )
        downloader = ModelDownloader(manifest, self.target, open_url=open_url)

        self.assertTrue(downloader.start())
        self.assertFalse(downloader.start())
        gate.set()
        downloader.wait(10)

        self.assertEqual(downloader.status.state, "ready")
        self.assertEqual(self.target.read_bytes(), PAYLOAD)

    def test_background_failure_sets_error_state(self):
        def open_url(request, timeout=None):
            raise OSError("boom")

        manifest = ModelManifest(url="https://x.test/model.gguf")
        downloader = ModelDownloader(manifest, self.target, open_url=open_url)

        downloader.start()
        downloader.wait(10)

        self.assertEqual(downloader.status.state, "error")
        self.assertTrue(downloader.status.error)

    def test_progress_is_reported_during_download(self):
        collected = []
        clock = {"now": 0.0}

        def fake_clock():
            # 每次读取进度时推进时间，触发一次上报。
            clock["now"] += 1.0
            return clock["now"]

        manifest = ModelManifest(
            url="https://x.test/model.gguf",
            sha256=sha256_of(PAYLOAD),
            size=len(PAYLOAD),
        )
        downloader = ModelDownloader(
            manifest,
            self.target,
            open_url=lambda request, timeout=None: FakeResponse(
                PAYLOAD, length=len(PAYLOAD)
            ),
            clock=fake_clock,
            chunk_size=64,
        )

        downloader.download()

        status = downloader.status
        self.assertEqual(status.state, "ready")
        self.assertEqual(status.downloaded, len(PAYLOAD))


if __name__ == "__main__":
    unittest.main()

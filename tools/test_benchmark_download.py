"""Pinned release acquisition only: all HTTP, sleeps and payloads are synthetic."""
import contextlib
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import call, patch
from urllib.error import HTTPError

import benchmark


PAYLOAD = b"fixed release bytes"
SHA = hashlib.sha256(PAYLOAD).hexdigest()
NAME = "sample-bots-Java-1.4.0.zip"
URL = f"https://github.com/robocode-dev/tank-royale/releases/download/v{benchmark.VERSION}/{NAME}"


def http_error(code):
    return HTTPError(URL, code, "synthetic failure", {}, io.BytesIO())


class DownloadTests(unittest.TestCase):
    def test_each_transient_status_retries_same_pinned_url_then_publishes_verified_bytes(self):
        for code in (502, 503, 504):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as directory:
                cache = Path(directory)
                with patch.object(benchmark.urllib.request, "urlopen", side_effect=[http_error(code), io.BytesIO(PAYLOAD)]) as request, \
                        patch.object(benchmark.time, "sleep") as sleep, contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(cache / NAME, benchmark.download(cache, NAME, SHA))
                self.assertEqual([call(1)], sleep.call_args_list)
                self.assertEqual(2, request.call_count)
                for args, kwargs in request.call_args_list:
                    self.assertEqual(URL, args[0].full_url)
                    self.assertEqual("AISurvivor-benchmark", args[0].get_header("User-agent"))
                    self.assertEqual({"timeout": 60}, kwargs)
                self.assertEqual(PAYLOAD, (cache / NAME).read_bytes())
                self.assertEqual([cache / NAME], list(cache.iterdir()))

    def test_three_attempts_use_only_fixed_one_and_two_second_backoff(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(benchmark.urllib.request, "urlopen", side_effect=[http_error(502), http_error(504), io.BytesIO(PAYLOAD)]) as request, \
                patch.object(benchmark.time, "sleep") as sleep, contextlib.redirect_stderr(io.StringIO()):
            benchmark.download(Path(directory), NAME, SHA)
            self.assertEqual(3, request.call_count)
            self.assertEqual([call(1), call(2)], sleep.call_args_list)

    def test_exhausted_transient_failures_propagate_without_cache_or_partial_publication(self):
        error = http_error(503)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(benchmark.urllib.request, "urlopen", side_effect=error) as request, \
                patch.object(benchmark.time, "sleep") as sleep, contextlib.redirect_stderr(io.StringIO()):
            cache = Path(directory)
            with self.assertRaises(HTTPError) as failure:
                benchmark.download(cache, NAME, SHA)
            self.assertIs(error, failure.exception)
            self.assertEqual(3, request.call_count)
            self.assertEqual([call(1), call(2)], sleep.call_args_list)
            self.assertEqual([], list(cache.iterdir()))

    def test_permanent_http_errors_are_never_retried(self):
        for code in (400, 403, 404, 500):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as directory, \
                    patch.object(benchmark.urllib.request, "urlopen", side_effect=http_error(code)) as request, \
                    patch.object(benchmark.time, "sleep") as sleep:
                cache = Path(directory)
                with self.assertRaises(HTTPError):
                    benchmark.download(cache, NAME, SHA)
                self.assertEqual(1, request.call_count)
                sleep.assert_not_called()
                self.assertEqual([], list(cache.iterdir()))

    def test_digest_mismatch_is_not_retried_and_preserves_existing_cache_bytes(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(benchmark.urllib.request, "urlopen", return_value=io.BytesIO(b"wrong new bytes")) as request, \
                patch.object(benchmark.time, "sleep") as sleep:
            cache = Path(directory)
            target = cache / NAME
            target.write_bytes(b"old invalid cache")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                benchmark.download(cache, NAME, SHA)
            self.assertEqual(1, request.call_count)
            sleep.assert_not_called()
            self.assertEqual(b"old invalid cache", target.read_bytes())
            self.assertEqual([target], list(cache.iterdir()))

    def test_valid_cache_hit_needs_no_request_sleep_or_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(benchmark.urllib.request, "urlopen") as request, \
                patch.object(benchmark.time, "sleep") as sleep:
            cache = Path(directory)
            target = cache / NAME
            target.write_bytes(PAYLOAD)
            self.assertEqual(target, benchmark.download(cache, NAME, SHA))
            request.assert_not_called()
            sleep.assert_not_called()
            self.assertEqual([target], list(cache.iterdir()))

    def test_stream_failure_cleans_partial_and_is_not_retried(self):
        class BrokenResponse(io.BytesIO):
            def read(self, size=-1):
                raise OSError("synthetic interrupted stream")
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(benchmark.urllib.request, "urlopen", return_value=BrokenResponse()) as request, \
                patch.object(benchmark.time, "sleep") as sleep:
            cache = Path(directory)
            with self.assertRaisesRegex(OSError, "interrupted stream"):
                benchmark.download(cache, NAME, SHA)
            self.assertEqual(1, request.call_count)
            sleep.assert_not_called()
            self.assertEqual([], list(cache.iterdir()))

    def test_interleaved_callers_own_distinct_temporary_files(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            snapshots = []
            def response(*args, **kwargs):
                snapshots.append(set(cache.glob("*.part")))
                if len(snapshots) == 1:
                    benchmark.download(cache, NAME, SHA)
                return io.BytesIO(PAYLOAD)
            with patch.object(benchmark.urllib.request, "urlopen", side_effect=response):
                benchmark.download(cache, NAME, SHA)
            self.assertEqual([1, 2], [len(snapshot) for snapshot in snapshots])
            self.assertTrue(snapshots[0] < snapshots[1])
            self.assertEqual([cache / NAME], list(cache.iterdir()))
            self.assertEqual(PAYLOAD, (cache / NAME).read_bytes())


if __name__ == "__main__":
    unittest.main()

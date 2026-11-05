import hashlib
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import facet_python_runtime as runtime


class RuntimeIdentity(unittest.TestCase):
    def test_archive_identity_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'source.tar.xz';path.write_bytes(b'pinned-fixture')
            expected=hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.object(runtime,'SIZE',14),patch.object(runtime,'SHA256',expected):
                runtime.check_archive(path)
                path.write_bytes(b'changed-bytes!')
                with self.assertRaises(ValueError):runtime.check_archive(path)

    def test_no_architecture_fallback(self):
        with patch.object(runtime.platform,'machine',return_value='unsupported'):
            with self.assertRaises(ValueError):runtime.main(SimpleNamespace(prefix=Path("/unused-runtime"),evidence=Path("/unused-evidence")))


if __name__=='__main__':unittest.main()

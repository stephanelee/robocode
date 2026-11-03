"""Missing zero-byte pinned files must exist before the exact source-set guard."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import strong_melee_opponents as native


class EmptyPinnedFileTest(unittest.TestCase):
    def test_prepare_materializes_absent_empty_package_file(self):
        class StopBeforeRuntime(Exception):pass
        path='bots/bot_core/__init__.py';empty=hashlib.sha256(b'').hexdigest()
        lock=dict(commit='empty-fixture',repository='fixture/source',files=[dict(path=path,size=0,sha256=empty)],
            sourceSHA256=hashlib.sha256((path+'\0'+empty+'\n').encode()).hexdigest(),opponents=[],runtime=dict(wheels=[]))
        with tempfile.TemporaryDirectory() as tmp:
            with (patch.object(native,'manifest',return_value=lock),patch.object(native.retcon,'read_url',return_value=b'') as download,
                    patch.object(native.subprocess,'run',side_effect=StopBeforeRuntime)):
                with self.assertRaises(StopBeforeRuntime):native.prepare(Path(tmp),'unused-python')
            created=Path(tmp)/'strong-python/empty-fixture/source'/path
            self.assertTrue(created.is_file());self.assertEqual(b'',created.read_bytes());self.assertEqual(1,download.call_count)


if __name__=='__main__':unittest.main()

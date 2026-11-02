import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import retcon


class RetconTests(unittest.TestCase):
    def test_source_hash_matches_catalog_algorithm_and_detects_extra_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'B.java').write_bytes(b'B')
            (root / 'A.java').write_bytes(b'A')
            expected = hashlib.sha256(b'A.java\0A\0B.java\0B\0').hexdigest()
            self.assertEqual(retcon.source_hash(root), expected)
            (root / 'unexpected').write_bytes(b'x')
            self.assertNotEqual(retcon.source_hash(root), expected)

    def test_rejects_source_traversal_and_absolute_paths(self):
        for path in ('../secret', '/absolute', 'C:/absolute', 'dir\\escape', '.', ''):
            with self.subTest(path=path), self.assertRaises(ValueError):
                retcon.checked_relative(path)

    def test_wrong_source_bytes_fail_before_dependency_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock = root / 'lock.json'
            lock.write_text(json.dumps(dict(commit='a' * 40, repository='example/test',
                sourcePath='bots/Retcon', sourceSHA256='0' * 64,
                files=[dict(path='Retcon.json', size=4, gitBlobSHA='0' * 40)])))
            with patch.object(retcon, 'LOCK', lock), patch.object(retcon, 'read_url', return_value=b'bad!') as fetch:
                with self.assertRaisesRegex(ValueError, 'Git blob mismatch'):
                    retcon.prepare(root / 'cache')
                self.assertEqual(fetch.call_count, 1)

    def test_lock_only_contains_retcon_sources_and_declared_license(self):
        lock = json.loads(retcon.LOCK.read_text())
        self.assertEqual((lock['name'], lock['version'], lock['license']), ('Retcon', '0.2.0', 'MIT'))
        self.assertEqual(lock['repository'], 'robocode-dev/rumble-bots')
        self.assertEqual(lock['sourcePath'], 'bots/java/Retcon')
        self.assertEqual(len(lock['commit']), 40)
        self.assertEqual(len(lock['sourceSHA256']), 64)
        self.assertEqual(len({row['path'] for row in lock['files']}), len(lock['files']))
        self.assertTrue(all(row['path'].endswith(('.java', '.sh', '.cmd', '.json')) for row in lock['files']))


if __name__ == '__main__':
    unittest.main()

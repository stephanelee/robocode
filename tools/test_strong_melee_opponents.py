import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import strong_melee_opponents as native


class NativeOpponentTests(unittest.TestCase):
    def test_lock_and_dependency_closure_are_exact_and_runtime_only(self):
        lock=native.manifest();self.assertEqual(110,len(lock['files']));self.assertEqual(12,len(lock['runtime']['wheels']))
        canonical=''.join(e['path']+'\0'+e['sha256']+'\n' for e in sorted(lock['files'],key=lambda e:e['path']))
        self.assertEqual(lock['sourceSHA256'],hashlib.sha256(canonical.encode()).hexdigest())
        self.assertFalse(any('.env' in e['path'] or 'tomcat' in e['path'].lower() for e in lock['files']))
        self.assertEqual({'Adaptive Prime','Diamond Port'},{b['name'] for b in lock['opponents']})
        self.assertTrue(all(len(w['sha256'])==64 and w['filename'].endswith('.whl') for w in lock['runtime']['wheels']))

    def test_fresh_shared_core_license_and_clean_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';(source/'bots/bot_core').mkdir(parents=True)
            (source/'bots/bot_core/module.py').write_bytes(b'pass\n');(source/'LICENSE').write_bytes(b'notice\n')
            lock=dict(opponents=[dict(sourcePath='bots/adaptive-prime',descriptor='adaptive-prime.json')],sharedSourcePath='bots/bot_core',files=[dict(path='bots/bot_core/module.py',sha256=hashlib.sha256(b'pass\n').hexdigest()),dict(path='LICENSE',sha256=hashlib.sha256(b'notice\n').hexdigest())])
            run=root/'game/Melee-train';env=native.stage(source,run,Path('/private/venv/bin/python'))
            self.assertEqual(str(Path('/private/venv/bin/python')),env['ROBOCODE_PYTHON_BIN']);self.assertFalse(Path(env['ROBOCODE_ENV_FILE']).exists())
            with patch.object(native,'manifest',return_value=lock):
                self.assertEqual(2,len(native.verify_staged(run,['adaptive-prime'])))
                (run/'opponents/bot_core/module.py').write_text('changed')
                with self.assertRaises(ValueError):native.verify_staged(run,['adaptive-prime'])
            self.assertEqual(dict(PATH='ok'),native.clean_environment(dict(PATH='ok',PYTHONPATH='bad',ROBOCODE_PYTHON_BIN='bad',AISURVIVOR_MODE='fixed',PIP_INDEX_URL='bad')))


if __name__=='__main__':unittest.main()

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import confirm_own_wave_collector_compat as compat
from test_legal_scan_dataset import manifest


def fixture():
    base = manifest()
    games = [dict(base['games'][2], gameId='fresh-r' + str(i), acquisitionId='capture-r' + str(i)) for i in range(3)] + [
        dict(base['games'][3], gameId='fresh-x' + str(i), acquisitionId='capture-x' + str(i)) for i in range(3)]
    base.update(games=games, provenance=dict(sourceCommit='a' * 40, engineVersion='1.4.0', **compat.NEW))
    registration = dict(schemaVersion=1, modelSha256=compat.MODEL_SHA, screenReportSha256=compat.SCREEN_SHA,
        oldCollector=compat.OLD.copy(), newCollector=compat.NEW.copy(), collectorSourceCommit='b' * 40,
        engineVersion='1.4.0', games=copy.deepcopy(games), excludedGameIds=sorted(compat.FAILED_IDS),
        excludedAcquisitionIds=sorted(compat.FAILED_ACQUISITIONS))
    model = dict(captureProvenance=compat.OLD.copy(), priorGameIds=['spent'], priorAcquisitionIds=['spent-capture'])
    return registration, base, model


class CollectorCompatTests(unittest.TestCase):
    def test_only_explicit_pair_and_exact_manifest_is_allowed(self):
        registration, data, model = fixture()
        compat.validate_registration(registration, data, model, 'a' * 40)
        for field in ('botJarSha256', 'collectorVersion'):
            changed = copy.deepcopy(data)
            changed['provenance'][field] = 'wrong'
            with self.assertRaises(ValueError):
                compat.validate_registration(registration, changed, model, 'a' * 40)
        changed = copy.deepcopy(data)
        changed['games'][0]['aisSeed'] = 1
        with self.assertRaises(ValueError):
            compat.validate_registration(registration, changed, model, 'a' * 40)
        with self.assertRaises(ValueError):
            compat.validate_registration(registration, data, model, 'c' * 40)

    def test_spent_and_failed_identity_and_missing_exclusion_rejected(self):
        for field, identity in (('gameId', 'spent'), ('gameId', 'w3-retcon-01'), ('acquisitionId', 'spent-capture'),
                                ('acquisitionId', 'own-wave-v3-w3-target-01')):
            registration, data, model = fixture()
            data['games'][0][field] = identity
            registration['games'] = copy.deepcopy(data['games'])
            with self.assertRaises(ValueError):
                compat.validate_registration(registration, data, model, 'a' * 40)
        registration, data, model = fixture()
        registration['excludedGameIds'] = []
        with self.assertRaises(ValueError):
            compat.validate_registration(registration, data, model, 'a' * 40)

    def test_frozen_file_changes_fail_before_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            model, screen = Path(directory) / 'model', Path(directory) / 'screen'
            model.write_text('{}')
            screen.write_text('{}')
            with self.assertRaises(ValueError):
                compat.validate_frozen(model, screen, Path(directory))

    def test_all_frozen_dependency_and_report_hashes_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dependency = root / 'helper.py'
            dependency.write_text('unchanged source')
            source_sha = compat.sha(dependency)
            model = dict(manifestSha256='manifest', eventsSha256='events', contextToolSha256=source_sha)
            model_path, screen_path = root / 'model.json', root / 'screen.json'
            model_path.write_text(json.dumps(model))
            model_sha = compat.sha(model_path)
            screen = dict(model, stage='screen', modelSha256=model_sha, decision={'status': 'development-pass'})
            screen_path.write_text(json.dumps(screen))
            with patch.object(compat, 'MODEL_SHA', model_sha), patch.object(compat, 'SCREEN_SHA', compat.sha(screen_path)), \
                    patch.object(compat, 'SOURCE_PINS', {'contextToolSha256': ('helper.py', source_sha)}):
                self.assertEqual(compat.validate_frozen(model_path, screen_path, root)[0], model)
                dependency.write_text('changed source')
                with self.assertRaises(ValueError):
                    compat.validate_frozen(model_path, screen_path, root)

    def test_registration_reads_immutable_git_blob_not_worktree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            path = root / 'registration.json'
            raw = b'{"frozen":true}\n'
            path.write_bytes(raw)
            subprocess.run(['git', '-C', str(root), 'add', 'registration.json'], check=True)
            subprocess.run(['git', '-C', str(root), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                            'commit', '-qm', 'fixture'], check=True)
            commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
            path.write_text('{"frozen":false}')
            result, digest = compat.registered(commit, 'registration.json', root)
            # Git may normalize CRLF; this fixture is committed as the same LF bytes.
            self.assertEqual(result, {'frozen': True})
            self.assertEqual(digest, hashlib.sha256(raw).hexdigest())
            with self.assertRaises(ValueError):
                compat.registered('main', 'registration.json', root)


if __name__ == '__main__':
    unittest.main()

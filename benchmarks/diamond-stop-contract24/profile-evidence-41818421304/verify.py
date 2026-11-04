"""Reassemble and verify original zero-case evidence only; never evaluate cases."""
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    expected = manifest['files']
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()}
    if actual != set(expected) | {'manifest.json'}:
        raise ValueError('Evidence file inventory differs')
    for name, value in expected.items():
        raw = (ROOT / name).read_bytes()
        if len(raw) != value['bytes'] or sha(raw) != value['sha256']:
            raise ValueError('Evidence bytes differ: ' + name)
    frozen_path = ROOT / manifest['freeze']['path']
    frozen_raw = frozen_path.read_bytes()
    if len(frozen_raw) != manifest['freeze']['bytes'] or sha(frozen_raw) != manifest['freeze']['sha256']:
        raise ValueError('Write-once freeze bytes differ')
    original_profile = (ROOT / 'original-files/build/diamond-stop24/profile.json').read_bytes()
    if sha(original_profile) != manifest['originalProfileSHA256']:
        raise ValueError('Original reviewed profile differs')
    profile = json.loads(original_profile)
    if json.loads(frozen_raw) != dict(schemaVersion=1, profile=profile):
        raise ValueError('Freeze does not embed the complete original profile')
    repository = ROOT.parents[2]
    for name, value in profile['tooling'].items():
        if sha((repository / name).read_bytes()) != value:
            raise ValueError('Reviewed tooling changed: ' + name)
    archive = b''.join((ROOT / name).read_bytes() for name in manifest['originalZIP']['parts'])
    if len(archive) != manifest['originalZIP']['bytes'] or sha(archive) != manifest['originalZIP']['sha256']:
        raise ValueError('Reassembled original ZIP differs')
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        names = [item.filename for item in zipped.infolist() if not item.is_dir()]
        if len(names) != 29 or len(set(names)) != 29 or sorted(names) != manifest['originalArtifactFiles']:
            raise ValueError('Expected exactly29 original regular artifact entries')
        for name in names:
            value = manifest['originalArtifacts'][name]
            restored = b''.join((ROOT / part).read_bytes() for part in value['parts'])
            if len(restored) != value['bytes'] or sha(restored) != value['sha256'] or zipped.read(name) != restored:
                raise ValueError('Original extraction differs: ' + name)
    print('PASS: exact ZIP/all29 originals/evidence/freeze/30 tooling files verified; zero cases evaluated')


if __name__ == '__main__':
    main()

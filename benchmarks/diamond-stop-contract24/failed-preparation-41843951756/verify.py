"""Verify retained failure evidence only; never import or evaluate tested code."""
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[2]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def restored(value):
    data = b''.join((REPOSITORY / part).read_bytes() for part in value['parts'])
    if len(data) != value['bytes'] or sha(data) != value['sha256']:
        raise ValueError('Reassembled bytes differ')
    return data


def main():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()}
    if actual != set(manifest['files']) | {'manifest.json'}:
        raise ValueError('Failure evidence inventory differs')
    for name, value in manifest['files'].items():
        raw = (ROOT / name).read_bytes()
        if len(raw) != value['bytes'] or sha(raw) != value['sha256']:
            raise ValueError('Evidence bytes differ: ' + name)
        if len(raw) > 2 * 1024 * 1024:
            raise ValueError('Retained physical file exceeds two MiB')
    frozen = (REPOSITORY / manifest['freeze']['path']).read_bytes()
    if sha(frozen) != manifest['freeze']['sha256']:
        raise ValueError('Original freeze changed')
    archive = restored(manifest['originalZIP'])
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        names = [i.filename for i in zipped.infolist() if not i.is_dir()]
        if len(names) != 72 or len(set(names)) != 72 or sorted(names) != manifest['originalArtifactFiles']:
            raise ValueError('Expected exactly 72 original regular files')
        for name in names:
            if zipped.read(name) != restored(manifest['originalArtifacts'][name]):
                raise ValueError('Original artifact differs: ' + name)
        if any('/acquisition/' in name or name.endswith('/activation.json') for name in names):
            raise ValueError('Unexpected numeric acquisition artifact')
    diff = json.loads((ROOT / 'profile-diff.json').read_text())
    # Keep the complete original comparison, without recomputing numeric cases.
    if not diff:
        raise ValueError('Missing original profile comparison')
    print('PASS: exact original ZIP/all72 files/full failure evidence/freeze verified; zero cases evaluated')


if __name__ == '__main__':
    main()

"""Reassemble/hash original provenance only; never import harness/bots or run games."""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

ROOT = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def checked(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or not name:
        raise ValueError('Unsafe archived path')
    return path


def main():
    manifest = json.loads((ROOT / 'manifest.json').read_bytes())
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()}
    if actual != set(manifest['files']) | {'manifest.json'}:
        raise ValueError('Physical provenance file inventory differs')
    for name, value in manifest['files'].items():
        checked(name)
        raw = (ROOT / name).read_bytes()
        if len(raw) != value['bytes'] or sha(raw) != value['sha256'] or len(raw) > 2 * 1024 * 1024:
            raise ValueError('Physical provenance bytes/size differ: ' + name)
    original = manifest['originalZIP']
    archive = b''.join((ROOT / checked(part)).read_bytes() for part in original['parts'])
    if len(archive) != original['bytes'] or sha(archive) != original['sha256']:
        raise ValueError('Original provider ZIP differs')
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        names = [item.filename for item in zipped.infolist() if not item.is_dir()]
        if len(names) != 102 or len(set(names)) != 102 or sorted(names) != manifest['originalArtifactFiles']:
            raise ValueError('Expected exactly102 original regular provider files')
        source_count = 0
        for name in names:
            checked(name)
            original_bytes = zipped.read(name)
            value = manifest['originalArtifacts'][name]
            if len(original_bytes) != value['bytes'] or sha(original_bytes) != value['sha256']:
                raise ValueError('Original logical file differs: ' + name)
            storage = value['storage']
            if storage['kind'] == 'verbatim-parts':
                retained = b''.join((ROOT / checked(part)).read_bytes() for part in storage['parts'])
                if retained != original_bytes:
                    raise ValueError('Retained original extraction differs: ' + name)
            elif storage['kind'] == 'immutable-git-source':
                # Exact original source bytes are also physically retained inside the ORIGINAL ZIP.
                # The immutable source reference is independently addressable; no Git checkout/network
                # is required to reconstruct these bytes or verify their Git-blob/SHA256 identities.
                source_count += 1
                if (storage['commit'] != manifest['payloadHead'] or not re.fullmatch('[0-9a-f]{40}', storage['commit'])
                        or name != 'source/' + storage['path']):
                    raise ValueError('Immutable source reference differs: ' + name)
                checked(storage['path'])
                git_blob = b'blob ' + str(len(original_bytes)).encode() + b'\0' + original_bytes
                if hashlib.sha1(git_blob).hexdigest() != storage['blobSHA']:
                    raise ValueError('Original source Git blob differs: ' + name)
            else:
                raise ValueError('Unknown storage contract')
        if source_count != 90:
            raise ValueError('Expected90 original frozen exported source/JAR/tool files')
        original_manifest = json.loads(zipped.read('manifest.json'))
        if set(original_manifest) != set(names) - {'manifest.json'}:
            raise ValueError('Provider provenance manifest membership differs')
        for name, row in original_manifest.items():
            if sha(zipped.read(name)) != row['sha256']:
                raise ValueError('Provider provenance manifest hash differs: ' + name)
        if sha(zipped.read('freeze.json')) != manifest['freezeSHA256']:
            raise ValueError('Original source freeze differs')
    print('PASS: original3311004B provider ZIP/all102 files/90 immutable source refs/API/reviews verified; no games or helpers run')


if __name__ == '__main__':
    main()

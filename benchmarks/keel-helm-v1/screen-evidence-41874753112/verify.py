"""Lossless byte/hash verification only: no harness, bot, engine or outcome reduction."""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[2]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def checked(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or '..' in path.parts or '\\' in name:
        raise ValueError('Unsafe retained archive path')
    return path


def main():
    manifest = json.loads((ROOT / 'manifest.json').read_bytes())
    expected_files = manifest['files']
    actual_files = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()}
    if actual_files != set(expected_files) | {'manifest.json'}:
        raise ValueError('Physical final-evidence file inventory differs')
    for name, value in expected_files.items():
        raw = (ROOT / checked(name)).read_bytes()
        if len(raw) != value['bytes'] or sha(raw) != value['sha256']:
            raise ValueError('Retained physical bytes differ: ' + name)
    value = manifest['originalZIP']
    chunks = [(ROOT / checked(name)).read_bytes() for name in value['parts']]
    if any(len(chunk) > 2 * 1024 * 1024 for chunk in chunks):
        raise ValueError('Original provider ZIP part exceeds2MiB')
    original_zip = b''.join(chunks)
    if len(original_zip) != value['bytes'] or sha(original_zip) != value['sha256']:
        raise ValueError('Original provider ZIP bytes differ')
    freeze = manifest['sourceFreeze']
    if sha((REPOSITORY / checked(freeze['path'])).read_bytes()) != freeze['sha256']:
        raise ValueError('Original source freeze changed')
    references = {}
    with zipfile.ZipFile(io.BytesIO(original_zip)) as zipped:
        names = [item.filename for item in zipped.infolist() if not item.is_dir()]
        if len(names) != 1349 or len(set(names)) != 1349 or sorted(names) != manifest['originalArtifactFiles']:
            raise ValueError('Expected exactly1349 original regular provider files')
        for name in names:
            checked(name)
            raw = zipped.read(name)
            value = manifest['originalArtifacts'][name]
            if len(raw) != value['bytes'] or sha(raw) != value['sha256']:
                raise ValueError('Original logical bytes differ: ' + name)
            storage = value['storage']
            if storage['kind'] == 'direct-file':
                if (ROOT / checked(storage['path'])).read_bytes() != raw:
                    raise ValueError('Direct retained original differs: ' + name)
            elif storage['kind'] == 'retained-pregame-provenance-entry':
                path = storage['manifest']
                if path not in references:
                    data = (REPOSITORY / checked(path)).read_bytes()
                    if sha(data) != storage['manifestSHA256']:
                        raise ValueError('Immutable pregame manifest differs')
                    references[path] = json.loads(data)
                prior = references[path]['originalArtifacts'][storage['entry']]
                if prior['bytes'] != len(raw) or prior['sha256'] != sha(raw):
                    raise ValueError('Original pregame provenance reference differs: ' + name)
            elif storage['kind'] == 'original-provider-zip-entry':
                if storage['entry'] != name:
                    raise ValueError('Original staged-work ZIP reference differs')
                # Raw bytes remain physically inside this exact original ZIP; no Git/network needed.
            else:
                raise ValueError('Unknown retained storage contract')
    whole = manifest['wholeRecordings']
    if len(whole) != 14 or len({row['path'] for row in whole}) != 14:
        raise ValueError('Expected14 whole raw recordings')
    for row in whole:
        p = ROOT / checked(row['path'])
        if not p.is_file() or len(p.read_bytes()) != row['bytes'] or sha(p.read_bytes()) != row['sha256']:
            raise ValueError('Whole original recording differs')
    raw_decision = (ROOT / 'decision.json').read_bytes()
    if sha(raw_decision) != manifest['rawDecisionSHA256']:
        raise ValueError('Original raw decision bytes differ')
    decision = json.loads(raw_decision)
    if decision['stage'] != 'screen' or decision['headSHA'] != manifest['payloadHead']:
        raise ValueError('Original decision stage/head changed')
    # This validates original identifiers/recording hashes, not metrics or eligibility.
    if {row['recordingSHA256'] for row in decision['games']} != {row['sha256'] for row in whole}:
        raise ValueError('Original decision/whole-recording identities differ')
    print('PASS: exact117899908B provider ZIP/all1349 files/14WHOLE recordings/rawdecision/freeze verified; no outcome calculation or games')


if __name__ == '__main__':
    main()

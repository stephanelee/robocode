"""Restore the byte-exact original acquisition files; never execute a diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent / 'raw'
    manifest = json.loads((source / 'manifest.json').read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    for name, entry in manifest['files'].items():
        if Path(name).name != name:
            raise ValueError('Non-local original filename')
        digest = hashlib.sha256()
        size = 0
        with (args.output / name).open('xb') as fd:
            for part in entry['parts']:
                if Path(part['path']).name != part['path']:
                    raise ValueError('Non-local part filename')
                data = (source / part['path']).read_bytes()
                if len(data) != part['size'] or hashlib.sha256(data).hexdigest() != part['sha256']:
                    raise ValueError('Part hash/size mismatch: ' + part['path'])
                fd.write(data)
                digest.update(data)
                size += len(data)
        if size != entry['size'] or digest.hexdigest() != entry['sha256']:
            raise ValueError('Original hash/size mismatch: ' + name)
    print('Restored and verified all', len(manifest['files']), 'original acquisition files; no cases executed')


if __name__ == '__main__':
    main()

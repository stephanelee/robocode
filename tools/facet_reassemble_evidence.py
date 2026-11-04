#!/usr/bin/env python3
"""Reassemble immutable game archives from verified parts; no model use or battles."""
import argparse
import hashlib
import json
from pathlib import Path


def main(args):
    root=args.root.resolve();manifest=json.loads((root/'raw-manifest.json').read_text())
    args.output.mkdir(parents=True,exist_ok=True)
    for game,row in manifest['games'].items():
        target=args.output/(game+'.tar.gz')
        if target.exists():raise ValueError('Refuse output overwrite: '+str(target))
        data=bytearray()
        for part in row['parts']:
            path=(root/part['path']).resolve()
            if not path.is_relative_to(root):raise ValueError('Part outside evidence root')
            value=path.read_bytes()
            if len(value)!=part['bytes'] or hashlib.sha256(value).hexdigest()!=part['sha256']:
                raise ValueError('Part identity mismatch: '+str(path))
            data.extend(value)
        if len(data)!=row['archiveBytes'] or hashlib.sha256(data).hexdigest()!=row['archiveSHA256']:
            raise ValueError('Archive identity mismatch: '+game)
        target.write_bytes(data)
        print(game,row['archiveSHA256'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())

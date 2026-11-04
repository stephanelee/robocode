#!/usr/bin/env python3
"""Compare every JAR file entry, ignoring ZIP ordering and timestamps only."""
import argparse
import hashlib
from pathlib import Path
import sys
import zipfile


def entries(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as jar:
        result = {}
        for entry in jar.infolist():
            if entry.is_dir():
                continue
            if entry.filename in result:
                raise ValueError(f"Duplicate entry in {path}: {entry.filename}")
            result[entry.filename] = hashlib.sha256(jar.read(entry)).digest()
        if not result:
            raise ValueError(f"JAR has no file entries: {path}")
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checked_in", type=Path)
    parser.add_argument("built", type=Path)
    args = parser.parse_args()
    try:
        expected, actual = entries(args.checked_in), entries(args.built)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"JAR verification failed: {error}", file=sys.stderr)
        return 1
    missing = sorted(expected.keys() - actual.keys())
    added = sorted(actual.keys() - expected.keys())
    changed = sorted(name for name in expected.keys() & actual.keys()
                     if expected[name] != actual[name])
    if missing or added or changed:
        for label, names in (("Only in checked-in JAR", missing),
                             ("Only in source build", added), ("Different contents", changed)):
            for name in names:
                print(f"{label}: {name}", file=sys.stderr)
        print("Rebuild and commit AISurvivor/AISurvivor.jar with the source changes.", file=sys.stderr)
        return 1
    print(f"Verified {len(expected)} matching JAR file entries (ZIP timestamps/order ignored).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Reproduce the immutable v6 model using only the designated training recording."""
import hashlib
import argparse
import math
from pathlib import Path

from meridian_motion_memory import TRAIN, read

DEST = Path(__file__).resolve().parents[1] / "experiments/Meridian/src/main/resources/crossing-prior.csv"
TRAIN_SHA = "ffec6865139914d0cd0bb8eb196bd48e5f0b65669d63cec40ccb97b66adcdbe0"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    rows, source = read(TRAIN)
    if source["sha256"] != TRAIN_SHA:
        raise ValueError("Unexpected training recording")
    lines = []
    for row in rows:
        if abs(row["offset"]) <= 1.25:
            # Undo the diagnostic's feature scaling for the bot's native Context representation.
            features = [value * scale for value, scale in zip(row["features"], (3, 4, .35, 15))]
            lines.append(",".join(map(repr, [*features, row["offset"]])))
    if len(lines) != 206:
        raise ValueError("Unexpected bank size")
    data = ("\n".join(lines) + "\n").encode("ascii")
    if args.verify:
        actual = [[float(v) for v in line.split(",")] for line in data.decode("ascii").splitlines()]
        expected = [[float(v) for v in line.split(",")] for line in DEST.read_text().splitlines()]
        if len(actual) != len(expected) or any(len(row) != 5 for row in expected):
            raise ValueError("Bank shape mismatch")
        if any(not math.isfinite(y) or abs(x-y) > 1e-12 for a, b in zip(actual, expected) for x, y in zip(a, b)):
            raise ValueError("Bank differs beyond the registered libm tolerance")
    else:
        DEST.write_bytes(data)
    print(len(lines), hashlib.sha256(data).hexdigest())


if __name__ == "__main__":
    main()

"""Reproduce the immutable v6 model using only the designated training recording."""
import hashlib
from pathlib import Path

from meridian_motion_memory import TRAIN, read

DEST = Path(__file__).resolve().parents[1] / "experiments/Meridian/src/main/resources/crossing-prior.csv"
TRAIN_SHA = "ffec6865139914d0cd0bb8eb196bd48e5f0b65669d63cec40ccb97b66adcdbe0"


def main():
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
    DEST.write_bytes(data)
    print(len(lines), hashlib.sha256(data).hexdigest())


if __name__ == "__main__":
    main()

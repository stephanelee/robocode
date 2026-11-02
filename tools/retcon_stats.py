"""Capture the pinned Retcon StatsWriter export without changing opponent source."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import uuid

SOURCE_COMMIT = "28981c7a469ac064194b41001cd7183ec5ecf189"
SOURCE_SHA256 = "5f7dfaaedefc39bd93d86440a2996342583d57b75b861942dc1e1899b7164bb2"
MAX_BYTES = 2_000_000
FILE_PATTERN = re.compile(r"Retcon-(\d+)-(-?\d+)\.json")


def java_temp_directory(java):
    result = subprocess.run([java, "-XshowSettings:properties", "-version"],
                            capture_output=True, text=True, timeout=15, check=True)
    match = re.search(r"^\s*java\.io\.tmpdir\s*=\s*(.+)$", result.stderr, re.MULTILINE)
    if not match:
        raise ValueError("Java did not report java.io.tmpdir")
    path = Path(match.group(1).strip()).absolute()
    if not path.is_dir():
        raise ValueError("Reported Java temporary directory does not exist")
    return path


def integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"Retcon stats {name} must be a non-negative integer")
    return value


def reject_constant(value):
    raise ValueError(f"Non-finite JSON token in Retcon export: {value}")


def validate(data, rounds, expected_id):
    if not isinstance(data, dict) or data.get("bot") != "Retcon":
        raise ValueError("Retcon stats bot identity mismatch")
    if type(data.get("botId")) is not int or data["botId"] != expected_id:
        raise ValueError("Retcon stats botId differs from official results")
    recorded = integer(data.get("roundsRecorded"), "roundsRecorded")
    if recorded > rounds:
        raise ValueError("Retcon stats exceed the requested round budget")
    counters = ("turnTimeoutMicros", "computeBudgetMicros", "turns", "computeP50Micros", "computeP99Micros",
                "computeMaxMicros", "overBudgetTurns", "skippedTurns")
    for key in counters:
        integer(data.get(key), key)
    skips = data.get("skippedPerRound")
    if not isinstance(skips, list) or len(skips) != rounds + 1:
        raise ValueError("Retcon skippedPerRound layout differs from the pinned writer")
    for value in skips:
        integer(value, "skippedPerRound")
    if sum(skips) != data["skippedTurns"]:
        raise ValueError("Retcon skipped-turn counters disagree")
    for key in ("skippedTurnList", "computeHistogram10us"):
        if not isinstance(data.get(key), str):
            raise ValueError(f"Missing Retcon diagnostic string {key}")
    groups = {
        "timing": (("firesChecked", "firesNotCreated", "createdTurnMatches", "originMatches",
                    "originMatchesAfterFirstMove", "headingMatches", "stillHitsChecked", "stillHitMatches",
                    "hitsOnMovingTargets"), "mismatches"),
        "waves": (("firesDetected", "unanchoredShots", "hitsTaken", "hitsOnWave", "wavePowerCorrected",
                   "waveHitsExact", "waveHitsOffByOneStep", "waveHitsOtherResidual"), "waveMismatches"),
        "movement": (("moveTurnsChecked", "moveMatches", "moveWallHitsChecked", "moveNotChecked"), "moveMismatches"),
        "gun": (("bulletsFired", "bulletHits"), "virtualGuns"),
    }
    for name, (keys, text_key) in groups.items():
        group = data.get(name)
        if not isinstance(group, dict) or not isinstance(group.get(text_key), str):
            raise ValueError(f"Retcon stats missing {name} fields")
        for key in keys:
            integer(group.get(key), f"{name}.{key}")
    gun = data["gun"]
    if gun["bulletHits"] > gun["bulletsFired"]:
        raise ValueError("Retcon actual hits exceed fired bullets")
    if data["waves"]["hitsOnWave"] > data["waves"]["hitsTaken"]:
        raise ValueError("Retcon matched-wave hits exceed hits taken")
    if data["movement"]["moveMatches"] > data["movement"]["moveTurnsChecked"]:
        raise ValueError("Retcon movement matches exceed checked turns")
    virtual = []
    seen = set()
    for token in gun["virtualGuns"].split():
        match = re.fullmatch(r"([^:\s]+):(\d+)/(\d+)", token)
        if not match or match[1] in seen:
            raise ValueError("Malformed or duplicate Retcon virtual-gun counters")
        name, hits, shots = match[1], int(match[2]), int(match[3])
        if hits > shots:
            raise ValueError("Retcon virtual hits exceed completed virtual shots")
        seen.add(name)
        virtual.append(dict(model=name, hits=hits, shots=shots, hitRate=hits / shots if shots else None))
    return dict(actualGun=dict(shots=gun["bulletsFired"], hits=gun["bulletHits"],
                               hitRate=gun["bulletHits"] / gun["bulletsFired"] if gun["bulletsFired"] else None),
                virtualGuns=virtual)


class Capture:
    """Own one direct Java-temp child; never recursively remove arbitrary paths."""
    def __init__(self, java_temp):
        self.root = Path(java_temp).absolute()
        self.path = Path(tempfile.mkdtemp(prefix="aisurvivor-retcon-", dir=self.root))
        self.token = uuid.uuid4().hex
        self.marker = self.path / ".capture-owner"
        self.marker.write_text(self.token, encoding="ascii")
        self.closed = False

    def assert_owned(self):
        if self.closed or self.path.is_symlink() or self.path.parent.resolve() != self.root.resolve():
            raise ValueError("Retcon capture is not an owned Java-temp child")
        if not self.marker.is_file() or self.marker.is_symlink() or self.marker.read_text(encoding="ascii") != self.token:
            raise ValueError("Retcon capture ownership marker mismatch")

    def collect(self, destination, rounds, expected_id, require=False):
        self.assert_owned()
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        summary = dict(schemaVersion=1, status="missing", expectedRounds=rounds, bot="Retcon", version="0.2.0", sourceCommit=SOURCE_COMMIT,
                       sourceSHA256=SOURCE_SHA256, rawArtifacts=[])
        files = []
        error = None
        for source in self.path.iterdir():
            if source.name == self.marker.name:
                continue
            if source.is_symlink() or not source.is_file() or source.stat().st_size > MAX_BYTES:
                error = "Unsafe or oversized Retcon export entry"
                continue
            if not (FILE_PATTERN.fullmatch(source.name) or re.fullmatch(r"Retcon-\d+--?\d+\.json\.tmp", source.name)):
                error = "Unexpected Retcon export filename"
                continue
            target = destination / source.name
            shutil.copyfile(source, target)
            summary["rawArtifacts"].append(dict(path=source.name, sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
            if source.suffix == ".json":
                files.append(source)
        try:
            if error:
                raise ValueError(error)
            if len(files) > 1:
                raise ValueError("Multiple Retcon game exports in one isolated capture")
            if files:
                match = FILE_PATTERN.fullmatch(files[0].name)
                data = json.loads(files[0].read_text(encoding="utf-8"), parse_constant=reject_constant)
                if int(match[1]) != expected_id:
                    raise ValueError("Retcon export filename botId differs from official results")
                summary.update(validate(data, rounds, expected_id))
                summary.update(data=data, status="complete" if data["roundsRecorded"] == rounds else "partial")
        except (ValueError, OSError) as exception:
            summary.update(status="invalid", error=str(exception))
        (destination / "capture.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if require and summary["status"] != "complete":
            raise ValueError(f"Required Retcon stats are {summary['status']}: {summary.get('error', 'round coverage incomplete')}")
        return summary

    def close(self):
        self.assert_owned()
        entries = list(self.path.iterdir())
        if any(path.is_dir() and not path.is_symlink() for path in entries):
            raise ValueError("Refusing cleanup of unexpected Retcon capture subdirectories")
        for path in entries:
            path.unlink()
        self.path.rmdir()
        self.closed = True

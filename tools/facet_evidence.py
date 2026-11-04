#!/usr/bin/env python3
"""Record Facet artifact identities without loading opponents or model data."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]
API_SHA256 = "ab65c4d5cec1808adeb71375adae6d15341ae250def89a6c10fc9da879de0752"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_digest(files):
    encoded = "".join(f"{name}\0{value}\n" for name, value in sorted(files.items()))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def record(bot, api, java):
    if digest(api) != API_SHA256:
        raise ValueError("Expected the frozen official Bot API 1.4.0 artifact")
    files = {}
    for path in sorted(bot.rglob("*")):
        relative = path.relative_to(bot)
        if not path.is_file() or any(part in {"build", ".gradle", "evidence"} for part in relative.parts):
            continue
        if relative.parts[0] == "src" or path.suffix in {
            ".java", ".properties", ".gradle", ".json", ".sh", ".cmd", ".py"
        }:
            files[relative.as_posix()] = digest(path)
    jar = bot / "Facet.jar"
    # The dependency is the official shaded API, not a bundle of opponent code.
    with zipfile.ZipFile(api) as official, zipfile.ZipFile(jar) as candidate:
        names = candidate.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate JAR entries")
        checked = 0
        for entry in official.infolist():
            if entry.is_dir() or entry.filename == "META-INF/MANIFEST.MF":
                continue
            if candidate.read(entry.filename) != official.read(entry):
                raise ValueError(f"Bundled API differs: {entry.filename}")
            checked += 1
    runtime = subprocess.run([str(java), "-version"], capture_output=True, text=True, check=True)
    return {
        "schemaVersion": 1,
        "purpose": "artifact provenance, not competitive evidence",
        "recorderSHA256": digest(Path(__file__).resolve()),
        "bot": "Facet",
        "sourceSHA256Algorithm": "SHA256(sorted relative path + NUL + file SHA256 + LF)",
        "sourceFiles": files,
        "sourceSHA256": tree_digest(files),
        "jarSHA256": digest(jar),
        "apiVersion": "1.4.0",
        "apiSHA256": digest(api),
        "bundledOfficialAPIEntriesVerified": checked,
        "configSHA256": digest(bot / "src/main/resources/facet.properties"),
        "opponentLockSHA256": {
            name: digest(ROOT / "tools/opponents" / name)
            for name in ["retcon.lock.json", "strong-melee.lock.json"]
        },
        "java": runtime.stdout + runtime.stderr,
        "os": platform.platform(),
        "architecture": platform.machine(),
        "performanceResults": "not recorded by this tool",
        "premeasurementApproval": "not established by this tool",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bot", type=Path, default=ROOT / "experiments/Facet")
    parser.add_argument("--api", required=True, type=Path)
    parser.add_argument("--java", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = record(args.bot.resolve(), args.api.resolve(), args.java.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: result[key] for key in ("sourceSHA256", "jarSHA256", "configSHA256")}))


if __name__ == "__main__":
    main()

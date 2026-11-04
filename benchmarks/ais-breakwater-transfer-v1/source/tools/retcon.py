"""Fetch only the pinned official Retcon source tree; preserve upstream files unchanged."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import urllib.request

LOCK = Path(__file__).resolve().parent / "opponents" / "retcon.lock.json"


def source_hash(directory):
    # Official rumble-bots scripts/validate_bot.py tree_hash algorithm.
    digest = hashlib.sha256()
    for path in sorted((p for p in directory.rglob("*") if p.is_file()),
                       key=lambda p: p.relative_to(directory).as_posix()):
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def checked_relative(name):
    path = PurePosixPath(name)
    if not name or not path.parts or path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe opponent source path: {name}")
    return Path(*path.parts)


def read_url(url, limit):
    request = urllib.request.Request(url, headers={"User-Agent": "AISurvivor-Retcon-benchmark"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError("Opponent download exceeds pinned size limit")
    return data


def prepare(cache):
    manifest = json.loads(LOCK.read_text(encoding="utf-8"))
    source = cache / "retcon" / manifest["commit"] / "Retcon"
    source.mkdir(parents=True, exist_ok=True)

    def fetch(entry):
        target = source / checked_relative(entry["path"])
        if target.is_symlink() or not target.resolve().is_relative_to(source.resolve()):
            raise ValueError("Opponent cache must not contain symlinks")
        data = target.read_bytes() if target.exists() else b""
        def matches(content):
            blob = b"blob " + str(len(content)).encode("ascii") + b"\0" + content
            return len(content) == entry["size"] and hashlib.sha1(blob).hexdigest() == entry["gitBlobSHA"]
        if not matches(data):
            url = f"https://raw.githubusercontent.com/{manifest['repository']}/{manifest['commit']}/{manifest['sourcePath']}/{entry['path']}"
            data = read_url(url, entry["size"])
            if not matches(data):
                raise ValueError(f"Pinned Git blob mismatch: {entry['path']}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(fetch, manifest["files"]))
    # The SHA-256 source tree is the security boundary; SHA-1 blobs are cache checks.
    if source_hash(source) != manifest["sourceSHA256"]:
        raise ValueError("Retcon source tree SHA256 differs from the official catalog pin")
    descriptor = json.loads((source / "Retcon.json").read_text(encoding="utf-8"))
    if (descriptor["name"], descriptor["version"], descriptor["license"]) != ("Retcon", "0.2.0", "MIT"):
        raise ValueError("Retcon identity/license differs from the pinned catalog")
    if "1v1" not in descriptor["gameTypes"]:
        raise ValueError("Retcon does not advertise 1v1 support")
    (source / "Retcon.sh").chmod(0o755)

    api_cache = cache / "retcon-api"
    api_cache.mkdir(parents=True, exist_ok=True)
    dependency = manifest["botApi"]
    jar = api_cache / f"robocode-tankroyale-bot-api-{dependency['version']}.jar"
    data = jar.read_bytes() if jar.exists() else b""
    if hashlib.sha256(data).hexdigest() != dependency["sha256"]:
        data = read_url(dependency["url"], 5_000_000)
        if hashlib.sha256(data).hexdigest() != dependency["sha256"]:
            raise ValueError("Retcon Bot API SHA256 mismatch")
        jar.write_bytes(data)
    provenance = {key: manifest[key] for key in ("name", "version", "repository", "commit", "sourcePath", "sourceSHA256", "license", "authors", "botApi")}
    return source, api_cache, provenance

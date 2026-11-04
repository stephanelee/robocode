"""Acquire and hash the registered artifacts. No server or bot is started."""
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
import benchmark
import retcon
import strong_melee_opponents as native

ROOT = benchmark.ROOT
CACHE = ROOT / ".benchmark-cache"
OUT = ROOT / "build/meridian"

def main():
    source, python, py = native.prepare(CACHE, sys.executable)
    _, _, rc = retcon.prepare(CACHE)
    runner = "robocode-tankroyale-runner-1.4.0.jar"
    jar = benchmark.download(CACHE, runner, benchmark.ASSETS[runner])
    java = Path(os.environ["JAVA_HOME"])
    bot = ROOT / "experiments/Meridian"
    files = [*sorted((bot / "src").glob("*.java")), *sorted((bot / "bot").iterdir()), bot / "Meridian.jar",
             *sorted((ROOT / "AISurvivor").glob("*")), ROOT / "tools/opponents/strong-melee.lock.json",
             ROOT / "tools/opponents/retcon.lock.json"]
    api = list((Path.home()/".gradle/caches/modules-2/files-2.1/dev.robocode.tankroyale/robocode-tankroyale-bot-api/1.4.0").glob("*/*.jar"))
    if len(api)!=1: raise ValueError("Expected one combat API jar")
    hashes = {str(f.relative_to(ROOT)): benchmark.digest(f) for f in files if f.is_file()}
    data = dict(assets=hashes, runner=benchmark.digest(jar), botApi140=benchmark.digest(api[0]),
                java25={f:benchmark.digest(java/f) for f in ("bin/java","lib/modules","release","lib/server/libjvm.so")},
                buildJava=(OUT/"build-java.sha256").read_text(), java17Version=(OUT/"build-java.txt").read_text(),
                os=Path("/etc/os-release").read_text(), arch=platform.machine(), image=os.environ.get("ImageVersion"),
                pythonBinary=benchmark.digest(python.resolve()), python=py, retcon=rc)
    (OUT/"profile.json").write_text(json.dumps(data,indent=2,sort_keys=True)+"\n")
    print("Verified:",hashlib.sha256((OUT/"profile.json").read_bytes()).hexdigest(),flush=True)

if __name__=="__main__": main()

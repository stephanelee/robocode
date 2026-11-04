#!/usr/bin/env python3
"""Offline Java17 build/test/fat-JAR packaging. No fetches or engine launches."""
import hashlib, json, os, pathlib, platform, shutil, subprocess, sys, zipfile
ROOT = pathlib.Path(__file__).resolve().parent
API = pathlib.Path(os.environ.get("FACET_BUILD_API_JAR", ROOT / "lib/robocode-tankroyale-bot-api-1.4.0.jar"))
EXPECTED_API = "ab65c4d5cec1808adeb71375adae6d15341ae250def89a6c10fc9da879de0752"
os.environ.setdefault("TMPDIR", str(ROOT/"build/tmp"))
pathlib.Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def run(args, env=None):
    p = subprocess.run([str(x) for x in args], cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.stdout, end="")
    if p.returncode:
        raise RuntimeError(f"command failed ({p.returncode}): {args}\n{p.stdout}")
    return p.stdout
def payload():
    entries = {}
    with zipfile.ZipFile(API) as z:
        for name in z.namelist():
            upper = name.upper()
            if name.endswith("/") or upper == "META-INF/MANIFEST.MF" or upper.endswith((".SF", ".RSA", ".DSA")):
                continue
            entries[name] = z.read(name)
    for folder in [ROOT/"build/classes/main", ROOT/"src/main/resources"]:
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                entries[p.relative_to(folder).as_posix()] = p.read_bytes()
    entries["META-INF/MANIFEST.MF"] = b"Manifest-Version: 1.0\r\nMain-Class: Facet\r\nMulti-Release: true\r\n\r\n"
    return entries
def pack(path, entries):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, data in sorted(entries.items()):
            info = zipfile.ZipInfo(name, (1980,1,1,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            z.writestr(info, data, compresslevel=9)
def main():
    if sha(API) != EXPECTED_API:
        raise RuntimeError("official API pin mismatch")
    official = ROOT/"src/test/resources/energy-trace"
    report = json.loads((official/"trace.json").read_text())
    for name, digest in report["rawSHA256"].items():
        if sha(official/(name+".jsonl")) != digest:
            raise RuntimeError("official trace raw hash mismatch: "+name)
    java_home = pathlib.Path(os.environ.get("JAVA_HOME", "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"))
    java, javac = java_home/"bin/java", java_home/"bin/javac"
    runtime = run([java, "-version"])
    if 'version "17.' not in runtime:
        raise RuntimeError("Java17 required")
    for name in ["main", "test"]:
        folder = ROOT/"build/classes"/name
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        sources = sorted((ROOT/"src"/name/"java").glob("*.java"))
        cp = str(API) + (os.pathsep+str(ROOT/"build/classes/main") if name=="test" else "")
        run([javac, "--release", "17", "-encoding", "UTF-8", "-g:source,lines", "-cp", cp, "-d", folder, *sources])
    cp = os.pathsep.join(map(str,[ROOT/"build/classes/main",ROOT/"build/classes/test",ROOT/"src/main/resources",ROOT/"src/test/resources",API]))
    clean_env = {k:v for k,v in os.environ.items() if not k.startswith("FACET_")}
    tests = run([java, "-Xmx256m", "-cp", cp, "FacetSemanticTest"], clean_env)
    evidence = ROOT/"evidence"
    evidence.mkdir(exist_ok=True)
    (evidence/"semantic-tests.txt").write_text(tests)
    fixture_env = dict(clean_env)
    for key, name in [("FACET_TRACE_PATH", "FACET_fixture-trace.jsonl"), ("FACET_METRICS_PATH", "FACET_fixture-metrics.jsonl"), ("FACET_MODEL_PATH", "FACET_fixture-model.bin")]:
        path = evidence/name
        path.unlink(missing_ok=True)
        path.with_name(path.name+".partial").unlink(missing_ok=True)
        fixture_env[key] = str(path)
    fixture = run([java, "-Xmx256m", "-cp", cp, "FacetTelemetryFixture"], fixture_env)
    (evidence/"telemetry-fixture.txt").write_text(fixture)
    metrics = [json.loads(line) for line in (evidence/"FACET_fixture-metrics.jsonl").read_text().splitlines()]
    trace = [json.loads(line) for line in (evidence/"FACET_fixture-trace.jsonl").read_text().splitlines()]
    assert len(metrics)==1 and metrics[0]["shotsConfirmed"]==1 and metrics[0]["counters"]["shot.hit"]==1
    assert len(trace)<100000 and any(e["kind"]=="command.prepared" for e in trace)
    lifecycle_env = dict(clean_env)
    for key, name in [("FACET_TRACE_PATH", "FACET_lifecycle-trace.jsonl"), ("FACET_METRICS_PATH", "FACET_lifecycle-metrics.jsonl")]:
        path = evidence/name
        path.unlink(missing_ok=True)
        lifecycle_env[key] = str(path)
    lifecycle = run([java, "-Xmx256m", "-cp", cp, "LifecycleTelemetryFixture"], lifecycle_env)
    (evidence/"lifecycle-fixture.txt").write_text(lifecycle)
    rows = [json.loads(line) for line in (evidence/"FACET_lifecycle-metrics.jsonl").read_text().splitlines()]
    assert len(rows)==4 and len({(row["gameId"],row["round"]) for row in rows})==4
    assert [row["shotsConfirmed"] for row in rows]==[1,0,1,0]
    assert all(row["schema"]==2 and row["counters"].get("metrics.ioFailure",0)==0 for row in rows)
    follow_env = dict(clean_env)
    follow_path = evidence/"FACET_followup-trace.jsonl"
    follow_path.unlink(missing_ok=True)
    follow_env["FACET_TRACE_PATH"] = str(follow_path)
    follow = run([java, "-Xmx256m", "-cp", cp, "FollowupTelemetryFixture"], follow_env)
    (evidence/"followup-telemetry.txt").write_text(follow)
    follow_rows = [json.loads(line) for line in follow_path.read_text().splitlines()]
    confirmations = [r for r in follow_rows if r["kind"]=="fire.confirmed"]
    assert len(confirmations)==2 and all("pending=true" in r["value"] for r in confirmations)
    assert "gun=combinedmelee" in confirmations[0]["value"] and "population=2" in confirmations[0]["value"]
    assert "gun=recent" in confirmations[1]["value"] and "label=DISPLACEMENT" in confirmations[1]["value"]
    assert any(r["kind"]=="population.transition" and "from=2 to=1" in r["value"] for r in follow_rows)
    assert len(follow_rows)<100000 and not any(r["kind"]=="failure" for r in follow_rows)
    jar = ROOT/"Facet.jar"
    pack(jar, payload())
    second = ROOT/"build/Facet-rebuild.jar"
    pack(second, payload())
    if jar.read_bytes() != second.read_bytes():
        raise RuntimeError("deterministic packaging mismatch")
    smoke = run([java, "-Xmx256m", "-jar", jar, "--verify-package"])
    (evidence/"package-smoke.txt").write_text(smoke)
    run([sys.executable, ROOT/"parity.py", jar, second])
    source_paths = sorted(p for p in (ROOT/"src").rglob("*") if p.is_file()) + [ROOT/"build.gradle",ROOT/"settings.gradle",ROOT/"build.py",ROOT/"parity.py",ROOT/"Facet.json",ROOT/"Facet.sh"]
    source_lines = [sha(p)+"  "+p.relative_to(ROOT).as_posix() for p in source_paths]
    source_digest = hashlib.sha256(("\n".join(source_lines)+"\n").encode()).hexdigest()
    (evidence/"source-files.sha256").write_text("\n".join(source_lines)+"\n")
    (evidence/"manifest.sha256").write_text(f"{sha(jar)}  Facet.jar\n{sha(API)}  official API 1.4.0\n{source_digest}  source-files.sha256 content\n")
    metadata = dict(schema=1, java=runtime, os=platform.platform(), machine=platform.machine(), apiSHA256=sha(API), engineSourceManifest=json.loads((ROOT/"src/test/resources/independent-contracts/engine-source-manifest.json").read_text()), independentValidationSHA256=sha(ROOT/"src/test/resources/independent-contracts/validation.json"), jarSHA256=sha(jar), sourceManifestSHA256=source_digest, configSHA256=sha(ROOT/"src/main/resources/facet.properties"), rawEnergySHA256=report["rawSHA256"], battles=0, gradle="NOT RUN; delegated to parent cached Maven/Gradle integration", independentReview="supplied revision4 duel prerequisite passed; active melee independent review pending", premeasurementApproved=False, meleeEnabled=True, engineMotionFireAcceptance="866 supplied actual transitions replayed; prior four-round duel adapter report supplied, no new battles", memoryLimitMiB=256, deterministicPackaging=True)
    (evidence/"build.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print("OFFLINE BUILD VERIFIED source="+source_digest+" jar="+sha(jar))
if __name__ == "__main__":
    main()

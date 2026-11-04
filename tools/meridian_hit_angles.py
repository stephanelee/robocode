"""Registered v2 screen against a fresh v1 control, then conditional confirmation."""
import json
import statistics
import subprocess
from pathlib import Path

import benchmark
from meridian_duels import OUT, ROOT, play, save

REG = ROOT / "benchmarks/meridian-hit-angles"
BASE = "a19c20ff265cf79fc5d9aa2b9a57563d05f08d3c"


def baseline():
    """Materialize only the four verified runnable assets from the v1 evidence commit."""
    folder = ROOT / "build/meridian-v1-control"
    (folder / "bot").mkdir(parents=True)
    frozen = json.loads((ROOT / "benchmarks/meridian-v1/profile.json").read_text())["assets"]
    for filename in ("Meridian.jar", "bot/Meridian.json", "bot/Meridian.cmd", "bot/Meridian.sh"):
        source = "experiments/Meridian/" + filename
        path = folder / filename
        path.write_bytes(subprocess.check_output(["git", "show", BASE + ":" + source], cwd=ROOT))
        if benchmark.digest(path) != frozen[source]:
            raise ValueError("Control bundle differs from the registered v1 asset: " + filename)
    (folder / "bot/Meridian.sh").chmod(0o755)
    return folder


def passes_screen(rows):
    if len(rows) != 4 or not all(r["status"] == "valid" and r["skips"] == 0 for r in rows):
        return False
    control, diamond, retcon, ais = rows
    return (all(r["share"] > .35 and r["wins"] >= 1 for r in (diamond, retcon, ais))
            and diamond["share"] > control["share"]
            and diamond["damagePer100EnemyShots"] < control["damagePer100EnemyShots"])


def main():
    raw = (OUT / "profile.json").read_bytes()
    if raw != (REG / "profile.json").read_bytes():
        raise ValueError("Runtime differs from registration")
    profile = json.loads(raw)
    control = baseline()
    games = json.loads((REG / "games.json").read_text())
    report = {"games": [], "decision": "pending"}
    dest = OUT / "decision.json"
    if dest.exists():
        raise FileExistsError(dest)
    save(dest, report)
    for stage in ("screen", "confirmation"):
        rows = []
        for game in [g for g in games if g["stage"] == stage]:
            bundle = control if game["meridianVersion"] == "0.1.0" else ROOT / "experiments/Meridian"
            try:
                row = dict(**game, status="valid", **play(game, profile, bundle=bundle))
            except Exception as error:
                row = dict(**game, status="invalid", error=str(error))
            rows.append(row)
            report["games"].append(row)
            save(dest, report)
        if stage == "screen" and not passes_screen(rows):
            report["decision"] = "reject at screen"
            break
        if stage == "confirmation":
            confirmed = True
            for enemy in ("Diamond Port", "Retcon", "AISurvivor"):
                group = [r for r in rows if r["enemy"] == enemy]
                confirmed &= (len(group) == 3 and all(r["status"] == "valid" and r["skips"] == 0 for r in group)
                              and statistics.mean(r["share"] for r in group) > .50
                              and sum(r["wins"] for r in group) > 30
                              and sum(r["share"] > .50 for r in group) >= 2)
            report["decision"] = "confirmed" if confirmed else "reject after confirmation"
    save(dest, report)


if __name__ == "__main__":
    main()

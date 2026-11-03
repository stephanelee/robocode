"""Compare v4's firing energy per shot against a fresh, frozen v3 control."""
from meridian_hit_angles import ROOT, main

REG = ROOT / "benchmarks/meridian-economy"
BASE = "809b880cdcee98d7946d05e6bcd30f623f604348"


def passes_screen(rows):
    if len(rows) != 4 or not all(r["status"] == "valid" and r["skips"] == 0 for r in rows):
        return False
    control, diamond, retcon, ais = rows
    if not all(r["shots"] > 0 for r in rows):
        return False
    return (all(r["share"] > .35 and r["wins"] >= 1 for r in (diamond, retcon, ais))
            and diamond["share"] > control["share"]
            and diamond["shotEnergy"] / diamond["shots"] < control["shotEnergy"] / control["shots"])


if __name__ == "__main__":
    main(reg=REG, base=BASE, control_version="0.3.0", control_profile="meridian-aim",
         screen_check=passes_screen, additional_profile=REG / "profile-20261026.json")

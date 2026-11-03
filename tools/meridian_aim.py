"""Compare v3's outgoing damage per shot against a fresh, frozen v2 control."""
from meridian_hit_angles import ROOT, main

REG = ROOT / "benchmarks/meridian-aim"
BASE = "85cccdeb5f9bde8af42f971510e13f0e089f28ec"


def passes_screen(rows):
    if len(rows) != 4 or not all(r["status"] == "valid" and r["skips"] == 0 for r in rows):
        return False
    control, diamond, retcon, ais = rows
    if not all(r["shots"] > 0 for r in rows):
        return False
    return (all(r["share"] > .35 and r["wins"] >= 1 for r in (diamond, retcon, ais))
            and diamond["share"] > control["share"]
            and diamond["damageDealt"] / diamond["shots"] > control["damageDealt"] / control["shots"])


if __name__ == "__main__":
    main(reg=REG, base=BASE, control_version="0.2.0", control_profile="meridian-hit-angles",
         screen_check=passes_screen)

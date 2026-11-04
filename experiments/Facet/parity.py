#!/usr/bin/env python3
"""Compare executable/resource payloads; ZIP timestamps/order/directories are irrelevant."""
import hashlib, pathlib, sys, zipfile
def payload(path):
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        if len(names) != len(set(names)):
            raise SystemExit("duplicate ZIP entries: "+str(path))
        data = {n:z.read(n) for n in names if n != "META-INF/MANIFEST.MF"}
        manifest = z.read("META-INF/MANIFEST.MF").decode()
        if "Main-Class: Facet" not in manifest or "Multi-Release: true" not in manifest:
            raise SystemExit("manifest entry point/multi-release mismatch")
        for name, value in data.items():
            if name.endswith(".class") and not name.startswith("META-INF/versions/"):
                major = int.from_bytes(value[6:8], "big")
                if name.startswith(("Facet", "Core", "Physics", "Memory", "Learner", "Planner", "Radar", "Melee", "Waves", "Energy", "Geometry", "FireControl", "Diagnostics")) and major != 61:
                    raise SystemExit("bot class is not Java17: "+name)
        return data
if __name__ == "__main__":
    a,b = map(pathlib.Path,sys.argv[1:3])
    left,right = payload(a),payload(b)
    if left != right:
        differences = sorted(n for n in left.keys()|right.keys() if left.get(n) != right.get(n))
        raise SystemExit("JAR payload mismatch: "+", ".join(differences[:30]))
    print("PASS source/JAR payload parity entries="+str(len(left))+" jar="+hashlib.sha256(a.read_bytes()).hexdigest())

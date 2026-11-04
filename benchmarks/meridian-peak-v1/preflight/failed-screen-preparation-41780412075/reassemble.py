from pathlib import Path
import json,hashlib
p=Path(__file__).resolve().parent
m=json.loads((p/"receipt.json").read_text())
blocks=[]
for row in m["parts"]:
 b=(p/row["file"]).read_bytes()
 assert len(b)==row["bytes"] and hashlib.sha256(b).hexdigest()==row["sha256"]
 blocks.append(b)
b=b"".join(blocks)
assert len(b)==m["artifactBytes"] and hashlib.sha256(b).hexdigest()==m["artifactSHA256"]
(p/"artifact.zip").write_bytes(b)
print(m["artifactSHA256"])
